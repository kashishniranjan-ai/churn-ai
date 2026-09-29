"""
Customer Feature Engineering — single source of truth
=====================================================
Builds the customer-level modelling table used by BOTH
``src/train_model.py`` (training / evaluation) and ``src/analysis.py``
(the exploratory repeat-purchase section).

Prediction task
---------------
"Will this customer make at least one purchase in the next quarter?"

The dataset spans 2010-12-01 .. 2011-12-09, so the final partial month
(December 2011, only 9 days of it) is unusable as a scoring window. We
therefore define two non-overlapping windows:

    feature window : 2010-12-01 .. 2011-08-31   (strictly before CUTOFF_DATE)
    target window  : 2011-09-01 .. 2011-11-30   (a full quarter)

Every feature is computed ONLY from transactions that happened strictly
before CUTOFF_DATE, and the label is derived ONLY from transactions on or
after CUTOFF_DATE. Because the windows are temporally disjoint, no feature
can observe the label -> no temporal leakage.

Scope / known limitation
------------------------
A customer must have at least one pre-cutoff purchase to receive any
features, so the resulting model only covers "known" customers. A brand new
visitor with no history cannot be scored, and is excluded here. This is
documented in the README as a limitation.
"""

from __future__ import annotations

import pandas as pd

# --- Modelling windows (fixed, so training & inference stay comparable) ---
CUTOFF_DATE = pd.Timestamp("2011-09-01")
TARGET_END = pd.Timestamp("2011-12-01")

# Ordered list of model input features. Keep this in sync with
# app/schemas.py::CustomerInput — the API validates against it.
FEATURE_COLUMNS: list[str] = [
    "tenure_days",
    "recency_days",
    "frequency",
    "monetary",
    "avg_order_value",
    "total_items",
    "avg_items_per_order",
    "distinct_products",
    "avg_unit_price",
    "months_active",
    "max_gap_days",
    "returns_rate",
]

TARGET_COLUMN = "will_repeat"

TRANSACTIONS_SQL = """
SELECT CustomerID, InvoiceNo, StockCode, Quantity, UnitPrice, TotalPrice, InvoiceDate
FROM transactions
WHERE strftime('%Y-%m', InvoiceDate) < '2011-12'
"""


def _to_frame(conn, sql: str) -> pd.DataFrame:
    """Run a SQL query and parse InvoiceDate into a real datetime column."""
    df = pd.read_sql_query(sql, conn)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def load_transactions(conn) -> pd.DataFrame:
    """All cleaned transactions, excluding the partial final month."""
    return _to_frame(conn, TRANSACTIONS_SQL)


def load_returns(returns_csv: str) -> pd.DataFrame:
    """
    Load the cancellation rows preserved by ``src/clean.py``.

    Only InvoiceDate and CustomerID matter here (we want a per-customer
    cancellation *rate*), so a missing file degrades gracefully to an empty
    frame rather than failing the whole pipeline.
    """
    try:
        df = pd.read_csv(returns_csv, encoding="ISO-8859-1")
    except FileNotFoundError:
        return pd.DataFrame(columns=["CustomerID", "InvoiceDate"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df



def build_customer_features(
    transactions: pd.DataFrame,
    returns: pd.DataFrame | None = None,
    cutoff: pd.Timestamp = CUTOFF_DATE,
    target_end: pd.Timestamp = TARGET_END,
) -> pd.DataFrame:
    """
    Build the per-customer modelling table.

    Returns a DataFrame with one row per customer containing ``CustomerID``,
    every column in ``FEATURE_COLUMNS`` (computed from ``transactions`` rows
    before ``cutoff``) and the ``will_repeat`` label (from rows in
    ``[cutoff, target_end)``).
    """
    feats_src = transactions[transactions["InvoiceDate"] < cutoff].copy()
    if feats_src.empty:
        raise ValueError(f"No transactions found before cutoff {cutoff.date()}")

    grp = feats_src.groupby("CustomerID")

    base = grp.agg(
        first_date=("InvoiceDate", "min"),
        last_date=("InvoiceDate", "max"),
        frequency=("InvoiceNo", "nunique"),
        monetary=("TotalPrice", "sum"),
        total_items=("Quantity", "sum"),
        distinct_products=("StockCode", "nunique"),
        avg_unit_price=("UnitPrice", "mean"),
    ).reset_index()

    # --- Tenure / recency, both measured against the cutoff, not "today" ---
    base["tenure_days"] = (cutoff - base["first_date"]).dt.days
    base["recency_days"] = (cutoff - base["last_date"]).dt.days

    # --- Derived ratios ---
    base["avg_order_value"] = base["monetary"] / base["frequency"]
    base["avg_items_per_order"] = base["total_items"] / base["frequency"]

    # --- How many distinct calendar months did the customer buy in? ---
    feats_src["_month"] = (
        feats_src["InvoiceDate"].dt.year * 12 + feats_src["InvoiceDate"].dt.month
    )
    months_active = feats_src.groupby("CustomerID")["_month"].nunique()
    base = base.merge(
        months_active.rename("months_active").reset_index(), on="CustomerID", how="left"
    )
    base["months_active"] = base["months_active"].fillna(0).astype(int)

    # --- Longest gap between consecutive purchase days (dormancy signal) ---
    ordered = feats_src[["CustomerID", "InvoiceDate"]].drop_duplicates()
    ordered = ordered.sort_values(["CustomerID", "InvoiceDate"])
    ordered["_gap"] = ordered.groupby("CustomerID")["InvoiceDate"].diff()
    max_gap = ordered.groupby("CustomerID")["_gap"].max().dt.days
    base = base.merge(
        max_gap.rename("max_gap_days").reset_index(), on="CustomerID", how="left"
    )
    # Customers with a single purchase date have no gap at all.
    base["max_gap_days"] = base["max_gap_days"].fillna(0).astype(int)

    # --- Return / cancellation rate, using ONLY pre-cutoff cancellations ---
    if returns is None or returns.empty:
        base["cancellations"] = 0.0
    else:
        pre_returns = returns[returns["InvoiceDate"] < cutoff]
        cancels = (
            pre_returns.groupby("CustomerID").size().rename("cancellations").reset_index()
        )
        base = base.merge(cancels, on="CustomerID", how="left")
        base["cancellations"] = base["cancellations"].fillna(0.0)

    # Rate over all commercial events (completed orders + cancellations),
    # so the feature is bounded on [0, 1] and comparable across customers.
    base["returns_rate"] = base["cancellations"] / (
        base["cancellations"] + base["frequency"]
    )
    base = base.drop(columns=["cancellations", "first_date", "last_date"])

    # --- Target: any purchase inside the scoring quarter ---
    target_src = transactions[
        (transactions["InvoiceDate"] >= cutoff)
        & (transactions["InvoiceDate"] < target_end)
    ]
    repeaters = set(target_src["CustomerID"].unique())
    base[TARGET_COLUMN] = base["CustomerID"].isin(repeaters).astype(int)

    base["monetary"] = base["monetary"].round(2)
    base["avg_order_value"] = base["avg_order_value"].round(2)
    base["avg_unit_price"] = base["avg_unit_price"].round(4)
    base["avg_items_per_order"] = base["avg_items_per_order"].round(4)
    base["returns_rate"] = base["returns_rate"].round(6)

    ordered_cols = ["CustomerID", *FEATURE_COLUMNS, TARGET_COLUMN]
    return base[ordered_cols]


def build_modelling_table(
    conn, returns_csv: str = "data/processed/returns.csv"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper returning ``(feature_table, transactions, returns)``."""
    transactions = load_transactions(conn)
    returns = load_returns(returns_csv)
    table = build_customer_features(transactions, returns)
    return table, transactions, returns
