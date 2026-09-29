"""
Customer Repeat-Purchase Model — Training & Serialization
=========================================================
Trains the real scikit-learn model that ``app/main.py`` serves.

Everything here runs on the REAL UCI Online Retail data loaded by
``src/create_db.py`` (``data/processed/retail.db``) via the shared,
leakage-safe feature builder in ``src/features.py``. No synthetic rows.

Artifacts produced (all in ``model/``):
    customer_prediction_pipeline.joblib  — ONE sklearn Pipeline
                                           (StandardScaler -> classifier),
                                           so preprocessing + model ship as a
                                           single self-contained artifact.
    training_metrics.json                — honest held-out metrics + baselines
    feature_metadata.json                — feature order, frozen threshold and
           the training-set statistics the API uses to rank explanations.

Methodology
-----------
1. Build the per-customer table from the pre-cutoff window only.
2. Stratified 80/20 train/test split by customer.
3. 5-fold stratified CV on TRAIN to select among four candidate learners.
4. Decision threshold chosen on TRAIN (cross_val_predict) by maximising F1,
   then FROZEN. The test set is touched exactly once, at the frozen threshold,
   so the reported numbers are not threshold-tuned.
5. Baselines (majority class, always-positive F1) are reported next to the
   model metrics so the improvement is verifiable, not asserted.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

import features as feature_module  # noqa: E402  (needs SRC_DIR on path)

warnings.filterwarnings("ignore")

MODEL_DIR = PROJECT_ROOT / "model"
DB_PATH = PROJECT_ROOT / "data" / "processed" / "retail.db"
RETURNS_CSV = PROJECT_ROOT / "data" / "processed" / "returns.csv"

PIPELINE_FILE = "customer_prediction_pipeline.joblib"
METRICS_FILE = "training_metrics.json"
METADATA_FILE = "feature_metadata.json"

RANDOM_STATE = 42
SKFOLD = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def candidate_models() -> dict[str, Pipeline]:
    """The four learners compared by 5-fold CV ROC-AUC."""
    return {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        "RandomForestClassifier": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=12,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "ExtraTreesClassifier": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    ExtraTreesClassifier(
                        n_estimators=400,
                        max_depth=14,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "GradientBoostingClassifier": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=300,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.9,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }




def extract_importances(pipeline: Pipeline, feature_names: list[str]) -> dict[str, float]:
    """Return normalised 0-1 feature importances for whichever learner won."""
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        raw = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):
        raw = np.abs(np.asarray(clf.coef_, dtype=float)).ravel()
    else:  # pragma: no cover - every candidate exposes one of the above
        raw = np.ones(len(feature_names))
    total = raw.sum()
    raw = raw / total if total > 0 else raw
    return {n: round(float(v), 4) for n, v in zip(feature_names, raw)}


def select_threshold(y_true, proba: np.ndarray) -> float:
    """Pick the F1-maximising threshold on the training data, then freeze it."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        cur = f1_score(y_true, (proba >= t).astype(int))
        if cur > best_f1:
            best_f1, best_t = cur, float(t)
    return round(best_t, 3)


def score(y_true, proba: np.ndarray, threshold: float) -> dict:
    """Full metric block at a fixed decision threshold."""
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
    }


def extract_effects(
    pipeline: Pipeline, X_ref: "pd.DataFrame", feature_names: list[str]
) -> tuple[dict[str, float], str]:
    """
    Return SIGNED, normalised per-feature effects plus the method used.

    Importances are magnitude-only, which is not enough to explain a single
    prediction ("is a long gap good or bad?"). Effects keep the sign.

    - Linear models: use the fitted coefficients (exact, ceteris paribus on
      the standardised features).
    - Tree ensembles: no native sign, so measure the model's own marginal
      behaviour - mean predicted probability when the feature sits above its
      median vs at/below it. Documented as a marginal (not causal) estimate.
    """
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "coef_"):
        signed = np.asarray(clf.coef_, dtype=float).ravel()
        denom = np.abs(signed).sum()
        signed = signed / denom if denom > 0 else signed
        return {n: round(float(v), 4) for n, v in zip(feature_names, signed)}, "linear_coefficient"

    proba = pipeline.predict_proba(X_ref)[:, 1]
    effects: dict[str, float] = {}
    for c in feature_names:
        col = X_ref[c].to_numpy(dtype=float)
        med = float(np.median(col))
        above, below = col > med, col <= med
        if above.sum() == 0 or below.sum() == 0:
            effects[c] = 0.0
            continue
        effects[c] = float(proba[above].mean() - proba[below].mean())
    denom = max((abs(v) for v in effects.values()), default=0.0)
    if denom > 0:
        effects = {k: v / denom for k, v in effects.items()}
    return {k: round(float(v), 4) for k, v in effects.items()}, "marginal_proba_shift"


def campaign_metrics(y_true, proba: np.ndarray, fraction: float = 0.2) -> dict:
    """
    Precision / lift when only the top `fraction` of customers are contacted.

    This is the decision-relevant metric for a win-back campaign: the business
    has limited outreach, so what matters is how much better a ranked list is
    than a random list, not the F1 at an arbitrary 0.5 cut.

    F1 is a poor headline for this problem because the positive class is the
    MAJORITY (~56%), so "predict repeat for everyone" already scores F1 ~0.72.
    Lift makes the real signal visible and comparable to random targeting.
    """
    y = np.asarray(y_true)
    p = np.asarray(proba)
    n = len(y)
    k = max(1, int(round(n * fraction)))
    top = np.argsort(-p)[:k]
    precision_top = float(y[top].mean())
    base_rate = float(y.mean())
    return {
        "contact_fraction": fraction,
        "customers_contacted": int(k),
        "precision_at_top": round(precision_top, 4),
        "random_targeting_precision": round(base_rate, 4),
        "lift_vs_random": round(
            precision_top / base_rate if base_rate > 0 else 0.0, 4
        ),
        "repeaters_captured_pct": round(
            float(y[top].sum()) / float(y.sum()) * 100 if y.sum() else 0.0, 2
        ),
    }



def train_model() -> dict:
    feature_names = list(feature_module.FEATURE_COLUMNS)
    target = feature_module.TARGET_COLUMN

    print("=" * 64)
    print("  Customer Repeat-Purchase Model - Training Pipeline")
    print("=" * 64)

    if not DB_PATH.exists():
        raise SystemExit(
            f"Missing database: {DB_PATH}\n"
            "Run `python run_pipeline.py` first (download_data -> clean -> "
            "create_db), or those three scripts individually."
        )

    # ---- 1. Real data, leakage-safe features ----
    conn = sqlite3.connect(DB_PATH)
    table, transactions, returns = feature_module.build_modelling_table(
        conn, str(RETURNS_CSV)
    )
    conn.close()

    X = table[feature_names]
    y = table[target]
    pos_rate = float(y.mean())

    print(f"\nTransactions used        : {len(transactions):,}")
    print(f"Customers (rows)         : {len(table):,}")
    print(f"Return rows (all time)   : {len(returns):,}")
    print(f"Positive rate (repeat)   : {pos_rate:.4f}")

    # ---- 2. Stratified customer-level split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train / Test customers   : {len(X_train):,} / {len(X_test):,}")

    # ---- 3. Model selection by 5-fold CV ROC-AUC on TRAIN only ----
    print("\n--- Model selection (5-fold CV on train, ROC-AUC) ---")
    results: dict[str, dict[str, float]] = {}
    for name, pipe in candidate_models().items():
        auc_scores = cross_val_score(
            pipe, X_train, y_train, cv=SKFOLD, scoring="roc_auc", n_jobs=-1
        )
        results[name] = {
            "cv_roc_auc_mean": round(float(auc_scores.mean()), 4),
            "cv_roc_auc_std": round(float(auc_scores.std()), 4),
        }
        print(f"  {name:<28} {auc_scores.mean():.4f} +/- {auc_scores.std():.4f}")

    best_name = max(results, key=lambda k: results[k]["cv_roc_auc_mean"])
    print(f"\nSelected model           : {best_name}")

    # ---- 4. Freeze threshold from out-of-fold predictions on TRAIN ----
    best_pipe = candidate_models()[best_name]
    oof_proba = cross_val_predict(
        best_pipe, X_train, y_train, cv=SKFOLD, method="predict_proba", n_jobs=-1
    )[:, 1]
    threshold = select_threshold(y_train, oof_proba)
    print(f"Frozen threshold (OOF)   : {threshold}")

    # ---- 5. Final fit on TRAIN, single evaluation on held-out TEST ----
    best_pipe.fit(X_train, y_train)
    test_proba = best_pipe.predict_proba(X_test)[:, 1]
    test_metrics = score(y_test, test_proba, threshold)

    majority_acc = max(float(y_test.mean()), 1.0 - float(y_test.mean()))
    always_yes_f1 = float(f1_score(y_test, np.ones(len(y_test), dtype=int)))
    baselines = {
        "majority_class_accuracy": round(majority_acc, 4),
        "always_predict_repeat_f1": round(always_yes_f1, 4),
    }

    print("\n--- Held-out test metrics (frozen threshold) ---")
    for k, v in test_metrics.items():
        print(f"  {k:<24}: {v}")
    print("--- Baselines the model must beat ---")
    for k, v in baselines.items():
        print(f"  {k:<24}: {v}")

    campaigns = {
        "top_10pct": campaign_metrics(y_test, test_proba, 0.10),
        "top_20pct": campaign_metrics(y_test, test_proba, 0.20),
        "top_30pct": campaign_metrics(y_test, test_proba, 0.30),
    }
    print("\n--- Campaign targeting on held-out test (rank by probability) ---")
    for label, c in campaigns.items():
        print(
            f"  {label}: contact {c['customers_contacted']:>4}  "
            f"precision {c['precision_at_top']:.3f}  "
            f"vs random {c['random_targeting_precision']:.3f}  "
            f"lift {c['lift_vs_random']:.2f}x  "
            f"captured {c['repeaters_captured_pct']:.1f}% of repeaters"
        )

    return {
        "best_name": best_name,
        "best_pipe": best_pipe,
        "results": results,
        "threshold": threshold,
        "test_metrics": test_metrics,
        "campaign_metrics": campaigns,
        "baselines": baselines,

        "pos_rate": pos_rate,
        "table": table,
        "X_train": X_train,
        "feature_names": feature_names,
        "target": target,
    }




def save_artifacts(ctx: dict) -> None:
    """Write the single pipeline artifact plus metrics / metadata JSON."""
    best_pipe: Pipeline = ctx["best_pipe"]
    feature_names = ctx["feature_names"]
    X_train = ctx["X_train"]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, MODEL_DIR / PIPELINE_FILE)

    feature_stats = {
        c: {
            "mean": round(float(X_train[c].mean()), 6),
            "std": round(float(X_train[c].std() or 1.0), 6),
            "median": round(float(X_train[c].median()), 6),
            "min": round(float(X_train[c].min()), 6),
            "max": round(float(X_train[c].max()), 6),
        }
        for c in feature_names
    }

    effects, effect_method = extract_effects(best_pipe, X_train, feature_names)

    metadata = {
        "feature_order": feature_names,
        "feature_stats": feature_stats,
        "feature_importances": extract_importances(best_pipe, feature_names),
        "feature_effects": effects,
        "effect_method": effect_method,
        "threshold": ctx["threshold"],
        "model_type": ctx["best_name"],
        "trained_on": {
            "database": str(DB_PATH.relative_to(PROJECT_ROOT)),
            "cutoff_date": str(feature_module.CUTOFF_DATE.date()),
            "target_window_end": str(feature_module.TARGET_END.date()),
            "n_customers": int(len(ctx["table"])),
        },
    }

    metrics = {
        "model_type": ctx["best_name"],
        "target": ctx["target"],
        "target_definition": (
            "Customer made at least one purchase between "
            f"{feature_module.CUTOFF_DATE.date()} and "
            f"{feature_module.TARGET_END.date()} (exclusive) — a full quarter."
        ),
        "n_customers": int(len(ctx["table"])),
        "n_features": len(feature_names),
        "positive_rate": round(ctx["pos_rate"], 4),
        "threshold": ctx["threshold"],
        "cv_roc_auc_mean": ctx["results"][ctx["best_name"]]["cv_roc_auc_mean"],
        "cv_roc_auc_std": ctx["results"][ctx["best_name"]]["cv_roc_auc_std"],
        "test_metrics": ctx["test_metrics"],
        "campaign_metrics": ctx["campaign_metrics"],
        "baselines": ctx["baselines"],
        "model_selection_cv_roc_auc": {
            k: v["cv_roc_auc_mean"] for k, v in ctx["results"].items()
        },
    }

    with open(MODEL_DIR / METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(MODEL_DIR / METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n--- Saved artifacts ---")
    for fname in (PIPELINE_FILE, METRICS_FILE, METADATA_FILE):
        path = MODEL_DIR / fname
        print(f"  model/{fname:<38} {path.stat().st_size / 1024:.1f} KB")

    print("\nTop feature importances")
    ranked = sorted(
        metadata["feature_importances"].items(), key=lambda kv: kv[1], reverse=True
    )
    for name, imp in ranked[:5]:
        print(f"  {name:<22} {imp:.3f}")

    print("\nTraining complete.")


if __name__ == "__main__":
    save_artifacts(train_model())
