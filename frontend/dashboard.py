"""
Customer Repeat-Purchase Dashboard - Streamlit Frontend
=======================================================
Talks to the FastAPI service in ``app/main.py``. Nothing is predicted locally:
every number on screen comes from the API, so the dashboard cannot drift from
the served model.

Run:  streamlit run frontend/dashboard.py
Set a non-default backend with:  set API_URL=http://localhost:8000
"""

import os
import io

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 15

st.set_page_config(
    page_title="Repeat Purchase Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
  .stApp { font-family: 'Inter', sans-serif; }

  .main-header {
      background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
      padding: 2.2rem 2rem; border-radius: 16px; margin-bottom: 1.6rem;
      text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.3);
  }
  .main-header h1 { color:#fff; font-size:2.2rem; font-weight:700; margin:0 0 .4rem 0; }
  .main-header p  { color:#a0aec0; font-size:1.02rem; margin:0; }

  .result-card { padding: 1.8rem; border-radius: 14px; margin: 1.2rem 0;
                 box-shadow: 0 4px 24px rgba(0,0,0,0.25); }
  .result-yes  { background: linear-gradient(135deg,#0d1f0d,#133a13);
                 border-left: 5px solid #3fb950; }
  .result-no   { background: linear-gradient(135deg,#2d1117,#4a1520);
                 border-left: 5px solid #f85149; }
  .result-title { font-size:1.45rem; font-weight:700; margin-bottom:.4rem; }

  .driver { background: rgba(255,255,255,0.05); border-left: 3px solid #f0883e;
            padding:.55rem 1rem; border-radius:6px; margin:.35rem 0;
            font-size:.9rem; color:#e2e8f0; }

  .probability-bar { height:12px; border-radius:6px; background:#2d3748;
                     overflow:hidden; margin:.5rem 0; }
  .probability-fill { height:100%; border-radius:6px; }

  .caveat { background: rgba(240,136,62,0.10); border-left:4px solid #f0883e;
            padding:.9rem 1.1rem; border-radius:8px; font-size:.9rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=15, show_spinner=False)
def api_health() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=TIMEOUT)
        return r.json() if r.status_code == 200 else None
    except requests.RequestException:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def api_metrics() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/metrics", timeout=TIMEOUT)
        return r.json() if r.status_code == 200 else None
    except requests.RequestException:
        return None


def api_predict(payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=TIMEOUT)
    except requests.RequestException:
        st.error("Cannot reach the API. Start it with: `uvicorn app.main:app --reload`")
        return None
    if r.status_code == 200:
        return r.json()
    st.error(f"API error {r.status_code}: {r.text[:400]}")
    return None


def api_predict_batch(records: list[dict]) -> list[dict]:
    """Send customers in <=500-record chunks (the API's documented limit)."""
    out: list[dict] = []
    for i in range(0, len(records), 500):
        chunk = records[i : i + 500]
        try:
            r = requests.post(
                f"{API_URL}/predict/batch", json={"customers": chunk}, timeout=120
            )
        except requests.RequestException:
            st.error("Cannot reach the API for batch prediction.")
            return []
        if r.status_code != 200:
            st.error(f"API error {r.status_code}: {r.text[:400]}")
            return []
        out.extend(r.json()["predictions"])
    return out


# ---------------------------------------------------------------------------
# Input specification - mirrors app/schemas.py::FEATURE_ORDER
# ---------------------------------------------------------------------------
# (key, label, min, max, default, step, group, help)
FIELDS: list[tuple] = [
    ("tenure_days", "Tenure (days since first order)", 0, 36500, 184, 1, "Timing",
     "How long the customer has been buying, measured at the scoring cutoff."),
    ("recency_days", "Recency (days since last order)", 0, 36500, 72, 1, "Timing",
     "Lower is more recent. One of the strongest behavioural signals."),
    ("months_active", "Months with purchase activity", 0, 1200, 2, 1, "Timing",
     "Count of distinct calendar months containing at least one order."),
    ("max_gap_days", "Longest gap between orders (days)", 0, 36500, 34, 1, "Timing",
     "Longest silent stretch inside the customer's own history."),
    ("frequency", "Number of orders", 1, 100000, 2, 1, "Value & volume",
     "Distinct orders placed in the history window."),
    ("monetary", "Total spend (GBP)", 0.0, 100000000.0, 559.55, 10.0, "Value & volume",
     "Sum of order value across the history window."),
    ("avg_order_value", "Average order value (GBP)", 0.0, 100000000.0, 285.02, 5.0,
     "Value & volume", "Total spend divided by number of orders."),
    ("total_items", "Total items bought", 0, 10000000, 318, 5, "Value & volume",
     "Units purchased in total."),
    ("avg_items_per_order", "Items per order", 0.0, 1000000.0, 152.75, 1.0,
     "Value & volume", "Basket size in units."),
    ("distinct_products", "Distinct products bought", 0, 1000000, 29, 1, "Value & volume",
     "Breadth of catalogue the customer has explored."),
    ("avg_unit_price", "Average unit price paid (GBP)", 0.0, 100000.0, 2.93, 0.1,
     "Value & volume", "Mean price per unit the customer pays."),
    ("returns_rate", "Return / cancellation rate", 0.0, 1.0, 0.0, 0.01, "Returns",
     "Cancelled invoices divided by all invoices. 0 = never returned, 1 = all returns."),
]

PRESETS: dict[str, dict] = {
    "Custom": {},
    "Loyal repeat buyer": {
        "tenure_days": 260, "recency_days": 8, "months_active": 8, "max_gap_days": 21,
        "frequency": 12, "monetary": 8200.0, "avg_order_value": 683.0,
        "total_items": 2400, "avg_items_per_order": 200.0, "distinct_products": 140,
        "avg_unit_price": 3.4, "returns_rate": 0.03,
    },
    "One-off browser": {
        "tenure_days": 180, "recency_days": 168, "months_active": 1, "max_gap_days": 0,
        "frequency": 1, "monetary": 95.0, "avg_order_value": 95.0, "total_items": 12,
        "avg_items_per_order": 12.0, "distinct_products": 3, "avg_unit_price": 7.9,
        "returns_rate": 0.0,
    },
    "Heavy returns / at risk": {
        "tenure_days": 200, "recency_days": 95, "months_active": 3, "max_gap_days": 90,
        "frequency": 4, "monetary": 1200.0, "avg_order_value": 300.0,
        "total_items": 400, "avg_items_per_order": 100.0, "distinct_products": 18,
        "avg_unit_price": 3.0, "returns_rate": 0.45,
    },
}

# Apply the preset by seeding widget state BEFORE the widgets are created.
chosen_preset = st.session_state.get("preset", "Custom")
if st.session_state.get("applied_preset") != chosen_preset:
    for key, value in PRESETS.get(chosen_preset, {}).items():
        st.session_state[key] = value
    st.session_state["applied_preset"] = chosen_preset


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="main-header">
  <h1>🛍️ Repeat Purchase Intelligence</h1>
  <p>Will this customer order again next quarter? — trained on real
     UCI Online Retail transactions</p>
</div>
""",
    unsafe_allow_html=True,
)

health = api_health()

with st.sidebar:
    st.markdown("### System status")
    if health:
        st.success("API online")
        st.caption(f"Model: **{health.get('model_type') or 'unknown'}**")
        st.caption(f"Features: **{health.get('feature_count')}**")
    else:
        st.error("API offline")
        st.code(f"uvicorn app.main:app --port {API_URL.rsplit(':', 1)[-1]}", language="bash")
        st.stop()

    metrics = api_metrics()
    if metrics:
        st.markdown("### Model performance")
        test = metrics.get("test_metrics", {})
        base = metrics.get("baselines", {})
        st.metric("ROC-AUC", f"{test.get('roc_auc', 0):.3f}")
        st.metric("Accuracy", f"{test.get('accuracy', 0):.1%}",
                  delta=f"vs {base.get('majority_class_accuracy', 0):.1%} majority baseline")
        st.metric("Recall", f"{test.get('recall', 0):.1%}")
        lift = metrics.get("campaign_metrics", {}).get("top_20pct", {})
        st.metric("Lift @ top 20%", f"{lift.get('lift_vs_random', 0):.2f}x",
                  help="Precision when contacting the top 20% of customers, "
                       "divided by random targeting.")
        st.caption(f"Decision threshold: **{metrics.get('threshold')}**")
        st.caption(f"Customers trained on: **{metrics.get('n_customers'):,}**")

tab_single, tab_batch, tab_report, tab_api = st.tabs(
    ["🔮 Single prediction", "📤 Batch", "📊 Model report", "🔌 API reference"]
)

# --------------------------------------------------------------- single -----
with tab_single:
    st.radio(
        "Start from a profile",
        list(PRESETS.keys()),
        key="preset",
        horizontal=True,
        help="Presets just fill the form; everything stays editable.",
    )

    with st.form("customer_form"):
        st.markdown("#### Customer history at the scoring date")
        payload: dict = {}
        for group in ("Timing", "Value & volume", "Returns"):
            st.markdown(f"**{group}**")
            fields = [f for f in FIELDS if f[6] == group]
            for row in (fields[i : i + 3] for i in range(0, len(fields), 3)):
                cols = st.columns(len(row))
                for col, (key, label, lo, hi, default, step, _, help_text) in zip(
                    cols, row
                ):
                    with col:
                        payload[key] = st.number_input(
                            label, min_value=lo, max_value=hi, value=default,
                            step=step, key=key, help=help_text,
                        )
        submitted = st.form_submit_button("Predict repeat purchase", type="primary")

    if submitted:
        result = api_predict(payload)
        if result:
            repeat = result["will_repeat"]
            prob = result["repeat_probability"]
            colour = "#3fb950" if repeat else "#f85149"
            st.markdown(
                f"""<div class="result-card {'result-yes' if repeat else 'result-no'}">
  <div class="result-title">{'🔁 Will buy again' if repeat else '🚪 Unlikely to return'}</div>
  <div>Next-purchase probability
       <strong style="font-size:1.3rem">{prob:.1%}</strong>
       &nbsp;·&nbsp; tier: {result['engagement_tier']}
       &nbsp;·&nbsp; threshold: {result['threshold']}</div>
  <div class="probability-bar">
    <div class="probability-fill" style="width:{prob * 100:.1f}%;background:{colour}"></div>
  </div>
</div>""",
                unsafe_allow_html=True,
            )
            st.markdown("#### Why the model decided this")
            st.caption(
                "Ranked by the model's own feature importances; the direction "
                "comes from its fitted coefficients, compared with the training "
                "average for each feature."
            )
            for driver in result["key_drivers"]:
                st.markdown(
                    f'<div class="driver">{driver}</div>', unsafe_allow_html=True
                )
            with st.expander("Raw response"):
                st.json(result)

# ---------------------------------------------------------------- batch -----
with tab_batch:
    st.markdown(
        f"Upload a CSV with exactly these columns: `{', '.join(f[0] for f in FIELDS)}`."
    )
    upload = st.file_uploader("Customer CSV", type=["csv"])
    if upload is not None:
        try:
            frame = pd.read_csv(upload)
        except Exception as exc:  # noqa: BLE001 - surface any parse error verbatim
            st.error(f"Could not parse CSV: {exc}")
            frame = None
        if frame is not None:
            missing = [f[0] for f in FIELDS if f[0] not in frame.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                st.dataframe(frame[list(dict.fromkeys([f[0] for f in FIELDS]))].head(10))
                st.caption(f"{len(frame):,} rows ready.")
                if st.button("Score this file", type="primary"):
                    recs = frame[[f[0] for f in FIELDS]].to_dict("records")
                    preds = api_predict_batch(recs)
                    if preds:
                        out = pd.DataFrame(
                            {
                                "repeat_probability": [p["repeat_probability"] for p in preds],
                                "prediction": [p["prediction"] for p in preds],
                                "engagement_tier": [p["engagement_tier"] for p in preds],
                            }
                        )
                        merged = pd.concat([frame.reset_index(drop=True), out], axis=1)
                        merged = merged.sort_values(
                            "repeat_probability", ascending=False
                        ).reset_index(drop=True)
                        st.success(
                            f"Scored {len(merged):,} customers. Ranked by "
                            "likelihood to buy again — contact from the top down."
                        )
                        st.dataframe(merged.head(200))
                        st.download_button(
                            "Download ranked CSV",
                            merged.to_csv(index=False).encode(),
                            file_name="scored_customers.csv",
                            mime="text/csv",
                        )

# --------------------------------------------------------------- report -----
@st.cache_data(ttl=60, show_spinner=False)
def api_features() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/features", timeout=TIMEOUT)
        return r.json() if r.status_code == 200 else None
    except requests.RequestException:
        return None


with tab_report:
    if not metrics:
        st.info("Model report unavailable — the API could not read training_metrics.json.")
    else:
        st.markdown("#### What the model predicts")
        st.write(metrics.get("target_definition", ""))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers", f"{metrics.get('n_customers', 0):,}")
        c2.metric("Repeat rate in data", f"{metrics.get('positive_rate', 0):.1%}")
        c3.metric("Model", metrics.get("model_type", ""))
        c4.metric("ROC-AUC (test)", f"{metrics['test_metrics']['roc_auc']:.3f}")

        st.markdown("#### Model selection — 5-fold CV ROC-AUC on training data")
        cv = metrics.get("model_selection_cv_roc_auc", {})
        chosen = metrics.get("model_type")
        st.bar_chart(pd.Series(cv).rename_axis("candidate").sort_values())
        st.caption(
            f"Highest CV ROC-AUC was selected: **{chosen}**. The decision "
            "threshold was then frozen on out-of-fold training predictions only "
            "— the test set was used once, for the numbers below."
        )

        st.markdown("#### Held-out test set vs baselines")
        test, base = metrics["test_metrics"], metrics.get("baselines", {})
        comparison = pd.DataFrame(
            [
                {"metric": "Accuracy", "model": test["accuracy"],
                 "trivial_baseline": base.get("majority_class_accuracy")},
                {"metric": "F1 (will repeat)", "model": test["f1_score"],
                 "trivial_baseline": base.get("always_predict_repeat_f1")},
                {"metric": "ROC-AUC", "model": test["roc_auc"], "trivial_baseline": 0.5},
            ]
        )
        st.dataframe(comparison, hide_index=True)
        st.markdown(
            '<div class="caveat"><strong>Read this honestly.</strong> Because '
            f"{metrics.get('positive_rate', 0):.0%} of customers do repeat, "
            '"predict repeat for everyone" already scores F1 ≈ '
            f"{base.get('always_predict_repeat_f1', 0):.2f}. F1 is therefore a "
            "weak headline here. The lift table below is the decision-relevant "
            "measure: it shows how much better targeted outreach is than random.</div>",
            unsafe_allow_html=True,
        )

        st.markdown("#### Campaign targeting on the held-out test set")
        campaigns = metrics.get("campaign_metrics", {})
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "contact top": label.replace("top_", "").replace("pct", "%"),
                        "customers": c["customers_contacted"],
                        "precision": c["precision_at_top"],
                        "random": c["random_targeting_precision"],
                        "lift": c["lift_vs_random"],
                        "% of repeaters reached": c["repeaters_captured_pct"],
                    }
                    for label, c in campaigns.items()
                ]
            ),
            hide_index=True,
        )

        feats = api_features()
        if feats:
            st.markdown("#### What actually moves the model")
            effects = feats.get("feature_effects", {})
            order = feats.get("feature_order", list(effects))
            st.bar_chart(
                pd.Series({k: effects.get(k, 0.0) for k in order})
                .rename_axis("feature")
                .sort_values()
            )
            st.caption(
                "Signed effect: right = increases the chance of a repeat purchase, "
                "left = decreases it. Method: "
                f"`{feats.get('effect_method', 'n/a')}`. Magnitude-only "
                "importances are available from `GET /features`."
            )

# ------------------------------------------------------------------ api -----
with tab_api:
    st.markdown(
        f"""
| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Liveness, loaded model, key metrics |
| GET | `/metrics` | Full training report incl. baselines and lift |
| GET | `/features` | Importances, signed effects, training distributions |
| POST | `/predict` | One customer → repeat probability + drivers |
| POST | `/predict/batch` | Up to 500 customers, vectorised |
| GET | `/docs` | Interactive OpenAPI UI |

**Example request**

```bash
curl -X POST {API_URL}/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"tenure_days": 250, "recency_days": 40, "frequency": 6,
        "monetary": 1850.0, "avg_order_value": 308.33, "total_items": 720,
        "avg_items_per_order": 120.0, "distinct_products": 45,
        "avg_unit_price": 2.9, "months_active": 5, "max_gap_days": 62,
        "returns_rate": 0.05}}'
```

Interactive docs: <{API_URL}/docs>
        """
    )


