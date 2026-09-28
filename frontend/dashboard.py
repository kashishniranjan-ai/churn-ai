"""
Customer Churn Predictor — Streamlit Frontend
==============================================
A rich, interactive dashboard that communicates with the FastAPI
backend. Features customer input forms, real-time predictions,
risk visualisations, and batch upload capability.
"""

import json
import time

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Churn Predictor — AI Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a premium look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ---- Global ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ---- Header ---- */
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #a0aec0;
        font-size: 1.05rem;
        margin: 0;
    }

    /* ---- Metric cards ---- */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 1.6rem;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.35);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #a0aec0;
        margin-top: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ---- Result card ---- */
    .result-card {
        padding: 2rem;
        border-radius: 14px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    }
    .result-churn {
        background: linear-gradient(135deg, #2d1117, #4a1520);
        border-left: 5px solid #f85149;
    }
    .result-no-churn {
        background: linear-gradient(135deg, #0d1f0d, #133a13);
        border-left: 5px solid #3fb950;
    }
    .result-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    /* ---- Risk factors ---- */
    .risk-factor {
        background: rgba(255,255,255,0.05);
        border-left: 3px solid #f0883e;
        padding: 0.6rem 1rem;
        border-radius: 6px;
        margin: 0.4rem 0;
        font-size: 0.9rem;
        color: #e2e8f0;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1a1a2e);
    }

    /* ---- Progress bar ---- */
    .probability-bar {
        height: 12px;
        border-radius: 6px;
        background: #2d3748;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .probability-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.6s ease;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def check_api_health() -> dict | None:
    """Ping the backend health endpoint."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        if r.status_code == 200:
            return r.json()
    except requests.ConnectionError:
        return None
    return None


def make_prediction(payload: dict) -> dict | None:
    """POST customer data and return the prediction response."""
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"API Error {r.status_code}: {r.text}")
            return None
    except requests.ConnectionError:
        st.error("⚠️ Cannot reach backend API. Is the FastAPI server running?")
        return None


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

# ---- Header ----
st.markdown("""
<div class="main-header">
    <h1>🔮 Customer Churn Predictor</h1>
    <p>AI-powered retention intelligence — predict, understand, and prevent customer churn</p>
</div>
""", unsafe_allow_html=True)

# ---- Sidebar ----
with st.sidebar:
    st.markdown("## ⚙️ System Status")

    health = check_api_health()
    if health:
        st.success("🟢 API Online")
        if health.get("training_metrics"):
            m = health["training_metrics"]
            st.markdown("### 📊 Model Performance")

            cols = st.columns(2)
            cols[0].metric("Accuracy", f"{m['accuracy']:.1%}")
            cols[1].metric("F1 Score", f"{m['f1_score']:.1%}")

            cols2 = st.columns(2)
            cols2[0].metric("Precision", f"{m['precision']:.1%}")
            cols2[1].metric("Recall", f"{m['recall']:.1%}")

            st.metric("ROC AUC", f"{m['roc_auc']:.1%}")
    else:
        st.error("🔴 API Offline")
        st.info(
            "Start the backend:\n\n"
            "```bash\n"
            "uvicorn app.main:app --reload\n"
            "```"
        )

    st.markdown("---")
    st.markdown("### 📖 Quick Guide")
    st.markdown(
        "1. Fill in customer details\n"
        "2. Click **Predict Churn**\n"
        "3. Review risk analysis\n"
        "4. Use batch upload for CSV files"
    )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#718096; font-size:0.8rem;'>"
        "Built with FastAPI + Streamlit<br>v1.0.0</div>",
        unsafe_allow_html=True,
    )

# ---- Tabs ----
tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📦 Batch Upload", "📚 API Docs"])

# ---------------------------------------------------------------------------
# TAB 1 — Single Prediction
# ---------------------------------------------------------------------------
with tab1:
    st.markdown("### 👤 Customer Profile")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No", key="senior")
        tenure = st.slider("Tenure (months)", 0, 72, 24, key="tenure")

    with col2:
        st.markdown("**Services**")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="contract")
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet")
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
            key="payment",
        )

    with col3:
        st.markdown("**Financials & Support**")
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=5.0, key="monthly")
        total_charges = st.number_input("Total Charges ($)", 0.0, 20000.0, 2500.0, step=100.0, key="total")
        num_support_tickets = st.slider("Support Tickets", 0, 10, 1, key="tickets")
        num_referrals = st.slider("Referrals", 0, 12, 0, key="referrals")

    st.markdown("---")

    predict_btn = st.button("🚀 Predict Churn", type="primary", use_container_width=True)

    if predict_btn:
        payload = {
            "gender": gender,
            "senior_citizen": senior_citizen,
            "tenure": tenure,
            "contract": contract,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "num_support_tickets": num_support_tickets,
            "num_referrals": num_referrals,
        }

        with st.spinner("Analyzing customer profile..."):
            time.sleep(0.3)  # tiny delay for visual feedback
            result = make_prediction(payload)

        if result:
            is_churn = result["prediction"] == "Churn"
            card_class = "result-churn" if is_churn else "result-no-churn"
            emoji = "⚠️" if is_churn else "✅"
            prob = result["churn_probability"]
            bar_color = (
                f"hsl({int((1 - prob) * 120)}, 80%, 50%)"  # red→green
            )

            # Result card
            st.markdown(f"""
            <div class="result-card {card_class}">
                <div class="result-title">{emoji} Prediction: {result['prediction']}</div>
                <p style="color:#a0aec0; margin:0;">
                    Risk Level: <strong>{result['risk_level']}</strong> &nbsp;|&nbsp;
                    Confidence: <strong>{result['confidence']:.1%}</strong>
                </p>
                <div style="margin-top:1rem;">
                    <span style="color:#e2e8f0; font-size:0.9rem;">
                        Churn Probability: <strong>{prob:.1%}</strong>
                    </span>
                    <div class="probability-bar">
                        <div class="probability-fill"
                             style="width:{prob*100}%; background:{bar_color};"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Risk factors
            if result.get("risk_factors"):
                st.markdown("#### 🔍 Risk Factors")
                for factor in result["risk_factors"]:
                    st.markdown(f'<div class="risk-factor">⚡ {factor}</div>', unsafe_allow_html=True)
            else:
                st.success("No significant risk factors identified — customer appears stable.")

            # Raw JSON expander
            with st.expander("📋 Full API Response (JSON)"):
                st.json(result)


# ---------------------------------------------------------------------------
# TAB 2 — Batch Upload
# ---------------------------------------------------------------------------
with tab2:
    st.markdown("### 📤 Upload a CSV for Batch Predictions")
    st.markdown(
        "Upload a CSV with these columns: `gender`, `senior_citizen`, `tenure`, "
        "`contract`, `internet_service`, `payment_method`, `monthly_charges`, "
        "`total_charges`, `num_support_tickets`, `num_referrals`"
    )

    uploaded = st.file_uploader("Choose CSV file", type=["csv"], key="batch_csv")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(), use_container_width=True)

        if st.button("🚀 Run Batch Prediction", type="primary"):
            records = df.to_dict(orient="records")
            try:
                r = requests.post(
                    f"{API_URL}/predict/batch",
                    json={"customers": records},
                    timeout=60,
                )
                if r.status_code == 200:
                    batch_result = r.json()
                    results_df = pd.DataFrame([
                        {
                            "Prediction": p["prediction"],
                            "Churn Prob": f"{p['churn_probability']:.1%}",
                            "Risk Level": p["risk_level"],
                            "Confidence": f"{p['confidence']:.1%}",
                        }
                        for p in batch_result["predictions"]
                    ])
                    combined = pd.concat([df.reset_index(drop=True), results_df], axis=1)
                    st.dataframe(combined, use_container_width=True)

                    # Summary metrics
                    churn_count = sum(1 for p in batch_result["predictions"] if p["prediction"] == "Churn")
                    total = batch_result["count"]

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Customers", total)
                    c2.metric("Predicted Churn", churn_count)
                    c3.metric("Churn Rate", f"{churn_count/total:.1%}")

                    # Download
                    csv_out = combined.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results CSV",
                        csv_out,
                        "churn_predictions.csv",
                        "text/csv",
                    )
                else:
                    st.error(f"API Error: {r.text}")
            except requests.ConnectionError:
                st.error("⚠️ Cannot reach the API. Is the backend running?")


# ---------------------------------------------------------------------------
# TAB 3 — API Documentation
# ---------------------------------------------------------------------------
with tab3:
    st.markdown("### 📚 API Reference")
    st.markdown(
        "The FastAPI backend auto-generates interactive docs. "
        "Visit the links below while the server is running:"
    )
    st.markdown(f"- **Swagger UI**: [{API_URL}/docs]({API_URL}/docs)")
    st.markdown(f"- **ReDoc**: [{API_URL}/redoc]({API_URL}/redoc)")

    st.markdown("---")
    st.markdown("### 🔗 Endpoints")

    with st.expander("GET /health — Health Check"):
        st.code(
            'curl -X GET "http://localhost:8000/health"',
            language="bash",
        )
        st.json({
            "status": "ok",
            "model_loaded": True,
            "version": "1.0.0",
            "timestamp": "2026-01-13T10:00:00Z",
        })

    with st.expander("POST /predict — Single Prediction"):
        st.code(
            """curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "gender": "Male",
    "senior_citizen": 0,
    "tenure": 5,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check",
    "monthly_charges": 95.50,
    "total_charges": 480.00,
    "num_support_tickets": 6,
    "num_referrals": 0
  }'""",
            language="bash",
        )

    with st.expander("POST /predict/batch — Batch Prediction"):
        st.code(
            """curl -X POST "http://localhost:8000/predict/batch" \\
  -H "Content-Type: application/json" \\
  -d '{"customers": [{ ... }, { ... }]}'""",
            language="bash",
        )

    with st.expander("GET /model/info — Model Metadata"):
        st.code(
            'curl -X GET "http://localhost:8000/model/info"',
            language="bash",
        )
