# 🔮 Customer Churn Predictor — End-to-End AI Deployment

> A production-grade ML system that predicts customer churn using a Random Forest
> model, served via a FastAPI REST API, with a Streamlit dashboard frontend,
> fully containerized with Docker and ready for cloud deployment.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📂 Project Structure

```
churn-ai-app/
├── model/
│   ├── train.py               # Training & serialization script
│   ├── churn_model.pkl         # Serialized Random Forest model
│   ├── preprocessor.pkl        # Serialized ColumnTransformer
│   ├── feature_names.pkl       # Ordered feature name list
│   └── training_metrics.json   # Accuracy, F1, AUC metrics
├── app/
│   └── main.py                 # FastAPI backend (REST API)
├── frontend/
│   └── dashboard.py            # Streamlit dashboard (UI)
├── tests/
│   └── test_api.py             # Pytest test suite
├── Dockerfile                  # Container build instructions
├── supervisord.conf            # Process manager config
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md                   # ← You are here
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/churn-ai-app.git
cd churn-ai-app

# Create virtual environment
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python model/train.py
```

This generates synthetic churn data, trains a Random Forest classifier, and
saves the model artifacts to the `model/` directory.

### 3. Start the FastAPI Backend

```bash
uvicorn app.main:app --reload --port 8000
```

API is now live at: **http://localhost:8000**
- Swagger Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 4. Start the Streamlit Frontend

```bash
streamlit run frontend/dashboard.py
```

Dashboard is now live at: **http://localhost:8501**

---

## 🔗 API Endpoints

| Method | Endpoint         | Description                       |
|--------|------------------|-----------------------------------|
| GET    | `/`              | Welcome message + navigation      |
| GET    | `/health`        | Health check (liveness probe)     |
| POST   | `/predict`       | Single customer churn prediction  |
| POST   | `/predict/batch` | Batch prediction (up to 100)      |
| GET    | `/model/info`    | Model metadata & training metrics |

### Example: cURL Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
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
  }'
```

### Example Response

```json
{
  "prediction": "Churn",
  "churn_probability": 0.8234,
  "confidence": 0.8234,
  "risk_level": "🔴 Critical",
  "risk_factors": [
    "Month-to-month contract (high flexibility = high churn risk)",
    "Short tenure (5 months — customers churn early)",
    "High monthly charges ($95.50)",
    "Fiber optic service (correlated with higher churn)",
    "Electronic check payment (less sticky payment method)",
    "High support tickets (6 — signals dissatisfaction)"
  ],
  "timestamp": "2026-01-13T10:00:00Z"
}
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Runs 15+ tests covering:
- Health endpoint validation
- Valid & invalid prediction inputs
- Pydantic schema enforcement (422 errors)
- Batch prediction limits
- Model metadata endpoint

---

## 🐳 Docker

### Build & Run

```bash
# Build the image
docker build -t churn-predictor .

# Run the container
docker run -p 8000:8000 -p 8501:8501 churn-predictor
```

- FastAPI: http://localhost:8000
- Streamlit: http://localhost:8501

### Docker Compose (optional)

```yaml
version: "3.9"
services:
  churn-app:
    build: .
    ports:
      - "8000:8000"
      - "8501:8501"
    restart: unless-stopped
```

---

## ☁️ Cloud Deployment

### Option A: Render

1. Push code to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Set **Build Command**: `pip install -r requirements.txt && python model/train.py`
5. Set **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
6. Deploy!

### Option B: Railway

1. Push to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from Repo
3. Railway auto-detects the Dockerfile
4. Set port variables if needed

### Option C: Google Cloud Run

```bash
# Build & push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/churn-predictor

# Deploy
gcloud run deploy churn-predictor \
  --image gcr.io/PROJECT_ID/churn-predictor \
  --platform managed \
  --port 8000 \
  --allow-unauthenticated
```

---

## 📊 Model Details

| Attribute         | Value                            |
|-------------------|----------------------------------|
| Algorithm         | Random Forest Classifier         |
| Trees             | 200                              |
| Max Depth         | 12                               |
| Class Weighting   | Balanced                         |
| Features          | 10 (6 numeric + 4 categorical)  |
| Preprocessing     | StandardScaler + OneHotEncoder   |
| Dataset           | 5,000 synthetic Telco customers  |
| Train/Test Split  | 80/20 (stratified)               |

---

## 🛠️ Tech Stack

- **ML**: scikit-learn, pandas, numpy, joblib
- **Backend**: FastAPI, Pydantic, Uvicorn
- **Frontend**: Streamlit
- **DevOps**: Docker, Supervisor
- **Testing**: pytest, httpx
- **Deployment**: Render / Railway / GCP Cloud Run

---

## 📋 Deliverables Checklist

- [x] Trained & serialized ML model (`model/churn_model.pkl`)
- [x] Saved preprocessor (`model/preprocessor.pkl`)
- [x] FastAPI REST API with `/health` and `/predict` endpoints
- [x] Pydantic input validation
- [x] Streamlit frontend dashboard
- [x] Comprehensive test suite
- [x] Dockerfile for containerization
- [x] README with full documentation
- [ ] Live public URL (deploy to Render/Railway/GCR)
- [ ] Demo video recording

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
