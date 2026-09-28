"""
Customer Churn Predictor — FastAPI Backend
==========================================
Production-grade REST API serving churn predictions.
Loads serialized model & preprocessor at startup,
validates inputs via Pydantic, returns structured JSON.
"""

import json
import pathlib
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Paths — resolve model directory robustly for both direct run and test import
# ---------------------------------------------------------------------------
import os

_app_file_dir = pathlib.Path(__file__).resolve().parent          # …/app/
_project_root_from_file = _app_file_dir.parent                   # …/churn-ai-app/
_project_root_from_cwd = pathlib.Path(os.getcwd()).resolve()     # cwd

# Prefer the file-based root, but fall back to cwd if model/ doesn't exist there
if (_project_root_from_file / "model" / "churn_model.pkl").exists():
    BASE_DIR = _project_root_from_file
else:
    BASE_DIR = _project_root_from_cwd

MODEL_DIR = BASE_DIR / "model"

# ---------------------------------------------------------------------------
# Global model state (loaded once at startup)
# ---------------------------------------------------------------------------
_model = None
_preprocessor = None
_feature_names = None
_metrics = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once when the server starts."""
    global _model, _preprocessor, _feature_names, _metrics

    model_path = MODEL_DIR / "churn_model.pkl"
    preprocessor_path = MODEL_DIR / "preprocessor.pkl"
    features_path = MODEL_DIR / "feature_names.pkl"
    metrics_path = MODEL_DIR / "training_metrics.json"

    for p in [model_path, preprocessor_path, features_path]:
        if not p.exists():
            raise RuntimeError(
                f"Missing artifact: {p}. Run `python model/train.py` first."
            )

    _model = joblib.load(model_path)
    _preprocessor = joblib.load(preprocessor_path)
    _feature_names = joblib.load(features_path)

    if metrics_path.exists():
        with open(metrics_path) as f:
            _metrics = json.load(f)

    print(f"Model loaded  |  Features: {len(_feature_names)}  |  Ready.")
    yield
    print("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Customer Churn Predictor API",
    description=(
        "Production-grade REST API for predicting customer churn. "
        "Send customer attributes via POST /predict and receive an "
        "instant churn probability with risk classification."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class CustomerInput(BaseModel):
    """Schema for a single customer prediction request."""

    gender: str = Field(
        ..., description="Customer gender", examples=["Male", "Female"]
    )
    senior_citizen: int = Field(
        ..., ge=0, le=1, description="1 if senior citizen, else 0", examples=[0]
    )
    tenure: int = Field(
        ..., ge=0, le=100, description="Months with the company", examples=[24]
    )
    contract: str = Field(
        ...,
        description="Contract type",
        examples=["Month-to-month", "One year", "Two year"],
    )
    internet_service: str = Field(
        ...,
        description="Internet service type",
        examples=["DSL", "Fiber optic", "No"],
    )
    payment_method: str = Field(
        ...,
        description="Payment method",
        examples=["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    )
    monthly_charges: float = Field(
        ..., ge=0, description="Monthly charges in USD", examples=[79.85]
    )
    total_charges: float = Field(
        ..., ge=0, description="Total charges to date in USD", examples=[3320.75]
    )
    num_support_tickets: int = Field(
        ..., ge=0, description="Number of support tickets filed", examples=[2]
    )
    num_referrals: int = Field(
        ..., ge=0, description="Number of referrals made", examples=[1]
    )

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        allowed = {"Male", "Female"}
        if v not in allowed:
            raise ValueError(f"gender must be one of {allowed}")
        return v

    @field_validator("contract")
    @classmethod
    def validate_contract(cls, v: str) -> str:
        allowed = {"Month-to-month", "One year", "Two year"}
        if v not in allowed:
            raise ValueError(f"contract must be one of {allowed}")
        return v

    @field_validator("internet_service")
    @classmethod
    def validate_internet(cls, v: str) -> str:
        allowed = {"DSL", "Fiber optic", "No"}
        if v not in allowed:
            raise ValueError(f"internet_service must be one of {allowed}")
        return v

    @field_validator("payment_method")
    @classmethod
    def validate_payment(cls, v: str) -> str:
        allowed = {"Electronic check", "Mailed check", "Bank transfer", "Credit card"}
        if v not in allowed:
            raise ValueError(f"payment_method must be one of {allowed}")
        return v


class PredictionResponse(BaseModel):
    """Prediction response schema."""

    customer_data: dict
    prediction: str
    churn_probability: float
    confidence: float
    risk_level: str
    risk_factors: list[str]
    timestamp: str


class HealthResponse(BaseModel):
    """Health-check response schema."""

    status: str
    model_loaded: bool
    version: str
    timestamp: str
    training_metrics: Optional[dict] = None


class BatchInput(BaseModel):
    """Accept a list of customers for batch prediction."""

    customers: list[CustomerInput]


class BatchResponse(BaseModel):
    """Batch prediction response."""

    count: int
    predictions: list[PredictionResponse]


# ---------------------------------------------------------------------------
# Helper: risk factor analysis
# ---------------------------------------------------------------------------

def _analyze_risk_factors(data: CustomerInput) -> list[str]:
    """Return human-readable risk factors for interpretability."""
    factors = []
    if data.contract == "Month-to-month":
        factors.append("Month-to-month contract (high flexibility = high churn risk)")
    if data.tenure < 12:
        factors.append(f"Short tenure ({data.tenure} months — customers churn early)")
    if data.monthly_charges > 80:
        factors.append(f"High monthly charges (${data.monthly_charges:.2f})")
    if data.internet_service == "Fiber optic":
        factors.append("Fiber optic service (correlated with higher churn)")
    if data.payment_method == "Electronic check":
        factors.append("Electronic check payment (less sticky payment method)")
    if data.num_support_tickets > 4:
        factors.append(f"High support tickets ({data.num_support_tickets} — signals dissatisfaction)")
    if data.senior_citizen == 1:
        factors.append("Senior citizen (slightly elevated churn demographic)")
    if data.num_referrals == 0 and data.tenure > 6:
        factors.append("Zero referrals despite tenure (low engagement signal)")
    return factors


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Root"])
async def root():
    """API root — welcome message."""
    return {
        "message": "🚀 Customer Churn Predictor API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint — used by cloud services to verify liveness."""
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        training_metrics=_metrics,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(customer: CustomerInput):
    """
    Predict customer churn.

    Accepts customer attributes, preprocesses them with the saved
    ColumnTransformer, runs inference, and returns a structured result
    with probability, risk level, and interpretable risk factors.
    """
    if _model is None or _preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build DataFrame with correct column order
    input_df = pd.DataFrame([customer.model_dump()])[_feature_names]

    # Preprocess & predict
    X_transformed = _preprocessor.transform(input_df)
    proba = float(_model.predict_proba(X_transformed)[0][1])
    prediction = "Churn" if proba >= 0.5 else "No Churn"
    confidence = proba if prediction == "Churn" else 1 - proba

    # Risk classification
    if proba >= 0.75:
        risk_level = "🔴 Critical"
    elif proba >= 0.50:
        risk_level = "🟠 High"
    elif proba >= 0.30:
        risk_level = "🟡 Medium"
    else:
        risk_level = "🟢 Low"

    risk_factors = _analyze_risk_factors(customer)

    return PredictionResponse(
        customer_data=customer.model_dump(),
        prediction=prediction,
        churn_probability=round(proba, 4),
        confidence=round(confidence, 4),
        risk_level=risk_level,
        risk_factors=risk_factors,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
async def predict_batch(batch: BatchInput):
    """Batch prediction endpoint — accepts up to 100 customers at once."""
    if len(batch.customers) > 100:
        raise HTTPException(status_code=400, detail="Maximum batch size is 100")

    results = []
    for customer in batch.customers:
        result = await predict(customer)
        results.append(result)

    return BatchResponse(count=len(results), predictions=results)


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Return model metadata and training metrics."""
    return {
        "model_type": "RandomForestClassifier",
        "n_estimators": 200,
        "max_depth": 12,
        "features": _feature_names,
        "feature_count": len(_feature_names) if _feature_names else 0,
        "training_metrics": _metrics,
    }
