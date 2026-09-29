"""
Customer Repeat-Purchase Prediction API
=======================================
FastAPI service serving the model trained by ``src/train_model.py`` on the real
UCI Online Retail data.

Design notes
------------
* One artifact, one truth: a single scikit-learn ``Pipeline`` (scaler +
  classifier) is loaded, so preprocessing can never drift from training.
* Feature order is validated at startup against ``feature_metadata.json``.
  A schema/model mismatch fails loudly at boot instead of silently
  mis-predicting (this was a real bug in the previous version of this file).
* Risk-factor explanations come from the model's own stored importances and
  signed effects - no hard-coded feature lists.
* Batch inference is vectorised (one ``predict_proba`` call for the batch).
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    FEATURE_ORDER,
    BatchInput,
    BatchResponse,
    CustomerInput,
    HealthResponse,
    PredictionResponse,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Overridable so containers / managed platforms can mount artifacts elsewhere.
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "model"))
PIPELINE_PATH = MODEL_DIR / "customer_prediction_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "feature_metadata.json"
METRICS_PATH = MODEL_DIR / "training_metrics.json"

VERSION = "2.0.0"

# Shared service state, populated by the lifespan handler at startup.
STATE: dict = {
    "pipeline": None,
    "metadata": {},
    "metrics": {},
    "feature_order": list(FEATURE_ORDER),
    "threshold": 0.5,
}


def _load_artifacts() -> None:
    """Load pipeline + metadata and fail loudly if anything is inconsistent."""
    if not PIPELINE_PATH.exists():
        raise RuntimeError(
            f"Missing model artifact: {PIPELINE_PATH}\n"
            "Run `python src/train_model.py` before starting the API."
        )

    STATE["pipeline"] = joblib.load(PIPELINE_PATH)
    STATE["metadata"] = (
        json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
    )
    STATE["metrics"] = (
        json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    )

    stored_order = STATE["metadata"].get("feature_order")
    if stored_order and tuple(stored_order) != tuple(FEATURE_ORDER):
        raise RuntimeError(
            "Feature order mismatch between app/schemas.py and the trained "
            "model - the API would feed features in the wrong order.\n"
            f"  schema: {tuple(FEATURE_ORDER)}\n  model : {tuple(stored_order)}\n"
            "Fix app/schemas.py::FEATURE_ORDER or retrain."
        )
    STATE["feature_order"] = list(stored_order or FEATURE_ORDER)
    STATE["threshold"] = float(STATE["metadata"].get("threshold", 0.5))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    print(
        f"[startup] loaded {STATE['metrics'].get('model_type', 'unknown')} "
        f"threshold={STATE['threshold']}"
    )
    yield


app = FastAPI(
    title="Customer Repeat-Purchase Prediction API",
    version=VERSION,
    description=(
        "Predicts whether an e-commerce customer will place another order in "
        "the next quarter, from their behavioural history. Trained on the real "
        "UCI Online Retail dataset."
    ),
    lifespan=lifespan,
)

# CORS: '*' cannot legally be combined with credentials, so origins are explicit.
_raw_origins = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
)
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]
ALLOW_CREDENTIALS = "*" not in ALLOWED_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

LABELS = {
    "tenure_days": "Customer tenure",
    "recency_days": "Days since last purchase",
    "frequency": "Number of orders",
    "monetary": "Total spend",
    "avg_order_value": "Average order value",
    "total_items": "Total items bought",
    "avg_items_per_order": "Items per order",
    "distinct_products": "Distinct products bought",
    "avg_unit_price": "Average unit price paid",
    "months_active": "Months with purchase activity",
    "max_gap_days": "Longest purchase gap",
    "returns_rate": "Return / cancellation rate",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tier(p: float) -> str:
    if p >= 0.80:
        return "Highly Likely"
    if p >= 0.60:
        return "Likely"
    if p >= 0.40:
        return "Uncertain"
    if p >= 0.20:
        return "Unlikely"
    return "Very Unlikely"


def _drivers(record: dict, top_n: int = 3) -> list[str]:
    """
    Explain one prediction using ONLY artifacts the model produced:
    stored importances (magnitude), stored signed effects (direction) and the
    training distribution (to say what 'typical' means).
    """
    meta = STATE["metadata"]
    stats = meta.get("feature_stats", {})
    imps = meta.get("feature_importances", {})
    effects = meta.get("feature_effects", {})
    if not stats or not imps:
        return ["Explanations unavailable - feature_metadata.json is missing."]

    ranked = []
    for feat in STATE["feature_order"]:
        s = stats.get(feat)
        if not s:
            continue
        std = s.get("std") or 1.0
        z = (float(record[feat]) - s.get("mean", 0.0)) / std
        weight = abs(imps.get(feat, 0.0) * z)
        # effect * z  > 0  => this customer's value pushes toward 'will repeat'
        push = effects.get(feat, 0.0) * z
        ranked.append((weight, feat, push, z, s))

    ranked.sort(key=lambda t: t[0], reverse=True)
    lines = []
    for _, feat, push, z, s in ranked[:top_n]:
        # Compare against the SAME statistic the ranking uses (the training
        # mean), otherwise the sentence can contradict its own numbers.
        value, typical = float(record[feat]), s.get("mean", 0.0)
        spread = "higher than" if z > 0 else "lower than" if z < 0 else "in line with"
        effect = "raises" if push > 0 else "lowers" if push < 0 else "does not shift"
        lines.append(
            f"{LABELS.get(feat, feat)} is {spread} the training average "
            f"({value:,.2f} vs {typical:,.2f}), which {effect} the "
            f"repeat-purchase estimate."
        )
    return lines


def _predict_records(records: list[dict]) -> list[dict]:
    """Vectorised inference for 1..N customer dicts -> response payloads."""
    pipeline = STATE["pipeline"]
    threshold = STATE["threshold"]
    if pipeline is None:  # pragma: no cover - lifespan loads before requests
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    frame = pd.DataFrame(records)[STATE["feature_order"]]
    try:
        proba = np.asarray(pipeline.predict_proba(frame)[:, 1], dtype=float)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    out = []
    for record, p in zip(records, proba):
        p = float(np.clip(p, 0.0, 1.0))
        will_repeat = bool(p >= threshold)
        out.append(
            PredictionResponse(
                customer_features=record,
                prediction="Will Repeat" if will_repeat else "Will Not Repeat",
                will_repeat=will_repeat,
                repeat_probability=round(p, 4),
                confidence=round(p if will_repeat else 1.0 - p, 4),
                engagement_tier=_tier(p),
                key_drivers=_drivers(record),
                threshold=threshold,
                timestamp=_now_iso(),
            )
        )
    return out


@app.get("/", include_in_schema=False)
def root():
    return {"service": "Customer Repeat-Purchase Prediction API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness + what is actually loaded (useful right after a deploy)."""
    metrics = STATE["metrics"]
    return HealthResponse(
        status="healthy" if STATE["pipeline"] is not None else "degraded",
        model_loaded=STATE["pipeline"] is not None,
        version=VERSION,
        model_type=metrics.get("model_type"),
        feature_count=len(STATE["feature_order"]),
        training_metrics={
            k: metrics.get(k)
            for k in ("roc_auc_mean", "test_metrics", "campaign_metrics", "threshold")
        }
        if metrics
        else None,
        timestamp=_now_iso(),
    )


@app.get("/metrics")
def metrics():
    """Full training report: every candidate's CV score, test metrics, baselines."""
    if not STATE["metrics"]:
        raise HTTPException(status_code=404, detail="training_metrics.json not found")
    return STATE["metrics"]


@app.get("/features")
def features():
    """Stored feature importances, signed effects and training distributions.

    Exposed so the dashboard explains predictions from model artifacts rather
    than re-deriving (and possibly contradicting) them client-side.
    """
    if not STATE["metadata"]:
        raise HTTPException(status_code=404, detail="feature_metadata.json not found")
    return STATE["metadata"]


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    """Predict whether ONE customer places another order in the next quarter."""
    try:
        return _predict_records([customer.model_dump()])[0]
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(payload: BatchInput):
    """Predict for up to 500 customers in a single vectorised pass."""
    preds = _predict_records([c.model_dump() for c in payload.customers])
    return BatchResponse(count=len(preds), predictions=preds)

