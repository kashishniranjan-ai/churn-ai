"""
API tests - Customer Repeat-Purchase service.  Run: python -m pytest tests/ -v

Tests read real customers from data/processed/retail.db when available, so the
API is exercised with genuine feature values. Validation tests need no database.
"""

import sqlite3
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import STATE, app  # noqa: E402
from app.schemas import FEATURE_ORDER  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "model" / "customer_prediction_pipeline.joblib"
DB_PATH = PROJECT_ROOT / "data" / "processed" / "retail.db"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Model artifact missing - run `python src/train_model.py` first.",
)


@pytest.fixture(scope="module")
def client():
    """TestClient as a context manager so the lifespan startup runs."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def real_customers():
    """Build features for 5 real customers straight from the database."""
    if not DB_PATH.exists():
        pytest.skip("retail.db missing - run the data pipeline first")
    from src.features import build_modelling_table

    returns_csv = PROJECT_ROOT / "data" / "processed" / "returns.csv"
    conn = sqlite3.connect(DB_PATH)
    try:
        table, _, _ = build_modelling_table(conn, str(returns_csv))
    finally:
        conn.close()
    return table.head(5).to_dict("records")


def _payload(**overrides) -> dict:
    """A valid customer with optional per-field overrides."""
    base = {
        "tenure_days": 250,
        "recency_days": 40,
        "frequency": 6,
        "monetary": 1850.0,
        "avg_order_value": 308.33,
        "total_items": 720,
        "avg_items_per_order": 120.0,
        "distinct_products": 45,
        "avg_unit_price": 2.9,
        "months_active": 5,
        "max_gap_days": 62,
        "returns_rate": 0.05,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------- startup ---
def test_startup_loads_pipeline(client):
    assert client.get("/health").status_code == 200
    assert STATE["pipeline"] is not None
    assert STATE["feature_order"] == list(FEATURE_ORDER)
    assert 0.0 < STATE["threshold"] < 1.0


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["feature_count"] == len(FEATURE_ORDER)


def test_metrics_reports_baselines(client):
    body = client.get("/metrics").json()
    assert body["model_type"]
    assert body["test_metrics"]["roc_auc"] > 0.5
    # The report must keep baselines visible, not only flattering metrics.
    assert "baselines" in body
    assert "campaign_metrics" in body


# -------------------------------------------------------------- prediction ---
def test_predict_single(client):
    response = client.post("/predict", json=_payload())
    assert response.status_code == 200
    body = response.json()
    for key in (
        "prediction",
        "will_repeat",
        "repeat_probability",
        "confidence",
        "engagement_tier",
        "key_drivers",
        "threshold",
    ):
        assert key in body, f"missing '{key}'"
    assert 0.0 <= body["repeat_probability"] <= 1.0
    assert body["prediction"] in ("Will Repeat", "Will Not Repeat")


def test_prediction_matches_threshold(client):
    body = client.post("/predict", json=_payload()).json()
    expected = (
        "Will Repeat" if body["repeat_probability"] >= body["threshold"] else "Will Not Repeat"
    )
    assert body["prediction"] == expected
    assert body["will_repeat"] is (body["repeat_probability"] >= body["threshold"])


def test_prediction_is_deterministic(client):
    first = client.post("/predict", json=_payload()).json()["repeat_probability"]
    second = client.post("/predict", json=_payload()).json()["repeat_probability"]
    assert first == pytest.approx(second)


def test_key_drivers_reference_real_feature_names(client):
    body = client.post("/predict", json=_payload()).json()
    assert len(body["key_drivers"]) == 3
    labels = " ".join(body["key_drivers"]).lower()
    # Drivers must be grounded in actual inputs, not generic churn filler.
    assert any(
        token in labels
        for token in ("tenure", "purchase", "order", "spend", "product", "return", "gap", "item")
    )


def test_predict_on_real_customers(client, real_customers):
    for record in real_customers:
        payload = {
            f: (float(record[f]) if f == "returns_rate" else int(record[f]))
            for f in FEATURE_ORDER
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200, response.text
        assert 0.0 <= response.json()["repeat_probability"] <= 1.0


def test_batch_predict(client):
    response = client.post(
        "/predict/batch",
        json={"customers": [_payload(), _payload(frequency=1, months_active=1, monetary=10.0)]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_batch_ranking_is_sensible(client):
    """Engaged multi-order customer must outrank a tiny one-off buyer."""
    response = client.post(
        "/predict/batch",
        json={
            "customers": [
                _payload(),
                _payload(
                    tenure_days=5, recency_days=200, frequency=1, monetary=12.0,
                    avg_order_value=12.0, total_items=3, avg_items_per_order=3,
                    distinct_products=1, avg_unit_price=4.0, months_active=1,
                    max_gap_days=0, returns_rate=0.0,
                ),
            ]
        },
    )
    scores = [p["repeat_probability"] for p in response.json()["predictions"]]
    assert scores[0] > scores[1]


# ------------------------------------------------------------- validation ---
@pytest.mark.parametrize("field", list(FEATURE_ORDER))
def test_missing_field_rejected(client, field):
    payload = _payload()
    del payload[field]
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize(
    "field,value",
    [
        ("monetary", -1.0),
        ("frequency", 0),
        ("tenure_days", -5),
        ("returns_rate", 1.5),
        ("returns_rate", -0.1),
        ("total_items", -3),
        ("months_active", 9999),
    ],
)
def test_out_of_range_rejected(client, field, value):
    assert client.post("/predict", json=_payload(**{field: value})).status_code == 422


def test_empty_batch_rejected(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


def test_unknown_field_is_not_fatal(client):
    """Extra keys must not break inference (forward-compatible clients)."""
    assert client.post("/predict", json=_payload(customer_id="C123")).status_code == 200

