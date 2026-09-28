"""
Tests for the FastAPI prediction endpoints.
Uses httpx + pytest with the FastAPI TestClient.
"""

import sys
import pathlib

# Ensure the project root is on the path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    """Use TestClient as context manager so lifespan events fire (model loads)."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Sample payloads
# ---------------------------------------------------------------------------

VALID_CUSTOMER = {
    "gender": "Male",
    "senior_citizen": 0,
    "tenure": 5,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check",
    "monthly_charges": 95.50,
    "total_charges": 480.00,
    "num_support_tickets": 6,
    "num_referrals": 0,
}

STABLE_CUSTOMER = {
    "gender": "Female",
    "senior_citizen": 0,
    "tenure": 60,
    "contract": "Two year",
    "internet_service": "DSL",
    "payment_method": "Credit card",
    "monthly_charges": 45.00,
    "total_charges": 2700.00,
    "num_support_tickets": 0,
    "num_referrals": 5,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """GET /health"""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_model_loaded(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    def test_health_has_metrics(self, client):
        response = client.get("/health")
        data = response.json()
        assert "training_metrics" in data


class TestPredictEndpoint:
    """POST /predict"""

    def test_predict_returns_200(self, client):
        response = client.post("/predict", json=VALID_CUSTOMER)
        assert response.status_code == 200

    def test_predict_has_required_fields(self, client):
        response = client.post("/predict", json=VALID_CUSTOMER)
        data = response.json()
        assert "prediction" in data
        assert "churn_probability" in data
        assert "confidence" in data
        assert "risk_level" in data
        assert "risk_factors" in data
        assert "timestamp" in data

    def test_predict_probability_range(self, client):
        response = client.post("/predict", json=VALID_CUSTOMER)
        data = response.json()
        assert 0.0 <= data["churn_probability"] <= 1.0
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_valid_prediction_label(self, client):
        response = client.post("/predict", json=VALID_CUSTOMER)
        data = response.json()
        assert data["prediction"] in ("Churn", "No Churn")

    def test_high_risk_customer(self, client):
        """Short-tenure, month-to-month, fiber optic, e-check -> likely high risk."""
        response = client.post("/predict", json=VALID_CUSTOMER)
        data = response.json()
        # Should have risk factors
        assert len(data["risk_factors"]) > 0

    def test_stable_customer(self, client):
        """Long-tenure, two-year contract, low charges -> likely low risk."""
        response = client.post("/predict", json=STABLE_CUSTOMER)
        data = response.json()
        assert data["churn_probability"] < 0.5

    def test_invalid_gender_returns_422(self, client):
        bad = VALID_CUSTOMER.copy()
        bad["gender"] = "Other"
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_invalid_contract_returns_422(self, client):
        bad = VALID_CUSTOMER.copy()
        bad["contract"] = "Weekly"
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_negative_tenure_returns_422(self, client):
        bad = VALID_CUSTOMER.copy()
        bad["tenure"] = -5
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_missing_field_returns_422(self, client):
        incomplete = {"gender": "Male", "tenure": 10}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422


class TestBatchEndpoint:
    """POST /predict/batch"""

    def test_batch_prediction(self, client):
        payload = {"customers": [VALID_CUSTOMER, STABLE_CUSTOMER]}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_too_large_returns_400(self, client):
        payload = {"customers": [VALID_CUSTOMER] * 101}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 400


class TestModelInfoEndpoint:
    """GET /model/info"""

    def test_model_info(self, client):
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "RandomForestClassifier"
        assert data["n_estimators"] == 200
        assert "features" in data


class TestRootEndpoint:
    """GET /"""

    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
