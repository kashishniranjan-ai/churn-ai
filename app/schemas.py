"""
Pydantic schemas for the Customer Repeat-Purchase API
=====================================================
``CustomerInput`` mirrors ``src/features.py::FEATURE_COLUMNS`` exactly, in the
same order. ``app/main.py`` reindexes incoming payloads against the
``feature_order`` stored in ``model/feature_metadata.json`` and raises a
startup error if the two ever drift apart, so a schema/model mismatch can never
silently produce wrong predictions.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Same order as src/features.py::FEATURE_COLUMNS
FEATURE_ORDER: tuple[str, ...] = (
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
)


class CustomerInput(BaseModel):
    """
    Aggregate behaviour for ONE customer, measured over a fixed historical
    window (here: the customer's history before 2011-09-01).

    All values are derived from the customer's own past transactions only.
    Currency is GBP (the UCI Online Retail dataset is a UK retailer).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
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
            ]
        }
    )

    tenure_days: int = Field(
        ..., ge=0, le=36500, description="Days between first purchase and the scoring cutoff."
    )
    recency_days: int = Field(
        ..., ge=0, le=36500, description="Days between last purchase and the scoring cutoff."
    )
    frequency: int = Field(
        ..., ge=1, le=100000, description="Number of distinct orders placed."
    )
    monetary: float = Field(
        ..., ge=0, le=100_000_000, description="Total spend in GBP."
    )
    avg_order_value: float = Field(
        ..., ge=0, le=100_000_000, description="monetary / frequency (GBP)."
    )
    total_items: int = Field(
        ..., ge=0, le=10_000_000, description="Total units purchased."
    )
    avg_items_per_order: float = Field(
        ..., ge=0, le=1_000_000, description="total_items / frequency."
    )
    distinct_products: int = Field(
        ..., ge=0, le=1_000_000, description="Number of distinct StockCodes bought."
    )
    avg_unit_price: float = Field(
        ..., ge=0, le=100_000, description="Mean unit price paid (GBP)."
    )
    months_active: int = Field(
        ..., ge=0, le=1200, description="Distinct calendar months containing a purchase."
    )
    max_gap_days: int = Field(
        ..., ge=0, le=36500, description="Longest gap (days) between consecutive purchase days."
    )
    returns_rate: float = Field(
        ...,
        ge=0,
        le=1,
        description="Cancellations / (cancellations + orders), bounded on [0, 1].",
    )

    @field_validator("monetary", "avg_order_value", "avg_unit_price")
    @classmethod
    def _round_currency(cls, v: float) -> float:
        """Guard against float noise like 1e-17 sneaking into a spend field."""
        return round(float(v), 6)


class PredictionResponse(BaseModel):
    """A single repeat-purchase prediction."""

    customer_features: dict
    prediction: str = Field(..., description='"Will Repeat" or "Will Not Repeat".')
    will_repeat: bool
    repeat_probability: float
    confidence: float
    engagement_tier: str
    key_drivers: list[str]
    threshold: float
    timestamp: str


class BatchInput(BaseModel):
    """Up to 500 customers per batch request."""

    customers: list[CustomerInput] = Field(..., min_length=1, max_length=500)


class BatchResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    model_type: Optional[str] = None
    feature_count: Optional[int] = None
    training_metrics: Optional[dict] = None
    timestamp: str
