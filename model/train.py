"""
Customer Churn Predictor — Model Training & Serialization Script
=================================================================
Generates a synthetic Telco Churn dataset, trains a Random Forest
classifier, and serializes both the model and its preprocessing
pipeline so that inference never depends on the training environment.

Artifacts produced:
    model/churn_model.pkl      — fitted RandomForestClassifier
    model/preprocessor.pkl     — fitted ColumnTransformer (scaler + encoder)
    model/feature_names.pkl    — ordered list of raw input feature names
    model/training_metrics.json — accuracy, precision, recall, F1, ROC-AUC
"""

import json
import pathlib
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Synthetic dataset generation
# ---------------------------------------------------------------------------

def generate_churn_dataset(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Create a realistic, reproducible Telco-style churn dataset."""
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 73, size=n_samples)
    monthly_charges = rng.uniform(18.0, 120.0, size=n_samples).round(2)
    total_charges = (tenure * monthly_charges * rng.uniform(0.85, 1.15, size=n_samples)).round(2)
    num_support_tickets = rng.integers(0, 10, size=n_samples)
    num_referrals = rng.integers(0, 12, size=n_samples)

    contract_choices = ["Month-to-month", "One year", "Two year"]
    contract = rng.choice(contract_choices, size=n_samples, p=[0.50, 0.30, 0.20])

    internet_choices = ["DSL", "Fiber optic", "No"]
    internet_service = rng.choice(internet_choices, size=n_samples, p=[0.35, 0.45, 0.20])

    payment_choices = [
        "Electronic check", "Mailed check", "Bank transfer", "Credit card"
    ]
    payment_method = rng.choice(payment_choices, size=n_samples)

    gender = rng.choice(["Male", "Female"], size=n_samples)
    senior_citizen = rng.choice([0, 1], size=n_samples, p=[0.84, 0.16])

    # --- Churn label (realistic probabilistic rules) ---
    churn_prob = np.full(n_samples, 0.15)
    churn_prob[contract == "Month-to-month"] += 0.20
    churn_prob[internet_service == "Fiber optic"] += 0.10
    churn_prob[payment_method == "Electronic check"] += 0.08
    churn_prob[tenure < 12] += 0.12
    churn_prob[monthly_charges > 80] += 0.08
    churn_prob[num_support_tickets > 5] += 0.10
    churn_prob[senior_citizen == 1] += 0.05
    churn_prob[tenure > 48] -= 0.15
    churn_prob[contract == "Two year"] -= 0.15
    churn_prob[num_referrals > 3] -= 0.10
    churn_prob = np.clip(churn_prob, 0.02, 0.95)

    churn = rng.binomial(1, churn_prob)

    return pd.DataFrame({
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
        "churn": churn,
    })


# ---------------------------------------------------------------------------
# 2. Training pipeline
# ---------------------------------------------------------------------------

def train_model():
    MODEL_DIR = pathlib.Path(__file__).resolve().parent
    print("=" * 60)
    print("  Customer Churn Predictor — Training Pipeline")
    print("=" * 60)

    # ---- Data ----
    df = generate_churn_dataset()
    print(f"\n✅ Dataset generated: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"   Churn rate: {df['churn'].mean():.1%}")

    X = df.drop("churn", axis=1)
    y = df["churn"]

    feature_names = list(X.columns)

    numeric_features = [
        "tenure", "monthly_charges", "total_charges",
        "num_support_tickets", "num_referrals", "senior_citizen",
    ]
    categorical_features = [
        "gender", "contract", "internet_service", "payment_method",
    ]

    # ---- Preprocessing ----
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
    )

    # ---- Train / Test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"   Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # ---- Fit preprocessor + model ----
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_transformed, y_train)
    print("\n✅ RandomForestClassifier trained (200 trees, depth=12)")

    # ---- Evaluate ----
    y_pred = model.predict(X_test_transformed)
    y_proba = model.predict_proba(X_test_transformed)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    print("\n📊 Evaluation Metrics (Test Set)")
    print("-" * 40)
    for k, v in metrics.items():
        print(f"   {k:<12}: {v}")
    print("-" * 40)
    print("\n" + classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # ---- Serialize everything ----
    joblib.dump(model, MODEL_DIR / "churn_model.pkl")
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.pkl")
    joblib.dump(feature_names, MODEL_DIR / "feature_names.pkl")

    with open(MODEL_DIR / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Artifacts saved:")
    for fname in ["churn_model.pkl", "preprocessor.pkl",
                   "feature_names.pkl", "training_metrics.json"]:
        fpath = MODEL_DIR / fname
        size_kb = fpath.stat().st_size / 1024
        print(f"   → {fname}  ({size_kb:.1f} KB)")

    print("\n🎉 Training complete. Ready for deployment.\n")
    return model, preprocessor, feature_names, metrics


if __name__ == "__main__":
    train_model()
