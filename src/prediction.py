"""
ChurnGuard AI — Prediction Engine

Loads the trained Random Forest model and preprocessor to generate
churn predictions with probability scores, risk categories, and
per-customer contributing factors.
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS_DIR: Path = Path(__file__).resolve().parent.parent / "models"

# Configurable risk thresholds
RISK_THRESHOLDS = {
    "high": 0.70,
    "medium": 0.40,
}

# Feature groups (must match preprocessing)
NUMERICAL_FEATURES: List[str] = [
    "Age", "Tenure", "MonthlyCharges", "TotalCharges",
    "SupportTickets", "SatisfactionScore", "UsageFrequency", "LastLoginDays",
]
CATEGORICAL_FEATURES: List[str] = [
    "Gender", "ContractType", "PaymentMethod",
    "InternetService", "TechSupport", "OnlineSecurity",
    "DeviceProtection", "StreamingServices",
]
ALL_FEATURES: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_and_preprocessor() -> Tuple[RandomForestClassifier, ColumnTransformer]:
    """Load the trained model and fitted preprocessor from disk.

    Returns:
        Tuple of (model, preprocessor).

    Raises:
        FileNotFoundError: If model or preprocessor files are missing.
    """
    model_path = MODELS_DIR / "random_forest_model.pkl"
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_model.py first.")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}.")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor


# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------
def classify_risk(
    probability: float,
    thresholds: Optional[Dict[str, float]] = None,
) -> str:
    """Classify churn risk based on predicted probability.

    Args:
        probability: Predicted churn probability (0–1).
        thresholds: Optional custom thresholds dict with 'high' and 'medium' keys.

    Returns:
        Risk level string: "🔴 HIGH RISK", "🟠 MEDIUM RISK", or "🟢 LOW RISK".
    """
    t = thresholds or RISK_THRESHOLDS
    if probability >= t["high"]:
        return "🔴 HIGH RISK"
    elif probability >= t["medium"]:
        return "🟠 MEDIUM RISK"
    else:
        return "🟢 LOW RISK"


def get_risk_label(probability: float, thresholds: Optional[Dict[str, float]] = None) -> str:
    """Get plain risk label without emoji."""
    t = thresholds or RISK_THRESHOLDS
    if probability >= t["high"]:
        return "High"
    elif probability >= t["medium"]:
        return "Medium"
    else:
        return "Low"


# ---------------------------------------------------------------------------
# Feature contribution analysis
# ---------------------------------------------------------------------------
def get_top_risk_factors(
    model: RandomForestClassifier,
    feature_names: List[str],
    customer_features: np.ndarray,
    top_n: int = 5,
) -> List[Dict[str, float]]:
    """Identify the top contributing features for a single prediction.

    Uses feature importance weighted by the customer's feature deviation
    from the training mean to approximate per-customer attribution.

    Args:
        model: Fitted RandomForestClassifier.
        feature_names: Names of preprocessed features.
        customer_features: Single customer's preprocessed feature vector.
        top_n: Number of top factors to return.

    Returns:
        List of dicts with 'feature' and 'importance' keys, sorted descending.
    """
    importances = model.feature_importances_
    # Weight by absolute feature value (as a proxy for deviation)
    abs_values = np.abs(customer_features.flatten())
    weighted = importances * abs_values
    weighted = weighted / (weighted.sum() + 1e-10)

    indices = np.argsort(weighted)[::-1][:top_n]
    factors = []
    for idx in indices:
        name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
        factors.append({"feature": name, "importance": float(weighted[idx])})

    return factors


# ---------------------------------------------------------------------------
# Single prediction
# ---------------------------------------------------------------------------
def predict_single(
    customer_data: Dict,
    model: Optional[RandomForestClassifier] = None,
    preprocessor: Optional[ColumnTransformer] = None,
    feature_names: Optional[List[str]] = None,
) -> Dict:
    """Predict churn for a single customer.

    Args:
        customer_data: Dictionary of customer attributes.
        model: Pre-loaded model (loaded automatically if None).
        preprocessor: Pre-loaded preprocessor (loaded automatically if None).
        feature_names: Feature names after preprocessing.

    Returns:
        Dictionary with prediction results.
    """
    if model is None or preprocessor is None:
        model, preprocessor = load_model_and_preprocessor()

    # Build DataFrame from input
    df = pd.DataFrame([customer_data])

    # Ensure all required columns exist
    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    # Preprocess
    X = df[ALL_FEATURES]
    X_processed = preprocessor.transform(X)

    # Predict
    probability = float(model.predict_proba(X_processed)[0, 1])
    prediction = int(probability >= 0.5)
    risk = classify_risk(probability)
    risk_label = get_risk_label(probability)

    # Get feature names if not provided
    if feature_names is None:
        feature_names = (
            NUMERICAL_FEATURES
            + list(preprocessor.named_transformers_["cat"]
                   .named_steps["encoder"]
                   .get_feature_names_out(CATEGORICAL_FEATURES))
        )

    # Top risk factors
    factors = get_top_risk_factors(model, feature_names, X_processed)

    return {
        "churn_prediction": prediction,
        "churn_probability": probability,
        "risk_level": risk,
        "risk_label": risk_label,
        "top_risk_factors": factors,
        "customer_data": customer_data,
    }


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------
def predict_batch(
    df: pd.DataFrame,
    model: Optional[RandomForestClassifier] = None,
    preprocessor: Optional[ColumnTransformer] = None,
) -> pd.DataFrame:
    """Generate churn predictions for a batch of customers.

    Args:
        df: DataFrame with customer features.
        model: Pre-loaded model.
        preprocessor: Pre-loaded preprocessor.

    Returns:
        Original DataFrame augmented with prediction columns.
    """
    if model is None or preprocessor is None:
        model, preprocessor = load_model_and_preprocessor()

    X = df[ALL_FEATURES].copy()
    X_processed = preprocessor.transform(X)

    probabilities = model.predict_proba(X_processed)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    result = df.copy()
    result["ChurnProbability"] = np.round(probabilities, 4)
    result["ChurnPrediction"] = predictions
    result["RiskLevel"] = [classify_risk(p) for p in probabilities]
    result["RiskLabel"] = [get_risk_label(p) for p in probabilities]

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """Demo prediction on a sample customer."""
    print("=" * 60)
    print("ChurnGuard AI — Prediction Engine Demo")
    print("=" * 60)

    sample_customer = {
        "Age": 35,
        "Gender": "Male",
        "Tenure": 3,
        "ContractType": "Month-to-Month",
        "MonthlyCharges": 95.0,
        "TotalCharges": 285.0,
        "PaymentMethod": "Electronic Check",
        "InternetService": "Fiber Optic",
        "TechSupport": "No",
        "OnlineSecurity": "No",
        "DeviceProtection": "No",
        "StreamingServices": "Yes",
        "SupportTickets": 5,
        "SatisfactionScore": 2.0,
        "UsageFrequency": 8,
        "LastLoginDays": 25,
    }

    result = predict_single(sample_customer)

    print(f"\n   Churn Probability: {result['churn_probability']:.1%}")
    print(f"   Risk Level:       {result['risk_level']}")
    print(f"   Top Risk Factors:")
    for f in result["top_risk_factors"]:
        print(f"     • {f['feature']}: {f['importance']:.3f}")


if __name__ == "__main__":
    main()
