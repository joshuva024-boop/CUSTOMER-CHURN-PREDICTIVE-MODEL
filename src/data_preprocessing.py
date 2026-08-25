"""
ChurnGuard AI — Data Preprocessing Pipeline

Implements a reusable Scikit-learn preprocessing pipeline using
ColumnTransformer. Handles missing values, encoding, scaling, and
feature engineering.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Tuple, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RAW_DATA_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "raw" / "customer_churn_data.csv"
PROCESSED_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR: Path = Path(__file__).resolve().parent.parent / "models"

TARGET_COLUMN: str = "Churn"
ID_COLUMN: str = "CustomerID"
TEST_SIZE: float = 0.20
RANDOM_STATE: int = 42

# Feature groups
NUMERICAL_FEATURES: List[str] = [
    "Age", "Tenure", "MonthlyCharges", "TotalCharges",
    "SupportTickets", "SatisfactionScore", "UsageFrequency", "LastLoginDays",
]
CATEGORICAL_FEATURES: List[str] = [
    "Gender", "ContractType", "PaymentMethod",
    "InternetService", "TechSupport", "OnlineSecurity",
    "DeviceProtection", "StreamingServices",
]


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------
def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw customer data from CSV.

    Args:
        path: Path to the raw CSV file.

    Returns:
        Raw DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run generate_dataset.py first."
        )
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} records from {path.name}")
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """Run data quality checks and return a summary report.

    Args:
        df: Raw DataFrame.

    Returns:
        Dictionary with quality metrics.
    """
    report = {
        "shape": df.shape,
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicates": df.duplicated().sum(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }

    print("\n📊 Data Quality Report")
    print(f"   Shape:       {report['shape']}")
    print(f"   Duplicates:  {report['duplicates']}")
    print(f"   Missing values:")
    for col, cnt in report["missing_values"].items():
        if cnt > 0:
            print(f"     • {col}: {cnt} ({report['missing_pct'][col]:.1f}%)")

    return report


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop duplicates, fix dtypes.

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    df = df.drop_duplicates().reset_index(drop=True)

    # Ensure numeric columns are correct dtype
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"   After cleaning: {len(df):,} records")
    return df


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """Build a Scikit-learn ColumnTransformer preprocessing pipeline.

    Returns:
        Configured ColumnTransformer.
    """
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def prepare_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, list]:
    """Separate features and target.

    Args:
        df: Cleaned DataFrame.

    Returns:
        Tuple of (feature DataFrame, target Series, customer IDs).
    """
    customer_ids: list = df[ID_COLUMN].tolist()
    X: pd.DataFrame = pd.DataFrame(df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])
    y: pd.Series = pd.Series(df[TARGET_COLUMN])
    return X, y, customer_ids


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Fraction for the test set.
        random_state: Seed for reproducibility.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"\n   Train set: {len(X_train):,} | Test set: {len(X_test):,}")
    print(f"   Train churn rate: {y_train.mean():.1%}")
    print(f"   Test  churn rate: {y_test.mean():.1%}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Full preprocessing run
# ---------------------------------------------------------------------------
def run_preprocessing() -> dict:
    """Execute the full preprocessing pipeline.

    Returns:
        Dictionary containing all artifacts for downstream use.
    """
    print("=" * 60)
    print("ChurnGuard AI — Data Preprocessing")
    print("=" * 60)

    # Load and inspect
    df = load_data()
    report = inspect_data(df)

    # Clean
    df = clean_data(df)

    # Save processed CSV
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = PROCESSED_DIR / "cleaned_data.csv"
    df.to_csv(processed_path, index=False)
    print(f"   Saved cleaned data to {processed_path.name}")

    # Prepare features
    X, y, customer_ids = prepare_features(df)

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Build & fit preprocessor
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names after transformation
    feature_names = (
        NUMERICAL_FEATURES
        + list(preprocessor.named_transformers_["cat"]
               .named_steps["encoder"]
               .get_feature_names_out(CATEGORICAL_FEATURES))
    )

    print(f"\n   Processed features: {len(feature_names)}")

    # Save preprocessor
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    print(f"   Saved preprocessor to {preprocessor_path.name}")

    # Return artifacts
    return {
        "df": df,
        "X": X,
        "y": y,
        "customer_ids": customer_ids,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_processed": X_train_processed,
        "X_test_processed": X_test_processed,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "quality_report": report,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """Run preprocessing as a standalone script."""
    artifacts = run_preprocessing()
    print("\n✅ Preprocessing complete!")
    print(f"   Total features after encoding: {len(artifacts['feature_names'])}")


if __name__ == "__main__":
    main()
