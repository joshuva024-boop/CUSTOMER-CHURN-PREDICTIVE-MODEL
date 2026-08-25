"""
ChurnGuard AI — Model Training

Trains a Random Forest Classifier with hyperparameter tuning via
RandomizedSearchCV, class imbalance handling, and cross-validation.
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS_DIR: Path = Path(__file__).resolve().parent.parent / "models"
RANDOM_STATE: int = 42
CV_FOLDS: int = 5
N_ITER_SEARCH: int = 50

# Hyperparameter search space
PARAM_DISTRIBUTIONS: Dict[str, Any] = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [5, 10, 15, 20, 25, None],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", None],
    "class_weight": ["balanced", "balanced_subsample", None],
    "criterion": ["gini", "entropy"],
}


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------
def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RandomForestClassifier:
    """Train a baseline Random Forest with default parameters.

    Args:
        X_train: Preprocessed training features.
        y_train: Training target.

    Returns:
        Fitted RandomForestClassifier.
    """
    print("\n🌲 Training baseline Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print(f"   Baseline training accuracy: {model.score(X_train, y_train):.4f}")
    return model


def train_tuned(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = N_ITER_SEARCH,
    cv_folds: int = CV_FOLDS,
) -> RandomForestClassifier:
    """Train a Random Forest with hyperparameter tuning.

    Uses RandomizedSearchCV with stratified k-fold cross-validation,
    optimizing for F1-score (prioritizing churn detection over raw accuracy).

    Args:
        X_train: Preprocessed training features.
        y_train: Training target.
        n_iter: Number of parameter settings sampled.
        cv_folds: Number of CV folds.

    Returns:
        Best fitted RandomForestClassifier.
    """
    print(f"\n🔬 Hyperparameter tuning ({n_iter} iterations, {cv_folds}-fold CV)...")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        cv=cv,
        scoring="f1",  # Prioritize churn recall/precision balance
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    print(f"\n   Best CV F1 score: {search.best_score_:.4f}")
    print(f"   Best parameters:")
    for param, value in search.best_params_.items():
        print(f"     • {param}: {value}")

    return search.best_estimator_


def save_model(
    model: RandomForestClassifier,
    filename: str = "random_forest_model.pkl",
) -> Path:
    """Save the trained model to disk.

    Args:
        model: Fitted RandomForestClassifier.
        filename: Output filename.

    Returns:
        Path to the saved model file.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    print(f"\n   💾 Model saved to {path}")
    return path


def run_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tune: bool = True,
) -> RandomForestClassifier:
    """Execute the full training pipeline.

    Args:
        X_train: Preprocessed training features.
        y_train: Training target.
        tune: Whether to run hyperparameter tuning.

    Returns:
        Trained RandomForestClassifier (best model).
    """
    print("=" * 60)
    print("ChurnGuard AI — Model Training")
    print("=" * 60)

    # Baseline
    baseline = train_baseline(X_train, y_train)

    # Tuned model
    if tune:
        best_model = train_tuned(X_train, y_train)
    else:
        best_model = baseline

    # Save
    save_model(best_model)

    print("\n✅ Training complete!")
    return best_model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """Run training as a standalone script (requires preprocessing first)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_preprocessing import run_preprocessing

    artifacts = run_preprocessing()
    model = run_training(artifacts["X_train_processed"], artifacts["y_train"])
    return model


if __name__ == "__main__":
    main()
