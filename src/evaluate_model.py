"""
ChurnGuard AI — Model Evaluation

Computes classification metrics, generates confusion matrix, ROC curve,
and feature importance charts. All values come from the actual trained model.
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VIZ_DIR: Path = Path(__file__).resolve().parent.parent / "visualizations"

COLORS = {
    "bg": "#0f1629",
    "card": "#1a1f3a",
    "text": "#e8eaf6",
    "accent_blue": "#6366f1",
    "accent_purple": "#8b5cf6",
    "churn_red": "#ef4444",
    "retain_green": "#22c55e",
    "grid": "#2a2f4a",
}


def _apply_theme() -> None:
    """Apply consistent dark chart theme."""
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["card"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["text"],
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "text.color": COLORS["text"],
        "legend.facecolor": COLORS["card"],
        "legend.edgecolor": COLORS["grid"],
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.3,
        "font.size": 10,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": COLORS["bg"],
    })


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------
def compute_metrics(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Compute all classification metrics from the trained model.

    Args:
        model: Fitted RandomForestClassifier.
        X_test: Preprocessed test features.
        y_test: True test labels.

    Returns:
        Dictionary of metric_name → value.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    print("\n📈 Model Evaluation Metrics")
    print("-" * 40)
    for name, value in metrics.items():
        print(f"   {name:>12s}: {value:.4f}")

    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

    return metrics


# ---------------------------------------------------------------------------
# Visualization generators
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Generate and save a confusion matrix heatmap."""
    _apply_theme()
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="RdPu",
        xticklabels=["Retained", "Churned"],
        yticklabels=["Retained", "Churned"],
        ax=ax, linewidths=1, linecolor=COLORS["grid"],
        annot_kws={"fontsize": 16, "fontweight": "bold"},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "confusion_matrix.png")


def plot_roc_curve(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Generate and save the ROC curve."""
    _apply_theme()
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color=COLORS["accent_purple"], linewidth=2.5,
            label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color=COLORS["grid"], linestyle="--", linewidth=1)
    ax.fill_between(fpr, tpr, alpha=0.15, color=COLORS["accent_purple"])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Random Forest", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    _save(fig, "roc_curve.png")


def plot_feature_importance(
    model: RandomForestClassifier,
    feature_names: List[str],
    top_n: int = 15,
) -> Dict[str, float]:
    """Generate and save a feature importance bar chart.

    Args:
        model: Fitted RandomForestClassifier.
        feature_names: Names of the preprocessed features.
        top_n: Number of top features to display.

    Returns:
        Dictionary of feature_name → importance (sorted descending).
    """
    _apply_theme()
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    # Build result dict (all features, sorted)
    all_sorted = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    importance_dict = {name: float(imp) for name, imp in all_sorted}

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdPu(np.linspace(0.3, 0.9, top_n))[::-1]
    y_pos = range(len(top_features))

    ax.barh(y_pos, top_importances[::-1], color=colors, edgecolor=COLORS["bg"], height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features[::-1], fontsize=10)
    ax.set_xlabel("Importance Score")
    ax.set_title("Top Feature Importances — Random Forest", fontsize=14, fontweight="bold")

    for i, (val, name) in enumerate(zip(top_importances[::-1], top_features[::-1])):
        ax.text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    _save(fig, "feature_importance.png")

    return importance_dict


# ---------------------------------------------------------------------------
# Full evaluation run
# ---------------------------------------------------------------------------
def run_evaluation(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
) -> Dict:
    """Run the complete evaluation pipeline.

    Args:
        model: Fitted model.
        X_test: Preprocessed test features.
        y_test: True test labels.
        feature_names: Feature names after preprocessing.

    Returns:
        Dictionary with metrics, predictions, and feature importances.
    """
    print("=" * 60)
    print("ChurnGuard AI — Model Evaluation")
    print("=" * 60)

    metrics = compute_metrics(model, X_test, y_test)
    plot_confusion_matrix(model, X_test, y_test)
    plot_roc_curve(model, X_test, y_test)
    importance_dict = plot_feature_importance(model, feature_names)

    print("\n✅ Evaluation complete!")

    return {
        "metrics": metrics,
        "feature_importances": importance_dict,
        "y_pred": model.predict(X_test),
        "y_prob": model.predict_proba(X_test)[:, 1],
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _save(fig: plt.Figure, filename: str) -> None:
    """Save figure and close."""
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    path = VIZ_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    print(f"   📊 Saved: {filename}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """Run evaluation as a standalone script."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import joblib
    from data_preprocessing import run_preprocessing

    artifacts = run_preprocessing()
    model_path = Path(__file__).resolve().parent.parent / "models" / "random_forest_model.pkl"
    model = joblib.load(model_path)

    run_evaluation(
        model,
        artifacts["X_test_processed"],
        artifacts["y_test"],
        artifacts["feature_names"],
    )


if __name__ == "__main__":
    main()
