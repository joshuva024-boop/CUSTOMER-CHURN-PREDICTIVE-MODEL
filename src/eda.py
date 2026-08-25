"""
ChurnGuard AI — Exploratory Data Analysis

Generates publication-quality visualizations for customer churn analysis.
All charts are saved as PNG files in the visualizations/ directory.
"""

import sys
import warnings
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VIZ_DIR: Path = Path(__file__).resolve().parent.parent / "visualizations"

# Professional dark theme colors
COLORS = {
    "bg": "#0f1629",
    "card": "#1a1f3a",
    "text": "#e8eaf6",
    "accent_blue": "#6366f1",
    "accent_purple": "#8b5cf6",
    "churn_red": "#ef4444",
    "retain_green": "#22c55e",
    "orange": "#f59e0b",
    "grid": "#2a2f4a",
    "palette": ["#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd", "#6ee7b7", "#f59e0b"],
}

# Custom Seaborn/Matplotlib theme
def _apply_theme() -> None:
    """Apply a dark, professional chart theme."""
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
        "font.family": "sans-serif",
        "font.size": 10,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": COLORS["bg"],
    })


# ---------------------------------------------------------------------------
# Chart generators
# ---------------------------------------------------------------------------
def plot_churn_distribution(df: pd.DataFrame) -> None:
    """1. Overall churn distribution — donut + bar chart."""
    _apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Donut chart
    counts = df["Churn"].value_counts()
    labels = ["Retained", "Churned"]
    colors = [COLORS["retain_green"], COLORS["churn_red"]]
    wedges, texts, autotexts = axes[0].pie(
        counts, labels=labels, autopct="%1.1f%%", startangle=90,
        colors=colors, pctdistance=0.75,
        wedgeprops=dict(width=0.4, edgecolor=COLORS["bg"], linewidth=2),
    )
    for t in autotexts:
        t.set_fontsize(12)
        t.set_fontweight("bold")
    axes[0].set_title("Customer Churn Distribution", fontsize=14, fontweight="bold")

    # Bar chart
    sns.countplot(data=df, x="Churn", ax=axes[1], hue="Churn", palette=colors, legend=False, edgecolor=COLORS["bg"])
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(labels)
    axes[1].set_title("Churn Count", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Count")
    for p in axes[1].patches:
        axes[1].annotate(
            f"{int(p.get_height()):,}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center", va="bottom", fontweight="bold", fontsize=12,
        )

    plt.tight_layout()
    _save(fig, "01_churn_distribution.png")


def plot_churn_by_contract(df: pd.DataFrame) -> None:
    """2. Churn rate by contract type."""
    _apply_theme()
    fig, ax = plt.subplots(figsize=(10, 5))
    churn_series = pd.Series(df.groupby("ContractType")["Churn"].mean()).sort_values(ascending=False)
    categories = list(churn_series.index)
    churn_rates = [float(v) for v in churn_series.values]
    bars = ax.barh(
        y=categories, width=churn_rates,
        color=[COLORS["churn_red"], COLORS["orange"], COLORS["retain_green"]],
        edgecolor=COLORS["bg"], height=0.5,
    )
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_xlabel("Churn Rate")
    ax.set_title("Churn Rate by Contract Type", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    for bar, val in zip(bars, churn_rates):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontweight="bold", fontsize=11)
    plt.tight_layout()
    _save(fig, "02_churn_by_contract.png")


def plot_churn_by_tenure(df: pd.DataFrame) -> None:
    """3. Churn by tenure — overlaid histograms."""
    _apply_theme()
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color in [(0, COLORS["retain_green"]), (1, COLORS["churn_red"])]:
        subset = df[df["Churn"] == label]["Tenure"]
        ax.hist(subset, bins=30, alpha=0.6, color=color,
                label="Churned" if label else "Retained", edgecolor=COLORS["bg"])
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Count")
    ax.set_title("Tenure Distribution by Churn Status", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _save(fig, "03_churn_by_tenure.png")


def plot_churn_by_monthly_charges(df: pd.DataFrame) -> None:
    """4. Monthly charges vs churn — violin plot."""
    _apply_theme()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(
        data=df, x="Churn", y="MonthlyCharges", ax=ax,
        hue="Churn", palette=[COLORS["retain_green"], COLORS["churn_red"]],
        legend=False, inner="quartile", linewidth=1.2,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Retained", "Churned"])

    ax.set_title("Monthly Charges by Churn Status", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    plt.tight_layout()
    _save(fig, "04_churn_by_monthly_charges.png")


def plot_churn_by_payment_method(df: pd.DataFrame) -> None:
    """5. Churn rate by payment method."""
    _apply_theme()
    fig, ax = plt.subplots(figsize=(10, 5))
    churn_series = pd.Series(df.groupby("PaymentMethod")["Churn"].mean()).sort_values(ascending=False)
    categories = list(churn_series.index)
    churn_rates = [float(v) for v in churn_series.values]
    colors = [COLORS["palette"][i] for i in range(len(categories))]
    bars = ax.barh(y=categories, width=churn_rates, color=colors,
                   edgecolor=COLORS["bg"], height=0.5)
    ax.set_xlabel("Churn Rate")
    ax.set_title("Churn Rate by Payment Method", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    for bar, val in zip(bars, churn_rates):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontweight="bold", fontsize=11)
    plt.tight_layout()
    _save(fig, "05_churn_by_payment_method.png")


def plot_churn_by_support_tickets(df: pd.DataFrame) -> None:
    """6. Churn rate by number of support tickets."""
    _apply_theme()
    fig, ax = plt.subplots(figsize=(10, 5))
    churn_series = pd.Series(df.groupby("SupportTickets")["Churn"].mean())
    categories = list(churn_series.index)
    churn_rates = [float(v) for v in churn_series.values]
    ax.bar(x=categories, height=churn_rates,
           color=COLORS["accent_blue"], edgecolor=COLORS["bg"])
    ax.set_xlabel("Number of Support Tickets")
    ax.set_ylabel("Churn Rate")
    ax.set_title("Churn Rate by Support Tickets", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    plt.tight_layout()
    _save(fig, "06_churn_by_support_tickets.png")


def plot_churn_by_satisfaction(df: pd.DataFrame) -> None:
    """7. Satisfaction score distribution by churn."""
    _apply_theme()
    fig, ax = plt.subplots(figsize=(10, 5))
    retained = df[df["Churn"] == 0]["SatisfactionScore"].dropna()
    churned = df[df["Churn"] == 1]["SatisfactionScore"].dropna()
    ax.hist(retained, bins=20, alpha=0.6, color=COLORS["retain_green"],
            label="Retained", edgecolor=COLORS["bg"])
    ax.hist(churned, bins=20, alpha=0.6, color=COLORS["churn_red"],
            label="Churned", edgecolor=COLORS["bg"])
    ax.set_xlabel("Satisfaction Score")
    ax.set_ylabel("Count")
    ax.set_title("Satisfaction Score Distribution by Churn", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _save(fig, "07_churn_by_satisfaction.png")


def plot_churn_by_internet_service(df: pd.DataFrame) -> None:
    """8. Churn rate by internet service type."""
    _apply_theme()
    fig, ax = plt.subplots(figsize=(10, 5))
    churn_series = pd.Series(df.groupby("InternetService")["Churn"].mean()).sort_values(ascending=False)
    categories = list(churn_series.index)
    churn_rates = [float(v) for v in churn_series.values]
    colors = [COLORS["churn_red"], COLORS["orange"], COLORS["retain_green"]]
    bars = ax.barh(y=categories, width=churn_rates, color=colors[:len(categories)],
                   edgecolor=COLORS["bg"], height=0.5)
    ax.set_xlabel("Churn Rate")
    ax.set_title("Churn Rate by Internet Service", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    for bar, val in zip(bars, churn_rates):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontweight="bold", fontsize=11)
    plt.tight_layout()
    _save(fig, "08_churn_by_internet_service.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """9. Correlation heatmap of numerical features."""
    _apply_theme()
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="dark", as_cmap=True)
    sns.heatmap(
        corr, mask=mask, cmap=cmap, ax=ax,
        vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f", linewidths=0.5,
        linecolor=COLORS["grid"],
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "09_correlation_heatmap.png")


def plot_customer_segments(df: pd.DataFrame) -> None:
    """10. Key customer segments — multi-panel overview."""
    _apply_theme()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Age distribution
    sns.histplot(data=df, x="Age", hue="Churn", ax=axes[0, 0],
                 palette=[COLORS["retain_green"], COLORS["churn_red"]],
                 kde=True, edgecolor=COLORS["bg"], alpha=0.6)
    axes[0, 0].set_title("Age Distribution", fontweight="bold")
    axes[0, 0].legend(["Retained", "Churned"])

    # Usage frequency
    sns.boxplot(data=df, x="Churn", y="UsageFrequency", ax=axes[0, 1],
                hue="Churn", palette=[COLORS["retain_green"], COLORS["churn_red"]], legend=False)
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_xticklabels(["Retained", "Churned"])
    axes[0, 1].set_title("Usage Frequency", fontweight="bold")

    # Gender split
    ct = pd.crosstab(df["Gender"], df["Churn"], normalize="index")
    ct.columns = ["Retained", "Churned"]
    ct.plot(kind="bar", stacked=True, ax=axes[1, 0],
            color=[COLORS["retain_green"], COLORS["churn_red"]],
            edgecolor=COLORS["bg"])
    axes[1, 0].set_title("Churn by Gender", fontweight="bold")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=0)
    axes[1, 0].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # Total charges
    sns.boxplot(data=df, x="Churn", y="TotalCharges", ax=axes[1, 1],
                hue="Churn", palette=[COLORS["retain_green"], COLORS["churn_red"]], legend=False)
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_xticklabels(["Retained", "Churned"])
    axes[1, 1].set_title("Total Charges", fontweight="bold")


    plt.suptitle("Customer Segments Overview", fontsize=16, fontweight="bold",
                 color=COLORS["text"], y=1.02)
    plt.tight_layout()
    _save(fig, "10_customer_segments.png")


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


def run_eda(df: pd.DataFrame) -> None:
    """Run all EDA visualizations.

    Args:
        df: Cleaned DataFrame with Churn column.
    """
    print("=" * 60)
    print("ChurnGuard AI — Exploratory Data Analysis")
    print("=" * 60)

    plot_churn_distribution(df)
    plot_churn_by_contract(df)
    plot_churn_by_tenure(df)
    plot_churn_by_monthly_charges(df)
    plot_churn_by_payment_method(df)
    plot_churn_by_support_tickets(df)
    plot_churn_by_satisfaction(df)
    plot_churn_by_internet_service(df)
    plot_correlation_heatmap(df)
    plot_customer_segments(df)

    print(f"\n✅ All visualizations saved to {VIZ_DIR}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """Run EDA as a standalone script."""
    data_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "cleaned_data.csv"
    if not data_path.exists():
        data_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "customer_churn_data.csv"
    df = pd.read_csv(data_path)
    run_eda(df)


if __name__ == "__main__":
    main()
