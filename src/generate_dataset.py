"""
ChurnGuard AI — Synthetic Customer Dataset Generator

Generates a realistic synthetic dataset of 5,000+ customer records with
churn labels driven by a logistic probability model (not random assignment),
ensuring meaningful correlations between features and churn outcome.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED: int = 42
NUM_CUSTOMERS: int = 5000
OUTPUT_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_FILE: str = "customer_churn_data.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically-stable sigmoid function."""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x)),
    )


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------
def generate_dataset(
    n_customers: int = NUM_CUSTOMERS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Generate a synthetic customer churn dataset.

    Churn probability is modelled as a logistic function of customer
    attributes so that the resulting dataset contains realistic,
    learnable correlations.

    Args:
        n_customers: Number of customer records to generate.
        seed: Random seed for reproducibility.

    Returns:
        A pandas DataFrame with all customer features and the Churn label.
    """
    rng = np.random.RandomState(seed)

    # --- Demographic features ------------------------------------------------
    customer_ids = [f"CUST-{i:05d}" for i in range(1, n_customers + 1)]
    ages = rng.randint(18, 75, size=n_customers)
    genders = rng.choice(["Male", "Female"], size=n_customers)

    # --- Contract & tenure ---------------------------------------------------
    contract_types = rng.choice(
        ["Month-to-Month", "One Year", "Two Year"],
        size=n_customers,
        p=[0.50, 0.30, 0.20],
    )
    # Tenure correlated with contract type
    tenure = np.zeros(n_customers, dtype=int)
    for i, ct in enumerate(contract_types):
        if ct == "Month-to-Month":
            tenure[i] = max(1, int(rng.exponential(12)))
        elif ct == "One Year":
            tenure[i] = max(6, int(rng.normal(24, 8)))
        else:
            tenure[i] = max(12, int(rng.normal(48, 12)))
    tenure = np.clip(tenure, 1, 72)

    # --- Financial features --------------------------------------------------
    monthly_charges = np.round(rng.uniform(20, 120, size=n_customers), 2)
    total_charges = np.round(monthly_charges * tenure * rng.uniform(0.85, 1.05, size=n_customers), 2)

    payment_methods = rng.choice(
        ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"],
        size=n_customers,
        p=[0.35, 0.20, 0.25, 0.20],
    )

    # --- Service features ----------------------------------------------------
    internet_services = rng.choice(
        ["Fiber Optic", "DSL", "No"],
        size=n_customers,
        p=[0.45, 0.35, 0.20],
    )
    tech_support = rng.choice(["Yes", "No"], size=n_customers, p=[0.40, 0.60])
    online_security = rng.choice(["Yes", "No"], size=n_customers, p=[0.35, 0.65])
    device_protection = rng.choice(["Yes", "No"], size=n_customers, p=[0.38, 0.62])
    streaming_services = rng.choice(["Yes", "No"], size=n_customers, p=[0.45, 0.55])

    # --- Behavioural features ------------------------------------------------
    support_tickets = rng.poisson(2, size=n_customers)
    support_tickets = np.clip(support_tickets, 0, 10)

    satisfaction_scores = rng.normal(3.0, 1.0, size=n_customers)
    satisfaction_scores = np.clip(np.round(satisfaction_scores, 1), 1.0, 5.0)

    usage_frequency = rng.randint(1, 31, size=n_customers)

    last_login_days = rng.exponential(15, size=n_customers).astype(int)
    last_login_days = np.clip(last_login_days, 0, 90)

    # --- Churn probability (logistic model) ----------------------------------
    # Encode categorical features numerically for the logistic score
    contract_score = np.where(
        np.array(contract_types) == "Month-to-Month", 1.2,
        np.where(np.array(contract_types) == "One Year", -0.3, -0.8),
    )
    payment_score = np.where(
        np.array(payment_methods) == "Electronic Check", 0.5, -0.2,
    )
    internet_score = np.where(
        np.array(internet_services) == "Fiber Optic", 0.4,
        np.where(np.array(internet_services) == "DSL", 0.0, -0.3),
    )
    tech_support_score = np.where(np.array(tech_support) == "No", 0.4, -0.3)

    # Combine into a linear predictor
    log_odds = (
        -0.5                                         # intercept
        + contract_score                             # contract type
        + payment_score                              # payment method
        + internet_score                             # internet service
        + tech_support_score                         # tech support
        - 0.03 * tenure                              # longer tenure → less churn
        + 0.008 * monthly_charges                    # higher charges → more churn
        + 0.15 * support_tickets                     # more tickets → more churn
        - 0.35 * satisfaction_scores                 # higher satisfaction → less churn
        - 0.01 * usage_frequency                     # higher usage → less churn
        + 0.01 * last_login_days                     # longer since login → more churn
        + rng.normal(0, 0.3, size=n_customers)       # noise
    )

    churn_prob = _sigmoid(log_odds)
    churn = (rng.uniform(size=n_customers) < churn_prob).astype(int)

    # --- Assemble DataFrame --------------------------------------------------
    df = pd.DataFrame({
        "CustomerID": customer_ids,
        "Age": ages,
        "Gender": genders,
        "Tenure": tenure,
        "ContractType": contract_types,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "PaymentMethod": payment_methods,
        "InternetService": internet_services,
        "TechSupport": tech_support,
        "OnlineSecurity": online_security,
        "DeviceProtection": device_protection,
        "StreamingServices": streaming_services,
        "SupportTickets": support_tickets,
        "SatisfactionScore": satisfaction_scores,
        "UsageFrequency": usage_frequency,
        "LastLoginDays": last_login_days,
        "Churn": churn,
    })

    # Inject a small amount of realistic messiness
    _inject_noise(df, rng)

    return df


def _inject_noise(df: pd.DataFrame, rng: np.random.RandomState) -> None:
    """Inject a small percentage of missing values and edge cases.

    This makes the dataset more realistic for preprocessing exercises.
    """
    n = len(df)
    # ~2% missing TotalCharges
    mask = rng.choice(n, size=int(n * 0.02), replace=False)
    df.loc[mask, "TotalCharges"] = np.nan

    # ~1.5% missing SatisfactionScore
    mask = rng.choice(n, size=int(n * 0.015), replace=False)
    df.loc[mask, "SatisfactionScore"] = np.nan

    # ~1% missing LastLoginDays
    mask = rng.choice(n, size=int(n * 0.01), replace=False)
    df.loc[mask, "LastLoginDays"] = np.nan


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def main() -> None:
    """Generate the dataset and save to CSV."""
    print("=" * 60)
    print("ChurnGuard AI — Synthetic Dataset Generator")
    print("=" * 60)

    df = generate_dataset()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_FILE
    df.to_csv(output_path, index=False)

    print(f"\n✅ Generated {len(df):,} customer records")
    print(f"   Churn rate: {df['Churn'].mean():.1%}")
    print(f"   Saved to:   {output_path}")
    print(f"\nFeature summary:")
    print(df.describe(include="all").to_string())


if __name__ == "__main__":
    main()
