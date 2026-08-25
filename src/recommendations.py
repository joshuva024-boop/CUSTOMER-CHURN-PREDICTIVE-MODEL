"""
ChurnGuard AI — Retention Recommendation Engine

Rule-based recommendation layer that maps risk levels and customer
attributes to actionable retention strategies. Each recommendation
includes a rationale connected to the customer's specific profile.
"""

import sys
from typing import Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# Recommendation rules
# ---------------------------------------------------------------------------
def get_recommendation(
    risk_label: str,
    customer_data: Dict,
    top_risk_factors: Optional[List[Dict]] = None,
) -> Dict[str, str]:
    """Generate a retention recommendation based on risk level and customer attributes.

    Args:
        risk_label: "High", "Medium", or "Low".
        customer_data: Dictionary of customer attributes.
        top_risk_factors: Optional list of top contributing factors.

    Returns:
        Dictionary with 'action', 'priority', 'details', and 'rationale' keys.
    """
    monthly_charges = customer_data.get("MonthlyCharges", 0)
    satisfaction = customer_data.get("SatisfactionScore", 3.0)
    support_tickets = customer_data.get("SupportTickets", 0)
    tenure = customer_data.get("Tenure", 12)
    tech_support = customer_data.get("TechSupport", "Yes")
    contract_type = customer_data.get("ContractType", "")
    online_security = customer_data.get("OnlineSecurity", "Yes")

    if risk_label == "High":
        return _high_risk_recommendation(
            monthly_charges, satisfaction, support_tickets,
            tenure, tech_support, contract_type, online_security,
        )
    elif risk_label == "Medium":
        return _medium_risk_recommendation(
            monthly_charges, satisfaction, support_tickets,
            tenure, contract_type,
        )
    else:
        return _low_risk_recommendation(tenure)


def _high_risk_recommendation(
    monthly_charges: float,
    satisfaction: float,
    support_tickets: int,
    tenure: int,
    tech_support: str,
    contract_type: str,
    online_security: str,
) -> Dict[str, str]:
    """Generate recommendations for HIGH-risk customers."""

    # Priority-ordered rule checks
    if monthly_charges > 80 and satisfaction < 3.0:
        return {
            "action": "Offer personalized discount + schedule customer success call",
            "priority": "🔴 URGENT",
            "details": (
                "This customer has high monthly charges and low satisfaction — "
                "a strong churn signal. Offer a 15–20% discount on their current "
                "plan and schedule a dedicated customer success call within 48 hours."
            ),
            "rationale": (
                f"Monthly charges (${monthly_charges:.0f}) are above average "
                f"and satisfaction ({satisfaction}/5) is critically low."
            ),
        }

    if support_tickets >= 4:
        return {
            "action": "Escalate to priority support team",
            "priority": "🔴 URGENT",
            "details": (
                "Frequent support interactions indicate unresolved issues. "
                "Assign a dedicated support agent, escalate outstanding tickets, "
                "and follow up with a satisfaction survey after resolution."
            ),
            "rationale": (
                f"Customer has {support_tickets} support tickets, indicating "
                "persistent unresolved issues driving frustration."
            ),
        }

    if satisfaction < 2.5:
        return {
            "action": "Initiate customer-success intervention",
            "priority": "🔴 URGENT",
            "details": (
                "Low satisfaction is a leading churn indicator. Schedule a "
                "1-on-1 call with a customer success manager to understand "
                "pain points and create a personalized improvement plan."
            ),
            "rationale": (
                f"Satisfaction score ({satisfaction}/5) is significantly below "
                "average, indicating deep dissatisfaction."
            ),
        }

    if tenure <= 6:
        return {
            "action": "Launch onboarding assistance program",
            "priority": "🔴 HIGH",
            "details": (
                "New customers who churn early likely had a poor onboarding "
                "experience. Assign an onboarding specialist, send guided "
                "tutorials, and schedule a 30-day check-in call."
            ),
            "rationale": (
                f"Customer has only {tenure} months of tenure — early churn "
                "suggests onboarding gaps."
            ),
        }

    if monthly_charges > 70:
        return {
            "action": "Offer plan optimization and loyalty discount",
            "priority": "🔴 HIGH",
            "details": (
                "High charges without perceived value drive churn. Review the "
                "customer's usage patterns and recommend a right-sized plan. "
                "Offer a 10–15% loyalty discount for annual commitment."
            ),
            "rationale": (
                f"Monthly charges (${monthly_charges:.0f}) may exceed perceived "
                "value — plan optimization could improve retention."
            ),
        }

    if tech_support == "No":
        return {
            "action": "Offer complimentary tech support add-on",
            "priority": "🔴 HIGH",
            "details": (
                "Customers without tech support are more likely to churn when "
                "they encounter issues. Offer a free trial of premium tech "
                "support to demonstrate its value."
            ),
            "rationale": (
                "Customer lacks tech support — providing this could reduce "
                "frustration and increase stickiness."
            ),
        }

    if contract_type == "Month-to-Month":
        return {
            "action": "Incentivize contract upgrade with exclusive offer",
            "priority": "🔴 HIGH",
            "details": (
                "Month-to-month customers have the lowest switching cost. "
                "Offer an exclusive discount (15–25% off) for upgrading to "
                "an annual or two-year contract."
            ),
            "rationale": (
                "Month-to-month contract type is a top churn risk factor — "
                "longer commitments significantly reduce churn probability."
            ),
        }

    # Fallback high-risk recommendation
    return {
        "action": "Schedule personal retention outreach",
        "priority": "🔴 HIGH",
        "details": (
            "This customer shows high churn risk across multiple factors. "
            "Schedule a personalized outreach call to understand their needs "
            "and offer a tailored retention package."
        ),
        "rationale": "Multiple risk factors indicate elevated churn probability.",
    }


def _medium_risk_recommendation(
    monthly_charges: float,
    satisfaction: float,
    support_tickets: int,
    tenure: int,
    contract_type: str,
) -> Dict[str, str]:
    """Generate recommendations for MEDIUM-risk customers."""

    if satisfaction < 3.0:
        return {
            "action": "Send personalized engagement campaign + satisfaction survey",
            "priority": "🟠 MEDIUM",
            "details": (
                "Moderate risk with below-average satisfaction. Send a "
                "personalized email campaign highlighting features the "
                "customer may not be using, paired with a short satisfaction survey."
            ),
            "rationale": (
                f"Satisfaction ({satisfaction}/5) is below average — proactive "
                "engagement can prevent escalation to high risk."
            ),
        }

    if contract_type == "Month-to-Month" and tenure > 12:
        return {
            "action": "Offer loyalty upgrade with annual contract incentive",
            "priority": "🟠 MEDIUM",
            "details": (
                "Long-tenure month-to-month customer could be locked in with "
                "the right incentive. Offer a loyalty discount for upgrading "
                "to an annual contract."
            ),
            "rationale": (
                f"Customer has been month-to-month for {tenure} months — "
                "an annual contract offer could reduce risk significantly."
            ),
        }

    if support_tickets >= 3:
        return {
            "action": "Proactive support follow-up",
            "priority": "🟠 MEDIUM",
            "details": (
                "Multiple support tickets suggest friction. Schedule a "
                "proactive check-in call and ensure all tickets are resolved "
                "satisfactorily."
            ),
            "rationale": (
                f"Customer has {support_tickets} support tickets — "
                "proactive follow-up can prevent escalation."
            ),
        }

    return {
        "action": "Launch targeted engagement campaign",
        "priority": "🟠 MEDIUM",
        "details": (
            "Send personalized content highlighting product features, "
            "success stories, and exclusive offers. Include a feedback "
            "mechanism to catch emerging issues early."
        ),
        "rationale": (
            "Moderate churn risk — targeted engagement is the most "
            "cost-effective intervention at this level."
        ),
    }


def _low_risk_recommendation(tenure: int) -> Dict[str, str]:
    """Generate recommendations for LOW-risk customers."""

    if tenure > 36:
        return {
            "action": "Enroll in VIP loyalty program",
            "priority": "🟢 LOW",
            "details": (
                "This is a loyal, long-tenure customer. Enroll them in "
                "the VIP loyalty program with exclusive perks, early access "
                "to new features, and personalized thank-you messages."
            ),
            "rationale": (
                f"Customer has {tenure} months of tenure and low churn risk — "
                "reward loyalty to strengthen the relationship."
            ),
        }

    return {
        "action": "Include in loyalty rewards campaign",
        "priority": "🟢 LOW",
        "details": (
            "Low-risk customers still benefit from engagement. Include "
            "them in the loyalty rewards program, send occasional "
            "appreciation messages, and offer referral bonuses."
        ),
        "rationale": (
            "Low churn probability — maintain engagement with lightweight "
            "loyalty initiatives."
        ),
    }


# ---------------------------------------------------------------------------
# Batch recommendations
# ---------------------------------------------------------------------------
def generate_batch_recommendations(df) -> List[Dict[str, str]]:
    """Generate recommendations for a batch of customers.

    Args:
        df: DataFrame with RiskLabel column and customer attributes.

    Returns:
        List of recommendation dictionaries.
    """
    recommendations = []
    for _, row in df.iterrows():
        customer_data = row.to_dict()
        risk_label = customer_data.get("RiskLabel", "Low")
        rec = get_recommendation(risk_label, customer_data)
        recommendations.append(rec)
    return recommendations


# ---------------------------------------------------------------------------
# Business insights generator
# ---------------------------------------------------------------------------
def generate_insights(df) -> List[str]:
    """Generate data-driven business insights from the dataset.

    Only returns insights that are supported by the actual data patterns.

    Args:
        df: DataFrame with Churn column and customer features.

    Returns:
        List of insight strings.
    """
    insights = []

    # Contract type insight
    if "ContractType" in df.columns and "Churn" in df.columns:
        churn_by_contract = df.groupby("ContractType")["Churn"].mean()
        if "Month-to-Month" in churn_by_contract.index:
            mtm_rate = churn_by_contract["Month-to-Month"]
            avg_rate = df["Churn"].mean()
            if avg_rate > 0 and mtm_rate > avg_rate * 1.2:
                insights.append(
                    f"📊 Customers with month-to-month contracts have a "
                    f"{mtm_rate:.0%} churn rate — {mtm_rate/avg_rate:.1f}x "
                    f"higher than average. Consider incentivizing annual contracts."
                )

    # Support tickets insight
    if "SupportTickets" in df.columns and "Churn" in df.columns:
        high_tickets = df[df["SupportTickets"] >= 4]["Churn"].mean()
        low_tickets = df[df["SupportTickets"] <= 1]["Churn"].mean()
        if not pd.isna(high_tickets) and not pd.isna(low_tickets) and high_tickets > low_tickets * 1.3:
            insights.append(
                f"🎫 Customers with 4+ support tickets churn at {high_tickets:.0%} "
                f"vs {low_tickets:.0%} for those with ≤1 ticket. "
                f"Proactive support reduces churn risk."
            )

    # Tenure insight
    if "Tenure" in df.columns and "Churn" in df.columns:
        short_tenure = df[df["Tenure"] <= 6]["Churn"].mean()
        long_tenure = df[df["Tenure"] > 36]["Churn"].mean()
        if long_tenure > 0 and short_tenure > long_tenure * 1.5:
            insights.append(
                f"⏱️ New customers (≤6 months) churn at {short_tenure:.0%} — "
                f"{short_tenure/long_tenure:.1f}x higher than long-tenure customers "
                f"({long_tenure:.0%}). Early engagement is critical."
            )

    # Monthly charges insight
    if "MonthlyCharges" in df.columns and "Churn" in df.columns:
        median_charges = df["MonthlyCharges"].median()
        high_charge_churn = df[df["MonthlyCharges"] > median_charges]["Churn"].mean()
        low_charge_churn = df[df["MonthlyCharges"] <= median_charges]["Churn"].mean()
        if low_charge_churn > 0 and high_charge_churn > low_charge_churn * 1.2:
            insights.append(
                f"💰 Higher-paying customers (>${median_charges:.0f}/mo) churn at "
                f"{high_charge_churn:.0%} vs {low_charge_churn:.0%} for lower-paying "
                f"customers. Price sensitivity is a key factor."
            )

    # Satisfaction insight
    if "SatisfactionScore" in df.columns and "Churn" in df.columns:
        low_sat = df[df["SatisfactionScore"] < 2.5]["Churn"].mean()
        high_sat = df[df["SatisfactionScore"] >= 4.0]["Churn"].mean()
        if high_sat > 0 and low_sat > high_sat * 1.5:
            insights.append(
                f"⭐ Customers with satisfaction below 2.5 churn at {low_sat:.0%} — "
                f"{low_sat/high_sat:.1f}x higher than satisfied customers "
                f"({high_sat:.0%}). Customer experience is paramount."
            )

    # Tech support insight
    if "TechSupport" in df.columns and "Churn" in df.columns:
        no_support_churn = df[df["TechSupport"] == "No"]["Churn"].mean()
        yes_support_churn = df[df["TechSupport"] == "Yes"]["Churn"].mean()
        if yes_support_churn > 0 and no_support_churn > yes_support_churn * 1.2:
            insights.append(
                f"🛠️ Customers without tech support churn at {no_support_churn:.0%} "
                f"vs {yes_support_churn:.0%} with support. Offering tech support "
                f"as a default add-on could improve retention."
            )

    # Internet service insight
    if "InternetService" in df.columns and "Churn" in df.columns:
        churn_by_internet = df.groupby("InternetService")["Churn"].mean()
        if "Fiber Optic" in churn_by_internet.index:
            fiber_rate = churn_by_internet["Fiber Optic"]
            avg_rate = df["Churn"].mean()
            if avg_rate > 0 and fiber_rate > avg_rate * 1.1:
                insights.append(
                    f"🌐 Fiber optic customers churn at {fiber_rate:.0%} — "
                    f"higher than average ({avg_rate:.0%}). Higher expectations "
                    f"may not be met. Review fiber service quality."
                )

    # Payment method insight
    if "PaymentMethod" in df.columns and "Churn" in df.columns:
        churn_by_payment = df.groupby("PaymentMethod")["Churn"].mean()
        if "Electronic Check" in churn_by_payment.index:
            echeck_rate = churn_by_payment["Electronic Check"]
            avg_rate = df["Churn"].mean()
            if avg_rate > 0 and echeck_rate > avg_rate * 1.2:
                insights.append(
                    f"💳 Electronic check users churn at {echeck_rate:.0%} — "
                    f"significantly above average. Encourage migration to "
                    f"auto-pay methods (bank transfer, credit card)."
                )

    return insights


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """Demo the recommendation engine."""
    print("=" * 60)
    print("ChurnGuard AI — Recommendation Engine Demo")
    print("=" * 60)

    sample = {
        "MonthlyCharges": 95.0,
        "SatisfactionScore": 1.8,
        "SupportTickets": 6,
        "Tenure": 3,
        "TechSupport": "No",
        "ContractType": "Month-to-Month",
        "OnlineSecurity": "No",
    }

    rec = get_recommendation("High", sample)
    print(f"\n   Priority: {rec['priority']}")
    print(f"   Action:   {rec['action']}")
    print(f"   Details:  {rec['details']}")
    print(f"   Reason:   {rec['rationale']}")


if __name__ == "__main__":
    main()
