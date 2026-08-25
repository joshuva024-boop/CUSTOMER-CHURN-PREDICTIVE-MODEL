"""
ChurnGuard AI — Streamlit Dashboard

A modern, professional SaaS-style customer retention intelligence platform.
All metrics, charts, and predictions come from the actual trained Random Forest model.

Launch: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

from prediction import (
    predict_batch,
    predict_single,
    load_model_and_preprocessor,
    classify_risk,
    get_risk_label,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)
from recommendations import get_recommendation, generate_insights
from evaluate_model import compute_metrics

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ChurnGuard AI — Customer Retention Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Dark Navy / Purple SaaS Theme
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global theme */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #0f1629 50%, #121832 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1230 0%, #141a3d 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown label {
        color: #e8eaf6 !important;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #1a1f3a 0%, #232848 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.2);
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #8b5cf6, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 8px 0;
    }
    .kpi-label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    .kpi-icon {
        font-size: 1.8rem;
        margin-bottom: 4px;
    }

    /* Section headers */
    .section-header {
        color: #e8eaf6;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 32px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
    }

    /* Risk badges */
    .risk-high {
        background: linear-gradient(135deg, #dc2626, #ef4444);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .risk-medium {
        background: linear-gradient(135deg, #d97706, #f59e0b);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .risk-low {
        background: linear-gradient(135deg, #16a34a, #22c55e);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    /* Cards */
    .info-card {
        background: linear-gradient(135deg, #1a1f3a 0%, #232848 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    /* Insight cards */
    .insight-card {
        background: linear-gradient(135deg, #1e2243 0%, #252a50 100%);
        border-left: 4px solid #8b5cf6;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin: 10px 0;
        color: #e8eaf6;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Risk meter */
    .risk-meter-container {
        background: linear-gradient(135deg, #1a1f3a 0%, #232848 100%);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }

    /* Header brand */
    .brand-header {
        text-align: center;
        padding: 20px 0 10px 0;
    }
    .brand-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #8b5cf6, #6366f1, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .brand-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 4px;
    }

    /* Table styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Plotly chart backgrounds */
    .js-plotly-plot .plotly .main-svg {
        border-radius: 12px;
    }

    /* Metric styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1f3a 0%, #232848 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 16px;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 24px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #94a3b8;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.2);
        color: #e8eaf6;
    }
</style>
"""

# ---------------------------------------------------------------------------
# Plotly theme
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(26, 31, 58, 0.8)",
    plot_bgcolor="rgba(26, 31, 58, 0.5)",
    font=dict(color="#e8eaf6", family="Inter"),
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(bgcolor="rgba(26, 31, 58, 0.6)", bordercolor="rgba(99,102,241,0.2)"),
)

COLOR_MAP = {
    "primary": "#6366f1",
    "secondary": "#8b5cf6",
    "accent": "#a78bfa",
    "red": "#ef4444",
    "orange": "#f59e0b",
    "green": "#22c55e",
    "blue_light": "#818cf8",
}


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data():
    """Load and prepare dataset with predictions."""
    # Try processed first, then raw
    processed_path = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
    raw_path = PROJECT_ROOT / "data" / "raw" / "customer_churn_data.csv"

    if processed_path.exists():
        df = pd.read_csv(processed_path)
    elif raw_path.exists():
        df = pd.read_csv(raw_path)
    else:
        st.error("❌ No dataset found. Run `python src/generate_dataset.py` first.")
        st.stop()

    return df


@st.cache_resource
def load_model():
    """Load the trained model and preprocessor."""
    try:
        model, preprocessor = load_model_and_preprocessor()
        return model, preprocessor
    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        st.stop()


@st.cache_data(ttl=3600)
def get_predictions(_model, _preprocessor, df):
    """Generate batch predictions."""
    result = predict_batch(df, _model, _preprocessor)
    return result


@st.cache_data(ttl=3600)
def get_model_metrics(_model, _preprocessor, df):
    """Compute model metrics on test set."""
    from data_preprocessing import (
        prepare_features, split_data, NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
    )
    X, y, _ = prepare_features(df)
    _, X_test, _, y_test = split_data(X, y)
    X_test_processed = _preprocessor.transform(X_test)

    feature_names = (
        NUMERICAL_FEATURES
        + list(_preprocessor.named_transformers_["cat"]
               .named_steps["encoder"]
               .get_feature_names_out(CATEGORICAL_FEATURES))
    )

    metrics = compute_metrics(_model, X_test_processed, y_test)

    # Additional data for charts
    y_pred = _model.predict(X_test_processed)
    y_prob = _model.predict_proba(X_test_processed)[:, 1]

    from sklearn.metrics import confusion_matrix, roc_curve
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    importances = _model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).head(15)

    return metrics, cm, fpr, tpr, importance_df


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def render_kpi_card(icon: str, label: str, value: str, color: str = "") -> str:
    """Generate HTML for a KPI card."""
    return f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value" style="{f'background: {color}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;' if color else ''}">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """


def render_risk_badge(risk_label: str) -> str:
    """Generate HTML risk badge."""
    css_class = f"risk-{risk_label.lower()}"
    return f'<span class="{css_class}">{risk_label.upper()} RISK</span>'


def create_gauge_chart(probability: float) -> go.Figure:
    """Create a risk gauge chart."""
    color = "#ef4444" if probability >= 0.7 else "#f59e0b" if probability >= 0.4 else "#22c55e"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 48, "color": "#e8eaf6"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8",
                     "tickfont": {"color": "#94a3b8"}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(26, 31, 58, 0.5)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(34, 197, 94, 0.15)"},
                {"range": [40, 70], "color": "rgba(245, 158, 11, 0.15)"},
                {"range": [70, 100], "color": "rgba(239, 68, 68, 0.15)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.8,
                "value": probability * 100,
            },
        },
    ))
    fig.update_layout(
        height=280,
        **PLOTLY_LAYOUT,
        margin=dict(l=30, r=30, t=30, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------
def page_dashboard(df_pred: pd.DataFrame, df_raw: pd.DataFrame) -> None:
    """Main dashboard with KPIs and overview charts."""

    # KPI calculations
    total_customers = len(df_pred)
    predicted_churners = (df_pred["ChurnPrediction"] == 1).sum()
    high_risk = (df_pred["RiskLabel"] == "High").sum()
    avg_prob = df_pred["ChurnProbability"].mean()
    retention_opp = total_customers - predicted_churners

    # KPI Row
    st.markdown('<div class="section-header">📊 Key Performance Indicators</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    with cols[0]:
        st.markdown(render_kpi_card("👥", "Total Customers", f"{total_customers:,}"), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(render_kpi_card("⚠️", "Predicted Churners", f"{predicted_churners:,}",
                                    "linear-gradient(135deg, #ef4444, #dc2626)"), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(render_kpi_card("🔴", "High-Risk Customers", f"{high_risk:,}",
                                    "linear-gradient(135deg, #ef4444, #f59e0b)"), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(render_kpi_card("📈", "Avg Churn Probability", f"{avg_prob:.1%}",
                                    "linear-gradient(135deg, #f59e0b, #eab308)"), unsafe_allow_html=True)
    with cols[4]:
        st.markdown(render_kpi_card("✅", "Retention Opportunity", f"{retention_opp:,}",
                                    "linear-gradient(135deg, #22c55e, #16a34a)"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Churn overview charts
    st.markdown('<div class="section-header">🔍 Churn Overview</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        # Churn vs Retained donut
        churn_counts = df_pred["ChurnPrediction"].value_counts().reset_index()
        churn_counts.columns = ["Status", "Count"]
        churn_counts["Status"] = churn_counts["Status"].map({0: "Retained", 1: "Churned"})
        fig = px.pie(
            churn_counts, values="Count", names="Status",
            color="Status",
            color_discrete_map={"Retained": COLOR_MAP["green"], "Churned": COLOR_MAP["red"]},
            hole=0.5,
        )
        fig.update_layout(title="Churn vs Retained", **PLOTLY_LAYOUT)
        fig.update_traces(textposition="inside", textinfo="percent+label",
                         textfont=dict(size=14, color="white"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Churn probability distribution
        fig = px.histogram(
            df_pred, x="ChurnProbability", nbins=40,
            color_discrete_sequence=[COLOR_MAP["secondary"]],
            labels={"ChurnProbability": "Churn Probability"},
        )
        fig.update_layout(title="Churn Probability Distribution", **PLOTLY_LAYOUT,
                         yaxis_title="Count", xaxis_title="Probability")
        st.plotly_chart(fig, use_container_width=True)

    # Risk breakdown
    col3, col4 = st.columns(2)

    with col3:
        risk_counts = df_pred["RiskLabel"].value_counts().reset_index()
        risk_counts.columns = ["Risk", "Count"]
        fig = px.bar(
            risk_counts, x="Risk", y="Count",
            color="Risk",
            color_discrete_map={"High": COLOR_MAP["red"], "Medium": COLOR_MAP["orange"], "Low": COLOR_MAP["green"]},
        )
        fig.update_layout(title="Risk Level Breakdown", **PLOTLY_LAYOUT, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Churn by contract type
        if "ContractType" in df_raw.columns:
            churn_by_contract = df_raw.groupby("ContractType")["Churn"].mean().reset_index()
            churn_by_contract.columns = ["Contract", "Churn Rate"]
            fig = px.bar(
                churn_by_contract, x="Contract", y="Churn Rate",
                color="Contract",
                color_discrete_sequence=[COLOR_MAP["primary"], COLOR_MAP["secondary"], COLOR_MAP["accent"]],
            )
            fig.update_layout(title="Churn Rate by Contract Type", **PLOTLY_LAYOUT, showlegend=False,
                             yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Customers
# ---------------------------------------------------------------------------
def page_customers(df_pred: pd.DataFrame) -> None:
    """Searchable/filterable customer table with recommendations."""

    st.markdown('<div class="section-header">👥 Customer Directory</div>', unsafe_allow_html=True)

    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        search = st.text_input("🔍 Search Customer ID", "", key="cust_search")
    with col2:
        risk_filter = st.multiselect("Risk Level", ["High", "Medium", "Low"],
                                     default=["High", "Medium", "Low"], key="cust_risk")
    with col3:
        sort_by = st.selectbox("Sort By", ["ChurnProbability", "Tenure", "MonthlyCharges", "SupportTickets"],
                               key="cust_sort")
    with col4:
        sort_order = st.selectbox("Order", ["Descending", "Ascending"], key="cust_order")

    # Apply filters
    filtered = df_pred.copy()
    if search:
        filtered = filtered[filtered["CustomerID"].str.contains(search, case=False, na=False)]
    if risk_filter:
        filtered = filtered[filtered["RiskLabel"].isin(risk_filter)]

    ascending = sort_order == "Ascending"
    filtered = filtered.sort_values(sort_by, ascending=ascending)

    # Generate recommendations for display
    display_cols = [
        "CustomerID", "Tenure", "MonthlyCharges", "SatisfactionScore",
        "SupportTickets", "ChurnProbability", "RiskLabel",
    ]
    display_df = filtered[display_cols].copy()
    display_df["ChurnProbability"] = (display_df["ChurnProbability"] * 100).round(1).astype(str) + "%"
    display_df.columns = [
        "Customer ID", "Tenure", "Monthly Charges", "Satisfaction",
        "Tickets", "Churn Prob", "Risk",
    ]

    # Add recommendations column
    recs = []
    for _, row in filtered.iterrows():
        rec = get_recommendation(row["RiskLabel"], row.to_dict())
        recs.append(rec["action"])
    display_df["Recommended Action"] = recs

    # Stats bar
    st.markdown(f"""
    <div class="info-card" style="display:flex; justify-content:space-around; text-align:center;">
        <div><span style="font-size:1.5rem; font-weight:700; color:#e8eaf6;">{len(filtered):,}</span>
        <br><span style="color:#94a3b8; font-size:0.8rem;">SHOWING</span></div>
        <div><span style="font-size:1.5rem; font-weight:700; color:#ef4444;">{(filtered['RiskLabel']=='High').sum():,}</span>
        <br><span style="color:#94a3b8; font-size:0.8rem;">HIGH RISK</span></div>
        <div><span style="font-size:1.5rem; font-weight:700; color:#f59e0b;">{(filtered['RiskLabel']=='Medium').sum():,}</span>
        <br><span style="color:#94a3b8; font-size:0.8rem;">MEDIUM RISK</span></div>
        <div><span style="font-size:1.5rem; font-weight:700; color:#22c55e;">{(filtered['RiskLabel']=='Low').sum():,}</span>
        <br><span style="color:#94a3b8; font-size:0.8rem;">LOW RISK</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Pagination
    page_size = 20
    total_pages = max(1, len(display_df) // page_size + (1 if len(display_df) % page_size else 0))
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key="cust_page")
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    st.dataframe(
        display_df.iloc[start_idx:end_idx],
        use_container_width=True,
        hide_index=True,
        height=550,
    )

    st.caption(f"Page {page} of {total_pages} • {len(display_df):,} total customers")

    # CSV export
    csv = filtered.to_csv(index=False)
    st.download_button(
        "📥 Export to CSV",
        csv,
        file_name="churnguard_customers.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Page: Customer Detail
# ---------------------------------------------------------------------------
def page_customer_detail(df_pred: pd.DataFrame, model, preprocessor) -> None:
    """Detailed customer profile with risk meter."""

    st.markdown('<div class="section-header">🔎 Customer Detail View</div>', unsafe_allow_html=True)

    customer_ids = df_pred["CustomerID"].tolist()
    selected_id = st.selectbox("Select Customer", customer_ids, key="detail_customer")

    customer = df_pred[df_pred["CustomerID"] == selected_id].iloc[0]
    probability = customer["ChurnProbability"]
    risk_label = customer["RiskLabel"]
    rec = get_recommendation(risk_label, customer.to_dict())

    # Top section
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f"""
        <div class="info-card">
            <h3 style="color:#e8eaf6; margin-top:0;">👤 {selected_id}</h3>
            <p style="color:#94a3b8;">Customer Profile Overview</p>
            <hr style="border-color:rgba(99,102,241,0.2);">
            <table style="width:100%; color:#e8eaf6;">
                <tr><td style="color:#94a3b8; padding:6px 0;">Age</td><td style="text-align:right;">{customer.get('Age', 'N/A')}</td></tr>
                <tr><td style="color:#94a3b8; padding:6px 0;">Gender</td><td style="text-align:right;">{customer.get('Gender', 'N/A')}</td></tr>
                <tr><td style="color:#94a3b8; padding:6px 0;">Tenure</td><td style="text-align:right;">{customer.get('Tenure', 'N/A')} months</td></tr>
                <tr><td style="color:#94a3b8; padding:6px 0;">Contract</td><td style="text-align:right;">{customer.get('ContractType', 'N/A')}</td></tr>
                <tr><td style="color:#94a3b8; padding:6px 0;">Monthly Charges</td><td style="text-align:right;">${customer.get('MonthlyCharges', 0):.2f}</td></tr>
                <tr><td style="color:#94a3b8; padding:6px 0;">Payment Method</td><td style="text-align:right;">{customer.get('PaymentMethod', 'N/A')}</td></tr>
                <tr><td style="color:#94a3b8; padding:6px 0;">Internet Service</td><td style="text-align:right;">{customer.get('InternetService', 'N/A')}</td></tr>
                <tr><td style="color:#94a3b8; padding:6px 0;">Tech Support</td><td style="text-align:right;">{customer.get('TechSupport', 'N/A')}</td></tr>
                <tr><td style="color:#94a3b8; padding:6px 0;">Support Tickets</td><td style="text-align:right;">{customer.get('SupportTickets', 0)}</td></tr>
                <tr><td style="color:#94a3b8; padding:6px 0;">Satisfaction</td><td style="text-align:right;">{customer.get('SatisfactionScore', 'N/A')}/5</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="risk-meter-container">
            <h3 style="color:#e8eaf6; margin-top:0;">⚡ Churn Risk Assessment</h3>
            <p style="color:#94a3b8;">Risk Score</p>
        </div>
        """, unsafe_allow_html=True)
        gauge = create_gauge_chart(probability)
        st.plotly_chart(gauge, use_container_width=True)

        # Risk level badge
        badge_color = "#ef4444" if risk_label == "High" else "#f59e0b" if risk_label == "Medium" else "#22c55e"
        st.markdown(f"""
        <div style="text-align:center; margin-top:-20px;">
            <span style="background:{badge_color}; color:white; padding:8px 24px; border-radius:20px;
                         font-weight:700; font-size:1rem; letter-spacing:1px;">
                {risk_label.upper()} RISK
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Recommendation
    st.markdown(f"""
    <div class="info-card" style="border-left:4px solid {badge_color};">
        <h4 style="color:#e8eaf6; margin-top:0;">🎯 Recommended Retention Strategy</h4>
        <p style="color:#94a3b8; margin-bottom:4px;">Priority: {rec['priority']}</p>
        <h4 style="color:#e8eaf6;">{rec['action']}</h4>
        <p style="color:#c4b5fd;">{rec['details']}</p>
        <p style="color:#94a3b8; font-size:0.85rem; margin-top:12px;"><strong>Rationale:</strong> {rec['rationale']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Risk meter visualization
    st.markdown(f"""
    <div class="info-card" style="text-align:center;">
        <h4 style="color:#e8eaf6; margin-top:0;">📊 Risk Scale</h4>
        <div style="display:flex; align-items:center; justify-content:center; gap:0; margin:16px 0;">
            <div style="background:linear-gradient(90deg,#22c55e,#22c55e); height:12px; width:30%; border-radius:6px 0 0 6px;"></div>
            <div style="background:linear-gradient(90deg,#f59e0b,#f59e0b); height:12px; width:30%;"></div>
            <div style="background:linear-gradient(90deg,#ef4444,#ef4444); height:12px; width:30%; border-radius:0 6px 6px 0;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; color:#94a3b8; font-size:0.8rem; padding:0 5%;">
            <span>LOW</span><span>MEDIUM</span><span>HIGH</span>
        </div>
        <div style="margin-top:8px;">
            <span style="color:{badge_color}; font-weight:700;">▲ {probability:.1%}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page: Predictions (What-If)
# ---------------------------------------------------------------------------
def page_predictions(model, preprocessor) -> None:
    """Interactive what-if prediction tool."""

    st.markdown('<div class="section-header">🔮 What-If Prediction Tool</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <p style="color:#c4b5fd; margin:0;">Enter customer attributes below to get a real-time churn prediction
        from the trained Random Forest model.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 75, 35, key="wif_age")
        tenure = st.slider("Tenure (months)", 1, 72, 12, key="wif_tenure")
        monthly_charges = st.slider("Monthly Charges ($)", 20.0, 120.0, 70.0, step=5.0, key="wif_charges")
        support_tickets = st.slider("Support Tickets", 0, 10, 2, key="wif_tickets")

    with col2:
        satisfaction = st.slider("Satisfaction Score", 1.0, 5.0, 3.0, step=0.5, key="wif_sat")
        usage_freq = st.slider("Usage Frequency (days/month)", 1, 30, 15, key="wif_usage")
        last_login = st.slider("Days Since Last Login", 0, 90, 10, key="wif_login")
        gender = st.selectbox("Gender", ["Male", "Female"], key="wif_gender")

    with col3:
        contract = st.selectbox("Contract Type",
                               ["Month-to-Month", "One Year", "Two Year"], key="wif_contract")
        payment = st.selectbox("Payment Method",
                              ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"],
                              key="wif_payment")
        internet = st.selectbox("Internet Service",
                               ["Fiber Optic", "DSL", "No"], key="wif_internet")
        tech_support = st.selectbox("Tech Support", ["Yes", "No"], key="wif_tech")
        online_security = st.selectbox("Online Security", ["Yes", "No"], key="wif_security")

    if st.button("🚀 Predict Churn", key="wif_predict", use_container_width=True):
        customer_data = {
            "Age": age,
            "Gender": gender,
            "Tenure": tenure,
            "ContractType": contract,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": monthly_charges * tenure,
            "PaymentMethod": payment,
            "InternetService": internet,
            "TechSupport": tech_support,
            "OnlineSecurity": online_security,
            "DeviceProtection": "No",
            "StreamingServices": "No",
            "SupportTickets": support_tickets,
            "SatisfactionScore": satisfaction,
            "UsageFrequency": usage_freq,
            "LastLoginDays": last_login,
        }

        result = predict_single(customer_data, model, preprocessor)
        probability = result["churn_probability"]
        risk_label = result["risk_label"]
        rec = get_recommendation(risk_label, customer_data)

        st.markdown("<br>", unsafe_allow_html=True)

        # Results
        r1, r2 = st.columns([1, 1])

        with r1:
            gauge = create_gauge_chart(probability)
            st.plotly_chart(gauge, use_container_width=True)

            badge_color = "#ef4444" if risk_label == "High" else "#f59e0b" if risk_label == "Medium" else "#22c55e"
            st.markdown(f"""
            <div style="text-align:center; margin-top:-20px;">
                <span style="background:{badge_color}; color:white; padding:8px 24px; border-radius:20px;
                             font-weight:700; font-size:1.1rem; letter-spacing:1px;">
                    {risk_label.upper()} RISK
                </span>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class="info-card" style="border-left:4px solid {badge_color};">
                <h4 style="color:#e8eaf6; margin-top:0;">🎯 Prediction Result</h4>
                <table style="width:100%; color:#e8eaf6;">
                    <tr><td style="color:#94a3b8; padding:8px 0;">Churn Probability</td>
                        <td style="text-align:right; font-weight:700; font-size:1.2rem; color:{badge_color};">{probability:.1%}</td></tr>
                    <tr><td style="color:#94a3b8; padding:8px 0;">Risk Level</td>
                        <td style="text-align:right; font-weight:700;">{risk_label}</td></tr>
                    <tr><td style="color:#94a3b8; padding:8px 0;">Priority</td>
                        <td style="text-align:right;">{rec['priority']}</td></tr>
                </table>
                <hr style="border-color:rgba(99,102,241,0.2);">
                <h5 style="color:#c4b5fd;">Recommended Action</h5>
                <p style="color:#e8eaf6;">{rec['action']}</p>
                <p style="color:#94a3b8; font-size:0.85rem;">{rec['details']}</p>
            </div>
            """, unsafe_allow_html=True)

        # Risk factors
        if result.get("top_risk_factors"):
            st.markdown('<div class="section-header">⚡ Top Risk Factors</div>', unsafe_allow_html=True)
            factors = result["top_risk_factors"]
            factor_df = pd.DataFrame(factors)
            fig = px.bar(
                factor_df, x="importance", y="feature", orientation="h",
                color="importance", color_continuous_scale="RdPu",
                labels={"importance": "Contribution", "feature": "Feature"},
            )
            fig.update_layout(title="Contributing Factors", **PLOTLY_LAYOUT,
                             showlegend=False, coloraxis_showscale=False, height=300)
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Analytics
# ---------------------------------------------------------------------------
def page_analytics(df_raw: pd.DataFrame, df_pred: pd.DataFrame) -> None:
    """Analytics page with EDA charts and business insights."""

    st.markdown('<div class="section-header">📈 Analytics & Insights</div>', unsafe_allow_html=True)

    # Business insights
    insights = generate_insights(df_raw)
    if insights:
        st.markdown('<h4 style="color:#e8eaf6;">💡 Data-Driven Insights</h4>', unsafe_allow_html=True)
        for insight in insights:
            st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    tab1, tab2, tab3, tab4 = st.tabs(["📊 By Feature", "📉 Distributions", "🔗 Correlations", "📋 Segments"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Churn by payment method
            churn_payment = df_raw.groupby("PaymentMethod")["Churn"].mean().reset_index()
            fig = px.bar(churn_payment, x="PaymentMethod", y="Churn",
                        color="PaymentMethod",
                        color_discrete_sequence=[COLOR_MAP["primary"], COLOR_MAP["secondary"],
                                                  COLOR_MAP["accent"], COLOR_MAP["blue_light"]])
            fig.update_layout(title="Churn Rate by Payment Method", **PLOTLY_LAYOUT,
                             yaxis_tickformat=".0%", showlegend=False,
                             yaxis_title="Churn Rate", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Churn by internet service
            churn_internet = df_raw.groupby("InternetService")["Churn"].mean().reset_index()
            fig = px.bar(churn_internet, x="InternetService", y="Churn",
                        color="InternetService",
                        color_discrete_map={"Fiber Optic": COLOR_MAP["red"],
                                           "DSL": COLOR_MAP["orange"],
                                           "No": COLOR_MAP["green"]})
            fig.update_layout(title="Churn Rate by Internet Service", **PLOTLY_LAYOUT,
                             yaxis_tickformat=".0%", showlegend=False,
                             yaxis_title="Churn Rate", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            # Churn by support tickets
            churn_tickets = df_raw.groupby("SupportTickets")["Churn"].mean().reset_index()
            fig = px.line(churn_tickets, x="SupportTickets", y="Churn",
                         markers=True, color_discrete_sequence=[COLOR_MAP["secondary"]])
            fig.update_layout(title="Churn Rate by Support Tickets", **PLOTLY_LAYOUT,
                             yaxis_tickformat=".0%", yaxis_title="Churn Rate",
                             xaxis_title="Number of Support Tickets")
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            # Churn by tech support
            churn_tech = df_raw.groupby("TechSupport")["Churn"].mean().reset_index()
            fig = px.bar(churn_tech, x="TechSupport", y="Churn",
                        color="TechSupport",
                        color_discrete_map={"Yes": COLOR_MAP["green"], "No": COLOR_MAP["red"]})
            fig.update_layout(title="Churn Rate by Tech Support", **PLOTLY_LAYOUT,
                             yaxis_tickformat=".0%", showlegend=False,
                             yaxis_title="Churn Rate", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(df_raw, x="Tenure", color="Churn", nbins=30,
                              barmode="overlay", color_discrete_map={0: COLOR_MAP["green"], 1: COLOR_MAP["red"]},
                              labels={"Churn": "Status"})
            fig.update_layout(title="Tenure Distribution by Churn", **PLOTLY_LAYOUT)
            fig.for_each_trace(lambda t: t.update(name="Retained" if t.name == "0" else "Churned"))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(df_raw, x="Churn", y="MonthlyCharges",
                        color="Churn", color_discrete_map={0: COLOR_MAP["green"], 1: COLOR_MAP["red"]})
            fig.update_layout(title="Monthly Charges by Churn", **PLOTLY_LAYOUT, showlegend=False)
            fig.update_xaxes(ticktext=["Retained", "Churned"], tickvals=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            fig = px.histogram(df_raw, x="SatisfactionScore", color="Churn", nbins=20,
                              barmode="overlay", color_discrete_map={0: COLOR_MAP["green"], 1: COLOR_MAP["red"]})
            fig.update_layout(title="Satisfaction Score Distribution", **PLOTLY_LAYOUT)
            fig.for_each_trace(lambda t: t.update(name="Retained" if t.name == "0" else "Churned"))
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            fig = px.box(df_raw, x="Churn", y="TotalCharges",
                        color="Churn", color_discrete_map={0: COLOR_MAP["green"], 1: COLOR_MAP["red"]})
            fig.update_layout(title="Total Charges by Churn", **PLOTLY_LAYOUT, showlegend=False)
            fig.update_xaxes(ticktext=["Retained", "Churned"], tickvals=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        corr = df_raw[numeric_cols].corr()
        fig = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale="RdPu",
            aspect="auto",
        )
        fig.update_layout(title="Feature Correlation Heatmap", **PLOTLY_LAYOUT, height=600)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(df_raw, x="Age", color="Churn", nbins=25,
                              barmode="overlay", color_discrete_map={0: COLOR_MAP["green"], 1: COLOR_MAP["red"]})
            fig.update_layout(title="Age Distribution by Churn", **PLOTLY_LAYOUT)
            fig.for_each_trace(lambda t: t.update(name="Retained" if t.name == "0" else "Churned"))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Gender churn rate
            churn_gender = df_raw.groupby("Gender")["Churn"].mean().reset_index()
            fig = px.bar(churn_gender, x="Gender", y="Churn",
                        color="Gender", color_discrete_sequence=[COLOR_MAP["primary"], COLOR_MAP["accent"]])
            fig.update_layout(title="Churn Rate by Gender", **PLOTLY_LAYOUT,
                             yaxis_tickformat=".0%", showlegend=False, yaxis_title="Churn Rate")
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Model Performance
# ---------------------------------------------------------------------------
def page_model_performance(model, preprocessor, df_raw) -> None:
    """Model performance metrics and visualizations."""

    st.markdown('<div class="section-header">🤖 Model Performance</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4 style="color:#e8eaf6; margin-top:0;">Random Forest Classifier</h4>
        <p style="color:#94a3b8;">Trained with hyperparameter tuning via RandomizedSearchCV.
        Optimized for F1-score to balance precision and recall for churn detection.
        Class imbalance handled with balanced class weights.</p>
    </div>
    """, unsafe_allow_html=True)

    metrics, cm, fpr, tpr, importance_df = get_model_metrics(model, preprocessor, df_raw)

    # Metrics cards
    cols = st.columns(5)
    metric_items = [
        ("🎯", "Accuracy", metrics["accuracy"]),
        ("📊", "Precision", metrics["precision"]),
        ("🔍", "Recall", metrics["recall"]),
        ("⚖️", "F1 Score", metrics["f1_score"]),
        ("📈", "ROC-AUC", metrics["roc_auc"]),
    ]
    for col, (icon, name, value) in zip(cols, metric_items):
        with col:
            st.markdown(render_kpi_card(icon, name, f"{value:.4f}"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        # Confusion Matrix
        labels = ["Retained", "Churned"]
        fig = px.imshow(
            cm, text_auto=True,
            x=labels, y=labels,
            color_continuous_scale="RdPu",
            labels=dict(x="Predicted", y="Actual"),
        )
        fig.update_layout(title="Confusion Matrix", **PLOTLY_LAYOUT, height=400)
        fig.update_traces(textfont=dict(size=18))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # ROC Curve
        roc_auc = metrics["roc_auc"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"ROC Curve (AUC = {roc_auc:.4f})",
            line=dict(color=COLOR_MAP["secondary"], width=2.5),
            fill="tozeroy", fillcolor="rgba(139, 92, 246, 0.1)",
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random Baseline",
            line=dict(color="#4a4f6a", width=1, dash="dash"),
        ))
        fig.update_layout(
            title="ROC Curve", **PLOTLY_LAYOUT, height=400,
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature Importance
    st.markdown('<div class="section-header">🏆 Feature Importance</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-card">
        Feature importance highlights which customer attributes have the strongest influence on churn predictions.
        These values are extracted directly from the trained Random Forest model.
    </div>
    """, unsafe_allow_html=True)

    fig = px.bar(
        importance_df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="RdPu",
    )
    fig.update_layout(
        title="Top 15 Feature Importances — Random Forest", **PLOTLY_LAYOUT,
        height=500, showlegend=False, coloraxis_showscale=False,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
def main():
    """Run the ChurnGuard AI dashboard."""

    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="brand-header">
            <div class="brand-title">🛡️ ChurnGuard AI</div>
            <div class="brand-subtitle">Predict · Prevent · Retain</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "👥 Customers", "🔮 Predictions", "📈 Analytics", "🤖 Model Performance"],
            key="nav",
        )

        st.markdown("---")

        # Risk threshold configuration
        st.markdown("### ⚙️ Settings")
        high_threshold = st.slider("High Risk Threshold", 0.5, 0.9, 0.7, 0.05, key="high_thresh")
        medium_threshold = st.slider("Medium Risk Threshold", 0.2, 0.6, 0.4, 0.05, key="med_thresh")

        st.markdown("---")
        st.markdown("""
        <div style="text-align:center; color:#64748b; font-size:0.75rem;">
            <p>ChurnGuard AI v1.0</p>
            <p>Powered by Random Forest</p>
            <p>© 2024 ChurnGuard</p>
        </div>
        """, unsafe_allow_html=True)

    # Load data and model
    df_raw = load_data()
    model, preprocessor = load_model()

    # Update risk thresholds
    from prediction import RISK_THRESHOLDS
    RISK_THRESHOLDS["high"] = high_threshold
    RISK_THRESHOLDS["medium"] = medium_threshold

    # Generate predictions
    df_pred = get_predictions(model, preprocessor, df_raw)

    # Header
    st.markdown("""
    <div class="brand-header" style="margin-bottom:20px;">
        <div class="brand-title" style="font-size:2.5rem;">🛡️ ChurnGuard AI</div>
        <div class="brand-subtitle" style="font-size:1rem;">Customer Retention Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    # Route to page
    if page == "🏠 Dashboard":
        page_dashboard(df_pred, df_raw)
    elif page == "👥 Customers":
        page_customers(df_pred)
    elif page == "🔮 Predictions":
        page_predictions(model, preprocessor)
    elif page == "📈 Analytics":
        page_analytics(df_raw, df_pred)
    elif page == "🤖 Model Performance":
        page_model_performance(model, preprocessor, df_raw)


if __name__ == "__main__":
    main()
