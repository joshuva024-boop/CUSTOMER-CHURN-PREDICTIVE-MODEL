# 🛡️ ChurnGuard AI — Customer Churn Predictive Model

> **"Predict churn before customers leave."**

A complete, end-to-end customer churn prediction platform powered by Machine Learning. ChurnGuard AI identifies at-risk customers, explains the factors driving churn, and recommends targeted retention strategies — all through a professional, interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Business Objective](#-business-objective)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [ML Workflow](#-ml-workflow)
- [Dataset](#-dataset)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Model Architecture](#-model-architecture)
- [Evaluation Metrics](#-evaluation-metrics)
- [Dashboard](#-dashboard)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [Sample Prediction](#-sample-prediction)
- [Business Use Cases](#-business-use-cases)
- [Future Improvements](#-future-improvements)

---

## 🎯 Problem Statement

Customer churn is one of the biggest challenges for subscription-based businesses. Losing customers is expensive — acquiring a new customer costs **5–25x more** than retaining an existing one. Businesses need to:

1. **Identify** which customers are likely to leave
2. **Understand** why they're at risk
3. **Act** with targeted retention strategies before it's too late

---

## 💼 Business Objective

Build an AI-powered customer retention platform that:

- Predicts churn probability for every customer using a Random Forest model
- Classifies customers into **High / Medium / Low** risk categories
- Explains the key factors contributing to each customer's churn risk
- Recommends personalized retention actions based on customer profiles
- Provides an interactive dashboard for business stakeholders

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **ML Pipeline** | End-to-end: data cleaning → EDA → feature engineering → model training → evaluation |
| 🌲 **Random Forest Classifier** | Tuned via RandomizedSearchCV with class imbalance handling |
| 📊 **10+ EDA Visualizations** | Professional charts analyzing churn patterns across all features |
| 🔴🟠🟢 **Risk Classification** | Configurable probability-based risk levels (High/Medium/Low) |
| 🎯 **Retention Engine** | Rule-based recommendations connected to customer attributes |
| 💡 **Business Insights** | Data-driven insights auto-generated from actual patterns |
| 🔮 **What-If Predictions** | Interactive tool to predict churn for hypothetical customers |
| 📱 **SaaS Dashboard** | Modern dark-themed Streamlit dashboard with interactive Plotly charts |
| 📥 **CSV Export** | Export filtered customer data with predictions |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.9+** | Core programming language |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computing |
| **Scikit-learn** | ML pipeline, preprocessing, model training, evaluation |
| **Random Forest** | Classification algorithm |
| **Matplotlib** | Static visualizations |
| **Seaborn** | Statistical visualization |
| **Plotly** | Interactive dashboard charts |
| **Streamlit** | Web dashboard framework |
| **Joblib** | Model serialization |

---

## 🔄 ML Workflow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Synthetic   │────▶│    Data      │────▶│     EDA     │
│  Dataset     │     │ Preprocessing│     │ Visualize   │
│  (5,000+)    │     │  Pipeline    │     │  Patterns   │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Retention   │◀────│    Risk      │◀────│   Random    │
│ Recommen-    │     │ Classification│    │   Forest    │
│  dations     │     │  Engine      │     │  Classifier │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  ChurnGuard  │
                    │  Dashboard   │
                    └──────────────┘
```

---

## 📊 Dataset

A **synthetic dataset** of 5,000+ customer records is generated with realistic distributions and correlations. Churn labels are driven by a logistic probability model (not random), ensuring that features have meaningful predictive power.

### Features

| Feature | Type | Description |
|---------|------|-------------|
| CustomerID | ID | Unique customer identifier |
| Age | Numeric | Customer age (18–75) |
| Gender | Categorical | Male / Female |
| Tenure | Numeric | Months as customer (1–72) |
| ContractType | Categorical | Month-to-Month / One Year / Two Year |
| MonthlyCharges | Numeric | Monthly billing amount ($20–$120) |
| TotalCharges | Numeric | Cumulative charges |
| PaymentMethod | Categorical | Electronic Check / Mailed Check / Bank Transfer / Credit Card |
| InternetService | Categorical | Fiber Optic / DSL / No |
| TechSupport | Categorical | Yes / No |
| OnlineSecurity | Categorical | Yes / No |
| DeviceProtection | Categorical | Yes / No |
| StreamingServices | Categorical | Yes / No |
| SupportTickets | Numeric | Number of support interactions (0–10) |
| SatisfactionScore | Numeric | Customer satisfaction (1.0–5.0) |
| UsageFrequency | Numeric | Days used per month (1–30) |
| LastLoginDays | Numeric | Days since last login (0–90) |
| **Churn** | **Target** | **0 = Retained, 1 = Churned** |

---

## 📈 Exploratory Data Analysis

The EDA module generates 10 publication-quality visualizations:

1. **Churn Distribution** — Overall churn vs retention split
2. **Churn by Contract Type** — Month-to-month shows highest churn
3. **Churn by Tenure** — Shorter tenure correlates with higher churn
4. **Churn by Monthly Charges** — Higher charges increase churn risk
5. **Churn by Payment Method** — Electronic check users churn more
6. **Churn by Support Tickets** — More tickets = higher churn
7. **Churn by Satisfaction Score** — Low satisfaction drives churn
8. **Churn by Internet Service** — Fiber optic has elevated churn
9. **Correlation Heatmap** — Feature relationships
10. **Customer Segments** — Multi-dimensional overview

---

## 🤖 Model Architecture

### Random Forest Classifier

- **Algorithm**: Random Forest (ensemble of decision trees)
- **Tuning**: RandomizedSearchCV (50 iterations, 5-fold stratified CV)
- **Optimization**: F1-score (balancing precision and recall for churn detection)
- **Imbalance**: Handled via `class_weight='balanced'`
- **Reproducibility**: `random_state=42`

### Hyperparameter Search Space

| Parameter | Values |
|-----------|--------|
| n_estimators | 100, 200, 300, 400, 500 |
| max_depth | 5, 10, 15, 20, 25, None |
| min_samples_split | 2, 5, 10, 15 |
| min_samples_leaf | 1, 2, 4, 8 |
| max_features | sqrt, log2, None |
| class_weight | balanced, balanced_subsample, None |
| criterion | gini, entropy |

---

## 📊 Evaluation Metrics

All metrics are computed from the actual trained model — **nothing is fabricated**.

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions |
| **Precision** | Of predicted churners, how many actually churned |
| **Recall** | Of actual churners, how many were detected |
| **F1 Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under the ROC curve |

> 📌 **Note**: We prioritize **Recall** and **F1-Score** over raw accuracy because missing a customer who is about to churn (false negative) is more costly than a false alarm.

---

## 🖥️ Dashboard

The **ChurnGuard AI** dashboard is a modern, SaaS-style analytics platform built with Streamlit.

### Pages

| Page | Description |
|------|-------------|
| **Dashboard** | KPI cards, churn overview, risk breakdown |
| **Customers** | Searchable/filterable/sortable customer table with CSV export |
| **Predictions** | What-if prediction tool with interactive form |
| **Analytics** | EDA charts, correlation analysis, business insights |
| **Model Performance** | Metrics, confusion matrix, ROC curve, feature importance |

### Design

- Dark navy / deep purple gradient background
- White cards with subtle shadows
- Purple/blue accent colors
- Red/orange/green risk indicators
- Interactive Plotly charts
- Responsive layout
- Configurable risk thresholds

---

## ⚡ Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/customer-churn-predictive-model.git
cd customer-churn-predictive-model

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1: Generate the Dataset

```bash
python src/generate_dataset.py
```

### Step 2: Run Preprocessing

```bash
python src/data_preprocessing.py
```

### Step 3: Generate EDA Visualizations

```bash
python src/eda.py
```

### Step 4: Train the Model

```bash
python src/train_model.py
```

### Step 5: Evaluate the Model

```bash
python src/evaluate_model.py
```

### Step 6: Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

### One-Liner (Run Everything)

```bash
python src/generate_dataset.py && python src/data_preprocessing.py && python src/eda.py && python src/train_model.py && python src/evaluate_model.py && streamlit run dashboard/app.py
```

---

## 📁 Project Structure

```
customer-churn-predictive-model/
│
├── data/
│   ├── raw/                        # Raw generated dataset
│   │   └── customer_churn_data.csv
│   └── processed/                  # Cleaned dataset
│       └── cleaned_data.csv
│
├── models/
│   ├── random_forest_model.pkl     # Trained model
│   └── preprocessor.pkl            # Fitted preprocessor
│
├── src/
│   ├── generate_dataset.py         # Synthetic data generator
│   ├── data_preprocessing.py       # Cleaning & preprocessing pipeline
│   ├── eda.py                      # Exploratory data analysis
│   ├── train_model.py              # Model training with tuning
│   ├── evaluate_model.py           # Metrics & visualization
│   ├── prediction.py               # Prediction engine
│   └── recommendations.py          # Retention recommendation engine
│
├── dashboard/
│   └── app.py                      # Streamlit dashboard
│
├── visualizations/                 # Generated EDA & model charts
│
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

---

## 🔮 Sample Prediction

**Input:**
```
Age: 35 | Tenure: 3 months | Contract: Month-to-Month
Monthly Charges: $95 | Payment: Electronic Check
Internet: Fiber Optic | Tech Support: No
Support Tickets: 5 | Satisfaction: 2.0/5
```

**Output:**
```
🔴 CHURN PROBABILITY: 78%
🔴 RISK: HIGH

🎯 RECOMMENDED ACTION:
   Escalate to priority support team.
   Customer has 5 support tickets indicating persistent
   unresolved issues. Assign dedicated support agent.
```

---

## 💼 Business Use Cases

1. **Telecom**: Predict subscriber churn and offer targeted plans
2. **SaaS**: Identify at-risk accounts for customer success outreach
3. **Banking**: Detect customers likely to close accounts
4. **E-commerce**: Predict customer disengagement and re-engage
5. **Insurance**: Identify policy holders at risk of non-renewal
6. **Subscription Services**: Optimize retention for streaming/media platforms

---

## 🚀 Future Improvements

- [ ] **Additional Models**: XGBoost, LightGBM, Neural Network comparison
- [ ] **SHAP Explanations**: Per-prediction SHAP values for better interpretability
- [ ] **Real-Time API**: FastAPI endpoint for live predictions
- [ ] **A/B Testing**: Track which retention strategies are most effective
- [ ] **Time Series**: Incorporate temporal churn patterns
- [ ] **Email Alerts**: Automated alerts when customers enter high-risk status
- [ ] **Database Integration**: PostgreSQL/MongoDB backend instead of CSV
- [ ] **Authentication**: User login and role-based access control
- [ ] **Batch Upload**: Upload CSV files for bulk prediction
- [ ] **Model Retraining**: Scheduled retraining pipeline with new data

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

Built as a demonstration of end-to-end Machine Learning engineering — from data generation to production-ready dashboard.

---

> *"Don't wait for customers to leave. Predict churn before it happens."*
