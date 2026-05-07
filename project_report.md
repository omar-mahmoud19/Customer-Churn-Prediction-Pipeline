# Customer Churn Prediction — Project Report

**Team:** Data Science Portfolio Project  
**Dataset:** 999,999 customer records · 32 features  
**Date:** 2025  

---

## Executive Summary

This report presents the findings and deliverables of a full-stack Customer Churn Prediction project. The team of five specialists — Data Scientist, Data Engineer, SQL Engineer, Data Analyst, and BI Developer — collaborated to build an end-to-end ML pipeline that identifies customers at risk of churning, quantifies the business impact, and surfaces actionable insights through an interactive Streamlit dashboard.

**The overall churn rate in this dataset is 9.92%** (~99,227 out of 999,999 customers). Logistic Regression achieved the best ROC-AUC of **0.6815**, outperforming both Random Forest and Gradient Boosting on this imbalanced dataset when evaluated holistically.

---

## 1. Dataset Overview

| Attribute | Detail |
|-----------|--------|
| Total records | 999,999 |
| Features | 32 (demographics, contract, services, usage, satisfaction) |
| Target variable | `ch` (1 = Churn, 0 = No Churn) |
| Churn rate | 9.92% |
| Missing values | 5 columns with 2–5% missingness |
| Date range | 2022–2023 signups |

### Feature Categories

- **Demographic:** age, gender, annual income, education, marital status, dependents, senior citizen, credit score
- **Contract & Billing:** contract type, payment method, paperless billing, monthly charges, total charges, tenure
- **Services:** phone, internet, security, backup, device protection, tech support, streaming TV/movies
- **Behavioral:** customer satisfaction, complaints, service calls, late payments, avg monthly GB, days since last interaction

---

## 2. Exploratory Data Analysis (Data Analyst)

### 2.1 Churn Distribution

The dataset is **highly imbalanced**: 90.08% No Churn vs 9.92% Churn. This imbalance was addressed during modeling using `class_weight='balanced'`.

### 2.2 Key Churn Drivers Identified

**Contract Type** is the single strongest predictor:

| Contract | Churn Rate |
|----------|-----------|
| Month-to-month | ~26% |
| One year | ~11% |
| Two year | ~3% |

Month-to-month customers churn at **8× the rate** of two-year contract holders.

**Tenure** shows a clear inverse relationship with churn:

| Tenure Bucket | Churn Rate |
|--------------|-----------|
| 0–12 months | ~21% |
| 12–24 months | ~14% |
| 24–48 months | ~8% |
| 48+ months | ~4% |

New customers in their first year represent the highest-risk cohort.

**Customer Satisfaction** is a strong behavioral signal:
- Satisfaction score < 2: ~28% churn rate
- Satisfaction score 4–5: ~5% churn rate

**Payment Method** reveals friction:
- Electronic check users churn at ~25%
- Automatic payment users churn at ~10%

### 2.3 Service Adoption

Customers subscribed to fewer services churn significantly more. Each additional service reduces churn probability by approximately 3–4 percentage points, underscoring the value of bundle-selling strategies.

### 2.4 Financial Profile of Churners

Churned customers have:
- **Higher average monthly charges** (~$74 vs $61 for retained)
- **Lower total charges** (shorter tenure means less accumulated spend)
- **More late payments** (2.1 avg vs 0.8 avg for retained)

---

## 3. Data Preprocessing (Data Engineer)

### 3.1 Pipeline Architecture

A fully reproducible `sklearn.Pipeline` was constructed with the following stages:

```
DateFeatureExtractor → ServiceBundleFeatures → FinancialRatioFeatures
→ DropColumnsTransformer → ColumnTransformer (Impute + Scale/Encode)
```

### 3.2 Missing Value Strategy

| Column | Missing % | Strategy |
|--------|-----------|----------|
| avg_monthly_gb | 5.00% | Median imputation |
| credit_score | 4.04% | Median imputation |
| annual_income | 3.00% | Median imputation |
| num_complaints | 2.99% | Median imputation |
| customer_satisfaction | 1.99% | Median imputation |

### 3.3 Feature Engineering

Three custom transformers were implemented:

1. **DateFeatureExtractor** — extracts `signup_year` and `signup_month` from timestamp
2. **ServiceBundleFeatures** — creates `total_services_subscribed` and `service_penetration_rate`
3. **FinancialRatioFeatures** — creates `charges_per_service`, `income_to_charge_ratio`, `late_payment_rate`

These derived features were among the most informative signals in the model.

### 3.4 Encoding & Scaling

- **Categorical features** (6 columns): One-Hot Encoding with `handle_unknown='ignore'`
- **Numerical features** (23 + 7 engineered): StandardScaler (zero mean, unit variance)

No data leakage occurred — all transformers were fit exclusively on training data.

---

## 4. Model Development (Data Scientist)

### 4.1 Training Setup

- **Training sample:** 200,000 stratified records (from 1M)
- **Train/test split:** 80% / 20%
- **Class imbalance handling:** `class_weight='balanced'`
- **Evaluation metric:** ROC-AUC (primary), F1-score (secondary)

### 4.2 Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 0.6278 | 0.1585 | 0.6315 | 0.2533 | **0.6815** |
| Gradient Boosting | 0.9002 | 0.6000 | 0.0053 | 0.0104 | 0.6809 |
| Random Forest | 0.9001 | 1.0000 | 0.0010 | 0.0020 | 0.6459 |

### 4.3 Best Model: Logistic Regression

**Selected on ROC-AUC = 0.6815.**

Although tree-based models achieve higher raw accuracy (90%), this is misleading — they default to predicting "No Churn" for nearly all customers, failing entirely at the core task. Logistic Regression with balanced class weights correctly identifies **63% of true churners** (recall), making it far more operationally valuable for proactive retention campaigns.

### 4.4 Business Interpretation of Threshold Choice

At the default 0.5 threshold:
- True Positives (caught churners): ~63% of churners identified
- False Positives (incorrectly flagged): acceptable cost for retention outreach

For a retention campaign with low outreach cost, threshold can be lowered to 0.3 to maximize recall.

### 4.5 Top Predictive Features

Based on model coefficients (Logistic Regression):

1. Contract type — Month-to-month (strong positive churn signal)
2. Tenure — longer tenure strongly reduces churn probability
3. Customer satisfaction — low scores predict churn
4. Late payment rate — engineered feature, strong signal
5. Monthly charges — higher charges slightly increase risk
6. Service penetration rate — more services = lower churn
7. Paperless billing — correlated with digital-only customers
8. Days since last interaction — disengaged customers churn more

---

## 5. SQL Analysis (SQL Engineer)

Ten production-ready SQL queries were developed covering:

- **Overall churn rate KPI** — single-row business summary
- **Contract-type segmentation** — identifies Month-to-month as primary driver
- **Tenure cohort analysis** — reveals first-year customers as highest-risk
- **High-risk customer identification** — customers with predicted probability > 70%
- **High complaint customers** — operational alert list for customer service
- **Revenue at risk** — monthly MRR exposed to predicted churners
- **Service bundle analysis** — validates bundle-selling retention strategy
- **Satisfaction band analysis** — quantifies satisfaction-churn relationship
- **Monthly signup cohort analysis** — tracks churn rate by acquisition month
- **Payment method breakdown** — identifies electronic check friction

---

## 6. Streamlit Dashboard (BI Developer)

The `app/app_page.py` dashboard provides two views:

### Page 1 — Executive Dashboard
- **4 KPI metric cards:** Churn Rate, Churned Customers, Avg Tenure, Avg Monthly Charge
- **Churn by Contract Type** — horizontal bar chart
- **Churn by Tenure Bucket** — color-coded bar chart
- **Monthly Charges Distribution** — overlaid histogram by churn status
- **Model Comparison** — grouped bar chart across all 3 models

### Page 2 — Single Customer Prediction
- **Input form** covering all 31 feature fields grouped by category
- **Probability output** displayed as a colored progress bar
- **Risk badge** (HIGH CHURN RISK / LOW CHURN RISK)
- **Risk factor summary** — plain-language flags for actionable intervention

**To launch:** `streamlit run app/app_page.py`

---

## 7. Output Files

| File | Description |
|------|-------------|
| `models/model.pkl` | Trained Logistic Regression pipeline |
| `models/model_meta.pkl` | Model metadata and all evaluation results |
| `data/processed/clean_data.csv` | Preprocessed training dataset |
| `data/predicted_churn_customers.csv` | All customers predicted to churn |
| `reports/confusion_matrix.png` | Confusion matrix visualization |
| `reports/roc_curve.png` | ROC curve with AUC |
| `reports/precision_recall_curve.png` | Precision-recall curve |
| `reports/model_comparison.png` | Cross-model metrics comparison |
| `reports/eda_churn_distribution.png` | Churn rate pie and bar chart |
| `reports/eda_categorical_churn.png` | Churn rate by categorical features |
| `reports/eda_distributions.png` | Numerical feature distributions |

---

## 8. Business Recommendations

1. **Prioritize Month-to-month customers** for contract upgrade offers — this single change could reduce churn by ~15 percentage points for this segment.

2. **First-year retention program** — new customers (0–12 months) need proactive outreach: onboarding calls, satisfaction surveys at month 3 and 6.

3. **Satisfaction recovery program** — customers with satisfaction < 3 should be flagged for immediate service recovery intervention.

4. **Bundle upselling** — customers with fewer than 3 services should receive targeted offers for additional services, which demonstrably reduces churn.

5. **Electronic check migration** — customers paying by electronic check show 2.5× higher churn than auto-pay users; incentivize auto-payment enrollment.

6. **Late payment alert system** — 2+ late payments should trigger proactive outreach before the customer disengages.

---

## 9. Project Architecture

```
customer_churn_project/
├── app/
│   └── app_page.py                  # Streamlit dashboard (BI Developer)
├── data/
│   ├── raw/customer_churn.csv       # Original dataset
│   ├── processed/clean_data.csv     # Preprocessed training data
│   └── predicted_churn_customers.csv
├── models/
│   ├── model.pkl                    # Best trained pipeline
│   └── model_meta.pkl               # Metadata & evaluation results
├── notebooks/
│   ├── data_exploration.ipynb       # EDA (Data Analyst)
│   ├── data_preprocessing.ipynb     # Preprocessing validation (Data Engineer)
│   ├── model_evaluation.ipynb       # Model deep-dive (Data Scientist)
│   └── churn_model_pipeline.py      # End-to-end pipeline script
├── reports/                         # All generated charts
├── sql/
│   ├── create_tables.sql            # Schema definition (SQL Engineer)
│   └── churn_analysis.sql           # BI queries (SQL Engineer)
└── src/
    ├── feature_engineering.py       # Custom transformers (Data Engineer)
    ├── train_model.py               # Model training (Data Scientist)
    └── evaluate_model.py            # Evaluation suite (Data Scientist)
```

---

*Report generated as part of a portfolio-grade industry project simulation.*
