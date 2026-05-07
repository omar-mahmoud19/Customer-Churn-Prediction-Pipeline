# Customer Churn Analysis and Retention Strategy

## Project Overview

This project analyzes customer behavior to predict customer churn using machine learning techniques.
Customer churn refers to customers who stop using a company's service. Predicting churn helps businesses identify high‑risk customers and take actions to retain them.

The system analyzes demographic data, financial behavior, service usage, and customer satisfaction to predict the probability that a customer will leave.

---

## Objectives

- Understand customer behavior patterns
- Identify key factors that lead to churn
- Build a machine learning model to predict churn
- Provide business insights to improve customer retention

---

## Dataset Description

### Target Variable

`churn`

```
1 = Customer left
0 = Customer stayed
```

### Feature Categories

#### Customer Information

- customer_id
- age
- gender
- education
- marital_status
- dependents
- annual_income

#### Financial Features

- monthlycharges
- totalcharges
- payment_method
- paperless_billing
- late_payments

#### Service Usage

- has_phone_service
- has_internet_service
- has_online_security
- has_online_backup
- has_device_protection
- has_tech_support
- has_streaming_tv
- has_streaming_movies

#### Customer Behavior

- customer_satisfaction
- num_complaints
- num_service_calls
- days_since_last_interaction
- avg_monthly_gb

#### Risk Indicators

- credit_score
- tenure
- contract

---

## Project Structure

```
customer-churn-project
│
├── data
│   ├── raw
│   │   └── churn_dataset.csv
│   └── processed
│       └── churn_cleaned.csv
│
├── notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── models
│   └── churn_model.pkl
│
├── app
│   └── streamlit_app.py
│
├── database
│   ├── create_tables.sql
│   └── churn_analysis.sql
│
├── reports
│   ├── eda_report.pdf
│   ├── model_report.pdf
│   └── retention_strategy.pdf
│
├── requirements.txt
└── README.md
```

---

## Project Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Deployment (Streamlit Dashboard)

---

## Machine Learning Models

The following models were tested:

- Logistic Regression
- Random Forest
- XGBoost

The best performing model is selected based on evaluation metrics.

---

## Model Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Example:

```
Best Model: Random Forest
Accuracy: 0.89
ROC-AUC: 0.92
```

---

## Key Insights

Analysis shows that churn is strongly related to:

- High number of complaints
- Low customer satisfaction
- Late payments
- Short contract periods
- Frequent service calls

These factors can help businesses identify high‑risk customers.

---

## Retention Strategy

Based on the analysis, companies can reduce churn by:

- Offering discounts to high‑risk customers
- Providing proactive customer support
- Upgrading service plans
- Contacting customers with low satisfaction scores

---

## Deployment

The trained model is deployed using Streamlit.

The dashboard allows users to input customer information such as:

- Age
- Tenure
- Monthly Charges
- Complaints

The system then predicts the Churn Probability.

---

## Technologies Used

- Python
- SQL
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Streamlit
- SHAP

---

## Future Improvements

- Add Deep Learning models
- Deploy the project using Docker
- Connect to a real customer database
- Build a real-time prediction API

---

## Author

Omar Mahmoud
Computer Science Student
