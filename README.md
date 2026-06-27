# 🔮 Customer Churn Prediction

A complete end-to-end Machine Learning project that predicts customer churn for a telecom company using real-world data of 999,999 customers.

---

## 📁 Project Structure

```
customer-churn-project/
│
├── data/
│   ├── raw/                        # Original dataset (do not modify)
│   │   └── customer_churn.csv
│   └── processed/                  # Cleaned & encoded dataset
│       └── clean_data.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Initial data exploration
│   ├── 02_data_preprocessing.ipynb # Cleaning, encoding, scaling
│   ├── 03_model_training.ipynb     # Model building & comparison
│   ├── 04_model_evaluation.ipynb   # Metrics, confusion matrix
│   └── 05_sql_analysis.ipynb       # SQL queries & insights
│
├── database/
│   ├── schema.sql                  # Database schema
│   ├── load_data.py                # Load CSV to SQL Server
│   └── queries/
│       ├── 01_churn_rate.sql
│       ├── 02_churn_by_contract.sql
│       ├── 03_high_risk_customers.sql
│       ├── 04_churn_by_charges.sql
│       └── 05_churn_by_tenure.sql
│
├── models/
│   ├── lgbm_model.pkl              # Trained LightGBM model
│   └── scaler/
│       └── scaler.pk1              # Fitted StandardScaler
│
├── app/
│   └── streamlit_app.py            # Interactive dashboard
│
├── reports/
│   ├── data_engineering_report.pdf
│   ├── data_science_report.pdf
│   ├── eda_report.pdf
│   ├── sql_analysis_report.pdf
│   └── streamlit_report.pdf
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

| Property   | Value                          |
| ---------- | ------------------------------ |
| Source     | customer_churn.csv             |
| Rows       | 999,999 customers              |
| Columns    | 32 features                    |
| Target     | `churn` (0 = Stayed, 1 = Left) |
| Churn Rate | 9.92%                          |

---

## 👥 Team Roles

| Role           | Responsibilities                      |
| -------------- | ------------------------------------- |
| Data Engineer  | Cleaning, encoding, scaling, pipeline |
| Data Scientist | Model building, evaluation, selection |
| Data Analyst   | EDA, visualizations, insights         |
| SQL Engineer   | Database design, queries, analysis    |
| BI Developer   | Streamlit dashboard, prediction tool  |

---

## 🤖 Models Trained

| Model               | F1-Score (Churn) | Recall | ROC-AUC   |
| ------------------- | ---------------- | ------ | --------- |
| **LightGBM** ✅     | **0.257**        | 0.636  | **0.635** |
| Logistic Regression | 0.256            | 0.642  | 0.635     |
| XGBoost             | 0.253            | 0.603  | 0.627     |
| Random Forest       | 0.001            | 0.000  | 0.500     |

**Winner: LightGBM** — Best F1-Score and ROC-AUC on 1M rows.

---

## 🔑 Key Insights

1. **Month-to-Month contracts** churn at **26.5%** — nearly 5x Two-Year contracts
2. **New customers** (0–10 months) are at highest churn risk
3. **High complaint count** is the strongest behavioral churn signal
4. **Low satisfaction scores** predict churn with high reliability
5. **Very High monthly charges** correlate with 12% churn rate

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

```bash
# Run preprocessing notebook
jupyter notebook notebooks/02_data_preprocessing.ipynb
```

### 3. Train model

```bash
# Run training notebook
jupyter notebook notebooks/03_model_training.ipynb
```

### 4. Launch dashboard

```bash
cd D:\project
streamlit run app/streamlit_app.py
```

---

## 🛠️ Tech Stack

| Category         | Tools                           |
| ---------------- | ------------------------------- |
| Language         | Python 3.14.1                   |
| Data Processing  | Pandas, NumPy, Scikit-learn     |
| Machine Learning | LightGBM, XGBoost, Scikit-learn |
| Database         | Microsoft SQL Server 16.0       |
| Visualization    | Matplotlib, Seaborn             |
| Deployment       | Streamlit                       |
| IDE              | VS Code + Jupyter Notebooks     |
