"""
app_page.py
===========
Role: BI Developer / Deployment Engineer
Purpose: Streamlit dashboard with KPIs, visualizations, and a
         single-customer churn prediction interface.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import sys

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
META_PATH  = os.path.join(BASE_DIR, "models", "model_meta.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "data", "raw", "customer_churn.csv")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563EB 100%);
        border-radius: 12px; padding: 20px; color: white; text-align: center;
        margin: 6px;
    }
    .metric-value { font-size: 2.2rem; font-weight: 800; }
    .metric-label { font-size: 0.9rem; opacity: 0.85; margin-top: 4px; }
    .churn-badge-yes {
        background: #EF4444; color: white; padding: 8px 18px;
        border-radius: 20px; font-size: 1.1rem; font-weight: 700;
    }
    .churn-badge-no {
        background: #16A34A; color: white; padding: 8px 18px;
        border-radius: 20px; font-size: 1.1rem; font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return pipeline, meta


@st.cache_data
def load_data(n: int = 100_000):
    return pd.read_csv(DATA_PATH, nrows=n)


pipeline, meta = load_model()
df = load_data()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/customer-support.png", width=70)
    st.title("Churn Intelligence")
    st.markdown(f"**Model:** `{meta['best_model_name']}`")
    st.markdown(f"**ROC-AUC:** `{max(r['roc_auc'] for r in meta['all_results']):.4f}`")
    st.divider()
    page = st.radio("Navigate", ["📊 Dashboard", "🔮 Predict Customer"])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📉 Customer Churn Intelligence Dashboard")
    st.caption("Real-time insights from 1M+ customer records")

    # ── KPI cards ──────────────────────────────────────────────────────────────
    churn_rate = df["ch"].mean() * 100
    churned    = int(df["ch"].sum())
    total      = len(df)
    avg_tenure = df["tenure"].mean()
    avg_charge = df["monthlycharges"].mean()

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1, c2, c3, c4],
        [f"{churn_rate:.1f}%", f"{churned:,}", f"{avg_tenure:.0f} mo", f"${avg_charge:.0f}"],
        ["Churn Rate", "Churned Customers", "Avg Tenure", "Avg Monthly Charge"],
    ):
        col.markdown(
            f'<div class="metric-card"><div class="metric-value">{val}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Charts row 1 ──────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Churn Rate by Contract Type")
        churn_contract = (
            df.groupby("contract")["ch"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#EF4444" if v > churn_rate else "#2563EB" for v in churn_contract["ch"]]
        ax.barh(churn_contract["contract"], churn_contract["ch"], color=colors, edgecolor="white")
        ax.axvline(churn_rate, color="gray", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Churn Rate (%)")
        ax.set_facecolor("#F8FAFC")
        fig.patch.set_facecolor("#F8FAFC")
        st.pyplot(fig)
        plt.close(fig)

    with col_right:
        st.subheader("Churn Rate by Tenure Bucket")
        df["tenure_bucket"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 48, 999],
            labels=["0-12 mo", "12-24 mo", "24-48 mo", "48+ mo"],
        )
        churn_tenure = (
            df.groupby("tenure_bucket", observed=True)["ch"]
            .mean().mul(100).reset_index()
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(
            churn_tenure["tenure_bucket"].astype(str),
            churn_tenure["ch"],
            color=["#EF4444", "#F97316", "#FACC15", "#22C55E"],
            edgecolor="white",
        )
        ax.axhline(churn_rate, color="gray", linestyle="--", linewidth=1.2)
        ax.set_ylabel("Churn Rate (%)")
        ax.set_facecolor("#F8FAFC")
        fig.patch.set_facecolor("#F8FAFC")
        st.pyplot(fig)
        plt.close(fig)

    # ── Charts row 2 ──────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Monthly Charges Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(
            df[df["ch"] == 0]["monthlycharges"].dropna(), bins=50,
            alpha=0.6, color="#2563EB", label="No Churn", density=True,
        )
        ax.hist(
            df[df["ch"] == 1]["monthlycharges"].dropna(), bins=50,
            alpha=0.6, color="#EF4444", label="Churn", density=True,
        )
        ax.set_xlabel("Monthly Charges ($)")
        ax.legend()
        ax.set_facecolor("#F8FAFC")
        fig.patch.set_facecolor("#F8FAFC")
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.subheader("Model Performance Comparison")
        results_df = pd.DataFrame(meta["all_results"])
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        x = np.arange(len(metrics))
        width = 0.25
        fig, ax = plt.subplots(figsize=(6, 4))
        palette = ["#2563EB", "#16A34A", "#D97706"]
        for i, (_, row) in enumerate(results_df.iterrows()):
            ax.bar(x + i * width, [row[m] for m in metrics], width,
                   label=row["model"], color=palette[i % len(palette)], edgecolor="white")
        ax.set_xticks(x + width)
        ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.legend(fontsize=8)
        ax.set_facecolor("#F8FAFC")
        fig.patch.set_facecolor("#F8FAFC")
        st.pyplot(fig)
        plt.close(fig)

    # ── Data table ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Model Metrics Summary")
    st.dataframe(
        results_df.sort_values("roc_auc", ascending=False)
        .set_index("model")
        .style.background_gradient(cmap="Blues", axis=None)
        .format("{:.4f}"),
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SINGLE CUSTOMER PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.title("🔮 Predict Churn for a Single Customer")
    st.caption("Enter customer details below and click **Predict** to get the churn probability.")

    with st.form("prediction_form"):
        st.subheader("👤 Demographic Information")
        c1, c2, c3 = st.columns(3)
        age            = c1.number_input("Age", 18, 90, 35)
        gender         = c2.selectbox("Gender", ["Male", "Female"])
        marital_status = c3.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        annual_income  = c1.number_input("Annual Income ($)", 10_000, 300_000, 65_000, step=1_000)
        education      = c2.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
        dependents     = c3.number_input("Dependents", 0, 10, 0)
        senior_citizen = c1.selectbox("Senior Citizen", [0, 1])

        st.subheader("📋 Contract & Billing")
        c4, c5, c6 = st.columns(3)
        contract        = c4.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment_method  = c5.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        paperless       = c6.selectbox("Paperless Billing", ["Yes", "No"])
        tenure          = c4.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = c5.number_input("Monthly Charges ($)", 15.0, 200.0, 65.0, step=0.5)
        total_charges   = c6.number_input("Total Charges ($)", 0.0, 10_000.0,
                                           float(tenure * monthly_charges))

        st.subheader("📡 Services Subscribed")
        sc1, sc2, sc3, sc4 = st.columns(4)
        has_phone     = int(sc1.checkbox("Phone Service", True))
        has_internet  = int(sc2.checkbox("Internet Service", True))
        has_security  = int(sc3.checkbox("Online Security", False))
        has_backup    = int(sc4.checkbox("Online Backup", False))
        has_device    = int(sc1.checkbox("Device Protection", False))
        has_support   = int(sc2.checkbox("Tech Support", False))
        has_tv        = int(sc3.checkbox("Streaming TV", False))
        has_movies    = int(sc4.checkbox("Streaming Movies", False))
        num_services  = sum([has_phone, has_internet, has_security, has_backup,
                             has_device, has_support, has_tv, has_movies])

        st.subheader("📊 Usage & Satisfaction")
        uc1, uc2, uc3 = st.columns(3)
        satisfaction      = uc1.slider("Customer Satisfaction (1–5)", 1.0, 5.0, 3.5, 0.5)
        num_complaints    = uc2.number_input("Number of Complaints", 0, 20, 0)
        num_service_calls = uc3.number_input("Service Calls", 0, 20, 1)
        late_payments     = uc1.number_input("Late Payments", 0, 30, 0)
        avg_monthly_gb    = uc2.number_input("Avg Monthly GB", 0.0, 500.0, 30.0)
        days_since        = uc3.number_input("Days Since Last Interaction", 0, 365, 30)
        credit_score      = uc1.number_input("Credit Score", 300, 900, 650)

        submitted = st.form_submit_button("🔮  Predict Churn", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([{
            "customer_id":              "CUST_INPUT",
            "signup_date":              "2022-01-01 00:00:00",
            "age":                      age,
            "gender":                   gender,
            "annual_income":            annual_income,
            "education":                education,
            "marital_status":           marital_status,
            "dependents":               dependents,
            "tenure":                   tenure,
            "contract":                 contract,
            "payment_method":           payment_method,
            "paperless_billing":        paperless,
            "senior_citizen":           senior_citizen,
            "monthlycharges":           monthly_charges,
            "totalcharges":             total_charges,
            "num_services":             num_services,
            "has_phone_service":        has_phone,
            "has_internet_service":     has_internet,
            "has_online_security":      has_security,
            "has_online_backup":        has_backup,
            "has_device_protection":    has_device,
            "has_tech_support":         has_support,
            "has_streaming_tv":         has_tv,
            "has_streaming_movies":     has_movies,
            "customer_satisfaction":    satisfaction,
            "num_complaints":           num_complaints,
            "num_service_calls":        num_service_calls,
            "late_payments":            late_payments,
            "avg_monthly_gb":           avg_monthly_gb,
            "days_since_last_interaction": days_since,
            "credit_score":             credit_score,
        }])

        prob  = pipeline.predict_proba(input_df)[0][1]
        label = pipeline.predict(input_df)[0]

        st.divider()
        st.subheader("Prediction Result")

        col_res, col_gauge = st.columns([1, 2])
        with col_res:
            if label == 1:
                st.markdown(
                    '<span class="churn-badge-yes">⚠️  HIGH CHURN RISK</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="churn-badge-no">✅  LOW CHURN RISK</span>',
                    unsafe_allow_html=True,
                )
            st.metric("Churn Probability", f"{prob * 100:.1f}%")
            st.metric("Predicted Class", "Churn ❌" if label == 1 else "No Churn ✅")

        with col_gauge:
            fig, ax = plt.subplots(figsize=(7, 1.4))
            bar_color = "#EF4444" if prob > 0.5 else "#16A34A"
            ax.barh(["Churn Probability"], [prob], color=bar_color, height=0.5, edgecolor="white")
            ax.barh(["Churn Probability"], [1], color="#E2E8F0", height=0.5, zorder=0)
            ax.axvline(0.5, color="gray", linestyle="--", linewidth=1.5)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            ax.text(prob + 0.02, 0, f"{prob*100:.1f}%", va="center", fontweight="bold")
            ax.set_facecolor("#F8FAFC")
            fig.patch.set_facecolor("#F8FAFC")
            st.pyplot(fig)
            plt.close(fig)

        # ── Risk factors ──────────────────────────────────────────────────────
        st.subheader("⚡ Risk Factor Summary")
        risk_factors = []
        if contract == "Month-to-month":
            risk_factors.append("🔴 Month-to-month contract — highest churn segment")
        if tenure < 12:
            risk_factors.append("🔴 Short tenure (<12 months) — new customer risk")
        if satisfaction < 3:
            risk_factors.append("🔴 Low satisfaction score")
        if num_complaints > 2:
            risk_factors.append("🔴 High complaint count")
        if late_payments > 2:
            risk_factors.append("🟡 Multiple late payments")
        if num_services < 2:
            risk_factors.append("🟡 Low service adoption")
        if not risk_factors:
            risk_factors.append("🟢 No major risk flags detected")
        for rf in risk_factors:
            st.markdown(f"- {rf}")
