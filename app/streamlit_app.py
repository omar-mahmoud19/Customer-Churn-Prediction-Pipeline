import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Churn Intelligence", page_icon="🔮", layout="wide", initial_sidebar_state="expanded")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
[data-testid="stAppViewContainer"] { background-color: #070b14; }
[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #1e2530; }
#MainMenu, footer, header { visibility: hidden; }
.metric-card { background:#0d1117; border:1px solid #1e2530; border-radius:12px; padding:20px 22px; position:relative; overflow:hidden; }
.metric-card::before { content:''; position:absolute; top:0; left:0; width:3px; height:100%; background:var(--accent); border-radius:12px 0 0 12px; }
.metric-value { font-size:28px; font-weight:700; color:#fff; letter-spacing:-1px; line-height:1; margin-bottom:6px; }
.metric-label { font-size:12px; color:#4a5568; font-weight:500; text-transform:uppercase; letter-spacing:0.5px; }
.metric-change { font-size:11px; margin-top:8px; font-weight:500; }
.page-header { padding:8px 0 32px; border-bottom:1px solid #1e2530; margin-bottom:32px; }
.page-header h2 { font-size:26px; font-weight:700; color:#fff; margin:0 0 6px; }
.page-header p { font-size:14px; color:#4a5568; margin:0; }
.chart-title { font-size:11px; font-weight:600; color:#4a5568; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px; }
.insight-card { background:#0d1117; border:1px solid #1e2530; border-radius:10px; padding:16px 18px; margin:8px 0; display:flex; gap:14px; }
.insight-text { font-size:13px; color:#a0aec0; line-height:1.6; }
.insight-text strong { color:#e2e8f0; }
.result-high { background:linear-gradient(135deg,#1a0a0a,#1f0f0f); border:1px solid #5c1f1f; border-radius:14px; padding:28px 32px; text-align:center; }
.result-low { background:linear-gradient(135deg,#0a1a0f,#0f1f14); border:1px solid #1f5c33; border-radius:14px; padding:28px 32px; text-align:center; }
.result-pct { font-size:52px; font-weight:700; letter-spacing:-2px; line-height:1; margin:8px 0; }
.result-label { font-size:14px; font-weight:500; margin-top:8px; }
.result-high .result-pct, .result-high .result-label { color:#fc8181; }
.result-low .result-pct, .result-low .result-label { color:#68d391; }
.progress-wrap { background:#1e2530; border-radius:99px; height:6px; margin-top:20px; overflow:hidden; }
.progress-high { height:100%; border-radius:99px; background:linear-gradient(90deg,#fc8181,#f56565); }
.progress-low { height:100%; border-radius:99px; background:linear-gradient(90deg,#68d391,#48bb78); }
.stButton > button { background:linear-gradient(135deg,#3b82f6,#2563eb) !important; color:white !important; border:none !important; border-radius:10px !important; padding:14px !important; font-size:14px !important; font-weight:600 !important; width:100% !important; }
[data-testid="stRadio"] label { color:#a0aec0 !important; font-size:13px !important; }
[data-testid="stSlider"] label, [data-testid="stSelectbox"] label { color:#a0aec0 !important; font-size:13px !important; }
.sidebar-stats { background:#0f1621; border:1px solid #1e2530; border-radius:10px; padding:16px; margin-top:24px; }
.stat-row { display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #1e2530; }
.stat-row:last-child { border-bottom:none; }
.stat-key { font-size:12px; color:#4a5568; }
.stat-val { font-size:12px; font-weight:600; color:#e2e8f0; }
</style>""", unsafe_allow_html=True)


# ── Load Data & Model ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, "data/processed/clean_data.csv"))

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, "models/lgbm_model.pkl"))

df = load_data()
model = load_model()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:24px 0 16px;border-bottom:1px solid #1e2530;margin-bottom:24px;">
        <div style="font-size:32px">🔮</div>
        <h1 style="font-size:20px;font-weight:700;color:#fff;margin:8px 0 4px;">Churn Intelligence</h1>
        <p style="font-size:12px;color:#4a5568;margin:0;">Customer Retention Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-size:10px;font-weight:600;color:#4a5568;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:12px;">Navigation</p>', unsafe_allow_html=True)
    page = st.radio("", ["📊 Overview", "📈 EDA", "🤖 Prediction"], label_visibility="collapsed")

    st.markdown("""
    <div class="sidebar-stats">
        <div class="stat-row"><span class="stat-key">Total Customers</span><span class="stat-val">999,999</span></div>
        <div class="stat-row"><span class="stat-key">Churn Rate</span><span class="stat-val" style="color:#fc8181">9.9%</span></div>
        <div class="stat-row"><span class="stat-key">Model</span><span class="stat-val">LightGBM</span></div>
        <div class="stat-row"><span class="stat-key">Features</span><span class="stat-val">37 columns</span></div>
    </div>
    """, unsafe_allow_html=True)


# ── Helper: Dark Chart ────────────────────────────────────────────────────────
def dark_fig():
    fig, ax = plt.subplots(facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    ax.spines[['top','right','left','bottom']].set_visible(False)
    ax.tick_params(colors='#4a5568', labelsize=11)
    ax.yaxis.set_tick_params(length=0)
    ax.yaxis.grid(True, color='#1e2530', linewidth=0.5)
    ax.set_axisbelow(True)
    return fig, ax


# ── PAGE 1: OVERVIEW ──────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.markdown('<div class="page-header"><h2>Overview</h2><p>High-level summary of customer churn across the dataset</p></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("👥", "999,999", "Total Customers", "", "#3b82f6"),
        ("📉", "99,227",  "Churned",          "▲ 9.92% churn rate", "#fc8181"),
        ("✅", "900,772", "Retained",          "▲ 90.08% retention", "#68d391"),
        ("⚠️", "26.5%",  "Month-to-Month Churn", "Highest risk segment", "#f6ad55"),
    ]
    for col, (icon, val, label, change, accent) in zip([c1,c2,c3,c4], cards):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="--accent:{accent}">
                <div style="font-size:18px;margin-bottom:12px">{icon}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-change" style="color:{accent}">{change}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<p style="font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;letter-spacing:1px;margin:32px 0 16px;">Key Insights</p>', unsafe_allow_html=True)
    i1, i2 = st.columns(2)
    with i1:
        st.markdown("""
        <div class="insight-card"><div style="font-size:16px">📋</div><div class="insight-text"><strong>Contract Type</strong> is the strongest churn predictor. Month-to-Month churn at <strong>26.5%</strong> vs <strong>5.6%</strong> for Two-Year.</div></div>
        <div class="insight-card"><div style="font-size:16px">😤</div><div class="insight-text"><strong>Complaints</strong> are a strong early warning signal. Churned customers submit significantly more complaints.</div></div>
        """, unsafe_allow_html=True)
    with i2:
        st.markdown("""
        <div class="insight-card"><div style="font-size:16px">⏱️</div><div class="insight-text"><strong>New customers</strong> (0–10 months) are at highest risk. Churn drops significantly after the first year.</div></div>
        <div class="insight-card"><div style="font-size:16px">💰</div><div class="insight-text"><strong>High monthly charges</strong> correlate with higher churn. Very High segment shows <strong>12%</strong> vs 8% for Medium.</div></div>
        """, unsafe_allow_html=True)


# ── PAGE 2: EDA ───────────────────────────────────────────────────────────────
elif page == "📈 EDA":
    st.markdown('<div class="page-header"><h2>Exploratory Analysis</h2><p>Visual breakdown of churn patterns across key features</p></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="chart-title">Churn Distribution</div>', unsafe_allow_html=True)
        fig, ax = dark_fig()
        vals = df['churn'].value_counts()
        ax.bar(['No Churn', 'Churn'], vals.values, color=['#3b82f6','#fc8181'], width=0.5, edgecolor='none')
        st.pyplot(fig); plt.close()

    with c2:
        st.markdown('<div class="chart-title">Churn Rate by Contract Type</div>', unsafe_allow_html=True)
        fig, ax = dark_fig()
        df['contract_label'] = df['contract'].map({0:'Month-to-Month', 1:'One Year', 2:'Two Year'})
        rates = df.groupby('contract_label')['churn'].mean() * 100
        bars = ax.bar(rates.index, rates.values, color=['#fc8181','#f6ad55','#68d391'], width=0.5, edgecolor='none')
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f'{b.get_height():.1f}%',
                    ha='center', va='bottom', color='#a0aec0', fontsize=10, fontweight='600')
        st.pyplot(fig); plt.close()

    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="chart-title">Avg Complaints by Churn</div>', unsafe_allow_html=True)
        fig, ax = dark_fig()
        comp = df.groupby('churn')['num_complaints'].mean()
        ax.bar(['No Churn','Churn'], comp.values, color=['#3b82f6','#fc8181'], width=0.5, edgecolor='none')
        st.pyplot(fig); plt.close()

    with c4:
        st.markdown('<div class="chart-title">Avg Satisfaction by Churn</div>', unsafe_allow_html=True)
        fig, ax = dark_fig()
        sat = df.groupby('churn')['customer_satisfaction'].mean()
        margin = abs(sat.values.max() - sat.values.min()) * 0.5
        bars = ax.bar(['No Churn','Churn'], sat.values, color=['#3b82f6','#fc8181'], width=0.5, edgecolor='none')
        ax.set_ylim(sat.values.min() - margin, sat.values.max() + margin)
        for b, v in zip(bars, sat.values):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+margin*0.1, f'{v:.2f}',
                    ha='center', va='bottom', color='#a0aec0', fontsize=10, fontweight='600')
        st.pyplot(fig); plt.close()


# ── PAGE 3: PREDICTION ────────────────────────────────────────────────────────
elif page == "🤖 Prediction":
    st.markdown('<div class="page-header"><h2>Churn Prediction</h2><p>Enter customer details to predict churn probability</p></div>', unsafe_allow_html=True)

    col_in, col_out = st.columns([1.4, 1])

    with col_in:
        c1, c2 = st.columns(2)
        with c1:
            age              = st.slider("Age", 18, 90, 35)
            tenure           = st.slider("Tenure (Months)", 0, 72, 12)
            monthly_charges  = st.slider("Monthly Charges ($)", 20, 120, 65)
        with c2:
            num_complaints        = st.slider("Complaints", 0, 7, 1)
            customer_satisfaction = st.slider("Satisfaction (1-9)", 1, 9, 5)
            credit_score          = st.slider("Credit Score", 300, 850, 650)

        c3, c4 = st.columns(2)
        with c3:
            contract     = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
            num_services = st.slider("Number of Services", 0, 8, 3)
        with c4:
            late_payments = st.slider("Late Payments", 0, 10, 0)

        st.markdown("<br>", unsafe_allow_html=True)
        predict = st.button("🔮  Run Prediction", use_container_width=True)

    with col_out:
        st.markdown('<p style="font-size:10px;font-weight:600;color:#4a5568;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">Prediction Result</p>', unsafe_allow_html=True)

        if predict:
            input_data = pd.DataFrame([{col: 0 for col in df.drop(columns=['churn']).columns}])
            input_data['age']                   = age
            input_data['tenure']                = tenure
            input_data['monthlycharges']        = monthly_charges
            input_data['num_complaints']        = num_complaints
            input_data['customer_satisfaction'] = customer_satisfaction
            input_data['credit_score']          = credit_score
            input_data['contract']              = {"Month-to-Month":0,"One Year":1,"Two Year":2}[contract]
            input_data['num_services']          = num_services
            input_data['late_payments']         = late_payments

            prob = model.predict_proba(input_data)[0][1]
            pct  = prob * 100

            if prob > 0.5:
                st.markdown(f"""
                <div class="result-high">
                    <div style="font-size:12px;color:#4a5568;text-transform:uppercase;letter-spacing:1px;">Churn Probability</div>
                    <div class="result-pct">{pct:.1f}%</div>
                    <div class="result-label">⚠️ High Risk — Immediate action needed</div>
                    <div class="progress-wrap"><div class="progress-high" style="width:{int(pct)}%"></div></div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-low">
                    <div style="font-size:12px;color:#4a5568;text-transform:uppercase;letter-spacing:1px;">Churn Probability</div>
                    <div class="result-pct">{pct:.1f}%</div>
                    <div class="result-label">✅ Low Risk — Customer is stable</div>
                    <div class="progress-wrap"><div class="progress-low" style="width:{int(pct)}%"></div></div>
                </div>""", unsafe_allow_html=True)

            # Risk Factors
            risks = []
            if num_complaints > 3:      risks.append(("😤", "<strong>High complaint count</strong> — Major churn driver"))
            if contract == "Month-to-Month": risks.append(("📋", "<strong>Month-to-Month contract</strong> — Highest risk segment"))
            if tenure < 12:             risks.append(("⏱️", "<strong>New customer</strong> — First year is critical"))
            if customer_satisfaction < 4: risks.append(("😞", "<strong>Low satisfaction</strong> — Needs immediate follow-up"))

            if risks:
                st.markdown('<p style="font-size:10px;font-weight:600;color:#4a5568;text-transform:uppercase;letter-spacing:1px;margin:20px 0 8px;">Risk Factors</p>', unsafe_allow_html=True)
                for icon, text in risks:
                    st.markdown(f'<div class="insight-card"><div style="font-size:16px">{icon}</div><div class="insight-text">{text}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#0d1117;border:1px solid #1e2530;border-radius:14px;padding:48px 32px;text-align:center;margin-top:8px;">
                <div style="font-size:40px;margin-bottom:16px">🔮</div>
                <div style="font-size:14px;color:#4a5568;line-height:1.6">Fill in the customer details<br>and click <strong style="color:#a0aec0">Run Prediction</strong></div>
            </div>""", unsafe_allow_html=True)