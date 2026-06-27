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
.ret-card { background:#0d1117; border:1px solid #1e2530; border-radius:12px; padding:20px; margin:10px 0; }
.ret-card-high { border-left:4px solid #fc8181; }
.ret-card-med  { border-left:4px solid #f6ad55; }
.ret-card-low  { border-left:4px solid #68d391; }
.ret-title { font-size:14px; font-weight:600; color:#e2e8f0; margin-bottom:10px; display:flex; gap:8px; align-items:center; }
.ret-tag { font-size:11px; font-weight:600; padding:2px 8px; border-radius:99px; flex-shrink:0; }
.tag-urgent { background:#2d1515; color:#fc8181; }
.tag-soon   { background:#2d1e0a; color:#f6ad55; }
.tag-normal { background:#0a2d15; color:#68d391; }
.score-bar-wrap { background:#1e2530; border-radius:99px; height:8px; margin:4px 0 12px; overflow:hidden; }
.score-bar { height:100%; border-radius:99px; }
.risk-score-box { background:#0d1117; border:1px solid #1e2530; border-radius:12px; padding:20px; text-align:center; }
.risk-score-num { font-size:48px; font-weight:700; letter-spacing:-2px; }
.risk-label-text { font-size:13px; color:#4a5568; margin-top:4px; }
</style>""", unsafe_allow_html=True)


# ── Load Data & Model ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, "data/processed/clean_data_sample.csv"))

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
    page = st.radio("", ["📊 Overview", "📈 EDA", "🤖 Prediction", "🎯 Retention Plan"], label_visibility="collapsed")

    st.markdown("""
    <div class="sidebar-stats">
        <div class="stat-row"><span class="stat-key">Total Customers</span><span class="stat-val">999,999</span></div>
        <div class="stat-row"><span class="stat-key">Churn Rate</span><span class="stat-val" style="color:#fc8181">9.9%</span></div>
        <div class="stat-row"><span class="stat-key">Model</span><span class="stat-val">LightGBM</span></div>
        <div class="stat-row"><span class="stat-key">Features</span><span class="stat-val">37 columns</span></div>
    </div>
    """, unsafe_allow_html=True)


# ── Helper ────────────────────────────────────────────────────────────────────
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
        ("👥", "999,999", "Total Customers",        "",                      "#3b82f6"),
        ("📉", "99,227",  "Churned",                "▲ 9.92% churn rate",    "#fc8181"),
        ("✅", "900,772", "Retained",               "▲ 90.08% retention",    "#68d391"),
        ("⚠️", "26.5%",  "Month-to-Month Churn",   "Highest risk segment",  "#f6ad55"),
    ]
    for col, (icon, val, label, change, accent) in zip([c1,c2,c3,c4], cards):
        with col:
            st.markdown(f"""<div class="metric-card" style="--accent:{accent}">
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
            age             = st.slider("Age", 18, 90, 35)
            tenure          = st.slider("Tenure (Months)", 0, 72, 12)
            monthly_charges = st.slider("Monthly Charges ($)", 20, 120, 65)
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
            input_data = pd.DataFrame([{col: 0 for col in df.drop(columns=['churn']).columns if col != 'contract_label'}])
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
                st.markdown(f"""<div class="result-high">
                    <div style="font-size:12px;color:#4a5568;text-transform:uppercase;letter-spacing:1px;">Churn Probability</div>
                    <div class="result-pct">{pct:.1f}%</div>
                    <div class="result-label">⚠️ High Risk — Immediate action needed</div>
                    <div class="progress-wrap"><div class="progress-high" style="width:{int(pct)}%"></div></div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="result-low">
                    <div style="font-size:12px;color:#4a5568;text-transform:uppercase;letter-spacing:1px;">Churn Probability</div>
                    <div class="result-pct">{pct:.1f}%</div>
                    <div class="result-label">✅ Low Risk — Customer is stable</div>
                    <div class="progress-wrap"><div class="progress-low" style="width:{int(pct)}%"></div></div>
                </div>""", unsafe_allow_html=True)

            risks = []
            if num_complaints > 3:       risks.append(("😤", "<strong>High complaint count</strong> — Major churn driver"))
            if contract == "Month-to-Month": risks.append(("📋", "<strong>Month-to-Month contract</strong> — Highest risk segment"))
            if tenure < 12:              risks.append(("⏱️", "<strong>New customer</strong> — First year is critical"))
            if customer_satisfaction < 4: risks.append(("😞", "<strong>Low satisfaction</strong> — Needs immediate follow-up"))

            if risks:
                st.markdown('<p style="font-size:10px;font-weight:600;color:#4a5568;text-transform:uppercase;letter-spacing:1px;margin:20px 0 8px;">Risk Factors</p>', unsafe_allow_html=True)
                for icon, text in risks:
                    st.markdown(f'<div class="insight-card"><div style="font-size:16px">{icon}</div><div class="insight-text">{text}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown("""<div style="background:#0d1117;border:1px solid #1e2530;border-radius:14px;padding:48px 32px;text-align:center;margin-top:8px;">
                <div style="font-size:40px;margin-bottom:16px">🔮</div>
                <div style="font-size:14px;color:#4a5568;line-height:1.6">Fill in the customer details<br>and click <strong style="color:#a0aec0">Run Prediction</strong></div>
            </div>""", unsafe_allow_html=True)


# ── PAGE 4: RETENTION PLAN ────────────────────────────────────────────────────
elif page == "🎯 Retention Plan":
    st.markdown('<div class="page-header"><h2>Retention Plan</h2><p>Analyze customer risk and get a tailored retention strategy</p></div>', unsafe_allow_html=True)

    st.markdown('<p style="font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;letter-spacing:1px;margin-bottom:16px;">Customer Profile</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        r_age          = st.slider("Age", 18, 90, 35, key="r_age")
        r_tenure       = st.slider("Tenure (Months)", 0, 72, 6, key="r_tenure")
        r_monthly      = st.slider("Monthly Charges ($)", 20, 120, 80, key="r_monthly")
    with c2:
        r_complaints   = st.slider("Complaints", 0, 7, 2, key="r_complaints")
        r_satisfaction = st.slider("Satisfaction (1-9)", 1, 9, 3, key="r_satisfaction")
        r_late         = st.slider("Late Payments", 0, 10, 1, key="r_late")
    with c3:
        r_contract     = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"], key="r_contract")
        r_services     = st.slider("Number of Services", 0, 8, 2, key="r_services")
        r_credit       = st.slider("Credit Score", 300, 850, 550, key="r_credit")

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🎯  Generate Retention Plan", use_container_width=True)

    if analyze_btn:
        contract_val = {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}[r_contract]
        input_data = pd.DataFrame([{col: 0 for col in df.drop(columns=['churn']).columns if col != 'contract_label'}])
        input_data['age']                   = r_age
        input_data['tenure']                = r_tenure
        input_data['monthlycharges']        = r_monthly
        input_data['num_complaints']        = r_complaints
        input_data['customer_satisfaction'] = r_satisfaction
        input_data['credit_score']          = r_credit
        input_data['contract']              = contract_val
        input_data['num_services']          = r_services
        input_data['late_payments']         = r_late

        prob = model.predict_proba(input_data)[0][1]
        pct  = prob * 100
        risk_score = int(pct)

        if risk_score >= 50:
            risk_level, risk_color, bar_color = "HIGH RISK",    "#fc8181", "linear-gradient(90deg,#fc8181,#f56565)"
        elif risk_score >= 30:
            risk_level, risk_color, bar_color = "MEDIUM RISK",  "#f6ad55", "linear-gradient(90deg,#f6ad55,#ed8936)"
        else:
            risk_level, risk_color, bar_color = "LOW RISK",     "#68d391", "linear-gradient(90deg,#68d391,#48bb78)"

        col_score, col_plan = st.columns([1, 2])

        with col_score:
            st.markdown(f"""<div class="risk-score-box">
                <div style="font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Churn Probability</div>
                <div class="risk-score-num" style="color:{risk_color}">{pct:.1f}%</div>
                <div class="score-bar-wrap" style="margin:12px 0;">
                    <div class="score-bar" style="width:{risk_score}%;background:{bar_color}"></div>
                </div>
                <div style="font-size:13px;font-weight:600;color:{risk_color}">{risk_level}</div>
                <div class="risk-label-text">Based on LightGBM model</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p style="font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">Risk Factor Scores</p>', unsafe_allow_html=True)

            factors = [
                ("Complaints",    min(r_complaints / 7, 1),        "#fc8181"),
                ("Satisfaction",  1 - (r_satisfaction / 9),        "#f6ad55"),
                ("Tenure Risk",   max(0, 1 - r_tenure / 72),       "#a78bfa"),
                ("Late Payments", min(r_late / 10, 1),             "#fc8181"),
                ("Contract Risk", [0.9, 0.4, 0.1][contract_val],   "#f6ad55"),
            ]
            for name, score, color in factors:
                st.markdown(f"""<div style="margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                        <span style="font-size:12px;color:#a0aec0">{name}</span>
                        <span style="font-size:12px;font-weight:600;color:{color}">{int(score*100)}%</span>
                    </div>
                    <div class="score-bar-wrap" style="margin:0;">
                        <div class="score-bar" style="width:{int(score*100)}%;background:{color}"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

        with col_plan:
            st.markdown('<p style="font-size:11px;font-weight:600;color:#4a5568;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">Recommended Retention Actions</p>', unsafe_allow_html=True)

            actions = []

            if r_contract == "Month-to-Month":
                actions.append(("urgent", "📋", "Contract Upgrade Offer",
                    "Offer a 25% discount for switching to a One-Year contract. Month-to-Month customers churn at 26.5% vs 5.6% for Two-Year — this is the highest-impact action."))
            elif r_contract == "One Year":
                actions.append(("soon", "📋", "Long-Term Contract Incentive",
                    "Offer a loyalty bonus (free month or service upgrade) to encourage upgrading to a Two-Year contract."))

            if r_complaints >= 4:
                actions.append(("urgent", "📞", "Priority Customer Support Call",
                    f"Customer has {r_complaints} complaints. Assign a dedicated support agent within 24 hours. Offer service credit ($20–$50) as goodwill gesture."))
            elif r_complaints >= 2:
                actions.append(("soon", "📞", "Proactive Support Outreach",
                    f"Schedule a courtesy call within 3 days to address {r_complaints} recent complaints and ensure satisfaction."))

            if r_satisfaction <= 3:
                actions.append(("urgent", "⭐", "Satisfaction Recovery Program",
                    f"Critical satisfaction score ({r_satisfaction}/9). Send a personalized survey immediately. Offer a free service upgrade for 2 months."))
            elif r_satisfaction <= 5:
                actions.append(("soon", "⭐", "Satisfaction Improvement",
                    f"Satisfaction at {r_satisfaction}/9 is below average. Send an NPS survey and offer a loyalty reward."))

            if r_tenure <= 6:
                actions.append(("urgent", "🎁", "New Customer Onboarding",
                    f"Customer is only {r_tenure} months in — the highest churn risk window. Enroll in 90-day onboarding program with weekly check-ins and a welcome bonus."))
            elif r_tenure <= 12:
                actions.append(("soon", "🎁", "First Year Loyalty Reward",
                    "Approaching first anniversary. Offer a loyalty gift (free month or service add-on) to reinforce relationship."))

            if r_monthly >= 90:
                actions.append(("soon", "💰", "Pricing Review",
                    f"Monthly charge of ${r_monthly} is in the highest churn tier. Offer a bundle discount or a cheaper equivalent plan."))

            if r_late >= 3:
                actions.append(("soon", "💳", "Payment Flexibility Program",
                    f"{r_late} late payments detected. Offer a flexible payment plan or auto-pay setup with a 5% discount incentive."))

            if r_services <= 2:
                actions.append(("normal", "📡", "Service Bundle Upsell",
                    f"Customer only has {r_services} services. Offer a bundle deal with 3+ services at 15% discount — more services means lower churn risk."))

            if not actions:
                actions.append(("normal", "✅", "Loyalty Maintenance",
                    "Customer shows low churn risk. Send a thank-you message and offer early access to new services to reinforce loyalty."))

            tag_map = {
                "urgent": ("URGENT", "tag-urgent", "ret-card-high"),
                "soon":   ("SOON",   "tag-soon",   "ret-card-med"),
                "normal": ("NORMAL", "tag-normal", "ret-card-low"),
            }

            for priority, icon, title, desc in actions:
                tag_label, tag_cls, card_cls = tag_map[priority]
                st.markdown(f"""<div class="ret-card {card_cls}">
                    <div class="ret-title">
                        <span>{icon}</span><span>{title}</span>
                        <span class="ret-tag {tag_cls}">{tag_label}</span>
                    </div>
                    <div style="font-size:13px;color:#a0aec0;line-height:1.7;">{desc}</div>
                </div>""", unsafe_allow_html=True)

            urgent_count = sum(1 for a in actions if a[0] == "urgent")
            soon_count   = sum(1 for a in actions if a[0] == "soon")
            normal_count = len(actions) - urgent_count - soon_count

            st.markdown(f"""<div style="background:#0d1117;border:1px solid #1e2530;border-radius:10px;padding:16px;margin-top:16px;display:flex;gap:24px;align-items:center;">
                <div style="text-align:center;">
                    <div style="font-size:24px;font-weight:700;color:#fc8181">{urgent_count}</div>
                    <div style="font-size:11px;color:#4a5568;text-transform:uppercase;letter-spacing:.5px">Urgent</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:24px;font-weight:700;color:#f6ad55">{soon_count}</div>
                    <div style="font-size:11px;color:#4a5568;text-transform:uppercase;letter-spacing:.5px">Soon</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:24px;font-weight:700;color:#68d391">{normal_count}</div>
                    <div style="font-size:11px;color:#4a5568;text-transform:uppercase;letter-spacing:.5px">Normal</div>
                </div>
                <div style="flex:1;font-size:13px;color:#4a5568;text-align:right;">
                    {len(actions)} action{"s" if len(actions) != 1 else ""} recommended · Start with URGENT items first
                </div>
            </div>""", unsafe_allow_html=True)