import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import re

st.set_page_config(page_title="Churn Intelligence", page_icon="🔮", layout="wide", initial_sidebar_state="collapsed")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Design Tokens ─────────────────────────────────────────────────────────────
BG          = "#070B14"
CARD        = "#111827"
CARD_GLASS  = "rgba(17,24,39,0.62)"
BORDER      = "#1F2937"
BLUE        = "#4F8CFF"
SUCCESS     = "#22C55E"
WARNING     = "#F59E0B"
DANGER      = "#EF4444"
PURPLE      = "#A78BFA"
CYAN        = "#06B6D4"
TEXT_PRI    = "#F9FAFB"
TEXT_SEC    = "#9CA3AF"

# ── Numeric-value counter helper ──────────────────────────────────────────
# Wraps a displayed metric value so the injected JS below can animate it
# counting up from 0 to its final number on first paint, while keeping any
# non-numeric prefix/suffix (%, commas, currency, etc.) intact.
def counter_span(display_value):
    match = re.search(r"[-+]?[0-9][0-9,]*\.?[0-9]*", str(display_value))
    if not match:
        return display_value
    num_str = match.group(0)
    target = float(num_str.replace(",", ""))
    decimals = len(num_str.split(".")[1]) if "." in num_str else 0
    prefix, suffix = display_value[:match.start()], display_value[match.end():]
    return (f'{prefix}<span class="counter-value" data-target="{target}" '
            f'data-decimals="{decimals}" data-format="{"comma" if "," in num_str else "plain"}">0</span>{suffix}')

PAGES = ["📊 Overview", "📈 EDA", "🤖 Prediction", "🎯 Retention Plan"]

# ── Per-page decorative shape (SVG watermark motif) ───────────────────────────
# NOTE: no leading indentation on any line here — Streamlit's markdown parser
# treats 4+ leading spaces as a code block and renders it as literal text
# instead of HTML, which is why the raw <svg> tags showed up on screen.
def page_shape_svg(accent):
    return (
        f'<svg class="header-shape" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">'
        f'<circle cx="150" cy="40" r="46" fill="{accent}" opacity="0.10"/>'
        f'<circle cx="178" cy="90" r="22" fill="{accent}" opacity="0.14"/>'
        f'<circle cx="120" cy="18" r="10" fill="{accent}" opacity="0.22"/>'
        f'<path d="M60 150 L140 70" stroke="{accent}" stroke-width="1.4" opacity="0.18" stroke-dasharray="4 6"/>'
        f'<circle cx="60" cy="150" r="5" fill="{accent}" opacity="0.3"/>'
        f'<rect x="20" y="20" width="14" height="14" rx="4" fill="{accent}" opacity="0.16" transform="rotate(18 27 27)"/>'
        f'<polygon points="170,150 180,168 160,168" fill="{accent}" opacity="0.18"/>'
        f'<path d="M10 90 Q 40 60, 70 90 T 130 90" stroke="{accent}" stroke-width="1.2" opacity="0.14" fill="none"/>'
        f'</svg>'
    )

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');

* {{ font-family: 'Inter', sans-serif; }}

html, body, [data-testid="stAppViewContainer"] {{
    background-color: {BG};
    background-image:
        radial-gradient(ellipse 700px 420px at 8% -8%, rgba(79,140,255,0.12), transparent 60%),
        radial-gradient(ellipse 620px 420px at 106% 10%, rgba(34,197,94,0.08), transparent 60%),
        radial-gradient(ellipse 600px 500px at 50% 118%, rgba(167,139,250,0.07), transparent 60%),
        radial-gradient(ellipse 500px 380px at 96% 92%, rgba(245,158,11,0.06), transparent 60%),
        radial-gradient(circle at 1px 1px, rgba(255,255,255,0.035) 1px, transparent 0);
    background-size: auto, auto, auto, auto, 26px 26px;
    background-attachment: fixed;
}}
[data-testid="stHeader"] {{ background-color: transparent; }}
#MainMenu, footer, header {{ visibility: hidden; }}
/* Sidebar intentionally hidden — navigation lives in the top bar */
[data-testid="stSidebar"] {{ display: none; }}
[data-testid="collapsedControl"] {{ display: none; }}
.block-container {{ padding-top: 1.2rem !important; max-width: 1240px; position: relative; z-index: 1; }}

/* ── Motion tokens ────────────────────────────────────────────────────── */
@keyframes fadeSlideUp {{ from {{ opacity: 0; transform: translateY(14px); }} to {{ opacity: 1; transform: translateY(0); }} }}
@keyframes fadeIn      {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
@keyframes popIn       {{ 0% {{ opacity: 0; transform: scale(0.93); }} 100% {{ opacity: 1; transform: scale(1); }} }}
@keyframes growBar     {{ from {{ width: 0% !important; }} }}
@keyframes floatOrb    {{ 0%, 100% {{ transform: translate(0,0); }} 50% {{ transform: translate(18px,-14px); }} }}
@keyframes floatShape  {{ 0%, 100% {{ transform: translateY(0) rotate(0deg); }} 50% {{ transform: translateY(-8px) rotate(3deg); }} }}
@keyframes spinLoader  {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
@keyframes softPulse   {{ 0%, 100% {{ box-shadow: 0 0 0 rgba(79,140,255,0); }} 50% {{ box-shadow: 0 0 22px var(--pulse-color, rgba(79,140,255,0.18)); }} }}
@keyframes spinSlow    {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
@keyframes drift       {{ 0%, 100% {{ transform: translate(0,0) rotate(0deg); }} 33% {{ transform: translate(10px,14px) rotate(8deg); }} 66% {{ transform: translate(-8px,6px) rotate(-6deg); }} }}
@keyframes twinkle     {{ 0%, 100% {{ opacity: 0.25; }} 50% {{ opacity: 0.7; }} }}

/* ── Ambient background graphics ─────────────────────────────────────── */
.bg-orb {{ position: fixed; border-radius: 50%; filter: blur(70px); pointer-events: none; z-index: 0; opacity: 0.5; }}
.bg-orb-1 {{ width: 380px; height: 380px; top: -110px; left: -90px; background: radial-gradient(circle, {BLUE}, transparent 70%); animation: floatOrb 16s ease-in-out infinite; }}
.bg-orb-2 {{ width: 340px; height: 340px; top: 45%; right: -120px; background: radial-gradient(circle, {SUCCESS}, transparent 70%); animation: floatOrb 20s ease-in-out infinite reverse; }}
.bg-orb-3 {{ width: 300px; height: 300px; bottom: -100px; left: 30%; background: radial-gradient(circle, {PURPLE}, transparent 70%); animation: floatOrb 24s ease-in-out infinite; }}
.bg-orb-4 {{ width: 220px; height: 220px; top: 6%; left: 46%; background: radial-gradient(circle, {WARNING}, transparent 70%); animation: floatOrb 18s ease-in-out infinite reverse; opacity: 0.28; }}
.bg-orb-5 {{ width: 260px; height: 260px; bottom: 4%; right: 8%; background: radial-gradient(circle, {BLUE}, transparent 70%); animation: floatOrb 22s ease-in-out infinite; opacity: 0.3; }}

/* Small floating geometric accents scattered around the page */
.shape-dot {{ position: fixed; border-radius: 50%; pointer-events: none; z-index: 0; animation: twinkle 4s ease-in-out infinite; }}
.shape-ring {{ position: fixed; border-radius: 50%; border: 1.5px solid; pointer-events: none; z-index: 0; }}
.shape-square {{ position: fixed; pointer-events: none; z-index: 0; animation: drift 12s ease-in-out infinite; }}
.shape-triangle {{ position: fixed; width: 0; height: 0; pointer-events: none; z-index: 0; animation: drift 15s ease-in-out infinite reverse; }}
.shape-plus {{ position: fixed; pointer-events: none; z-index: 0; opacity: 0.35; animation: twinkle 5s ease-in-out infinite; }}
.shape-plus::before, .shape-plus::after {{ content: ''; position: absolute; background: currentColor; }}
.shape-plus::before {{ width: 100%; height: 2px; top: 50%; left: 0; transform: translateY(-50%); }}
.shape-plus::after  {{ width: 2px; height: 100%; left: 50%; top: 0; transform: translateX(-50%); }}
.shape-orbit {{ position: fixed; border-radius: 50%; border: 1px dashed; pointer-events: none; z-index: 0; animation: spinSlow 40s linear infinite; opacity: 0.25; }}

/* ── Top Navigation Bar ───────────────────────────────────────────────── */
.topnav {{
    background: linear-gradient(160deg, rgba(17,24,39,0.72), rgba(13,20,32,0.72)); backdrop-filter: blur(20px) saturate(160%); -webkit-backdrop-filter: blur(20px) saturate(160%);
    border: 1px solid {BORDER}; border-radius: 18px;
    padding: 14px 22px; margin-bottom: 26px;
    display: flex; align-items: center; justify-content: space-between; gap: 20px;
    position: relative; overflow: hidden;
    animation: fadeSlideUp 0.5s ease-out both;
    box-shadow: 0 10px 30px -12px rgba(0,0,0,0.6);
}}
.topnav::before {{
    content: ''; position: absolute; top: -40%; right: -6%; width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(79,140,255,0.16), transparent 70%);
    border-radius: 50%; pointer-events: none;
}}
.topnav::after {{
    content: ''; position: absolute; bottom: -60%; left: 10%; width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(34,197,94,0.10), transparent 70%);
    border-radius: 50%; pointer-events: none;
}}
.topnav-brand {{ display: flex; align-items: center; gap: 12px; position: relative; z-index: 1; }}
.topnav-brand .icon {{
    font-size: 26px; width: 46px; height: 46px; display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, rgba(79,140,255,0.2), rgba(79,140,255,0.05));
    border: 1px solid rgba(79,140,255,0.35); border-radius: 13px;
    animation: softPulse 3s ease-in-out infinite;
}}
.topnav-brand .titles h1 {{ font-size: 16px; font-weight: 700; color: {TEXT_PRI}; margin: 0; letter-spacing: -0.2px; }}
.topnav-brand .titles .team-name {{
    font-family: 'Space Grotesk', sans-serif; font-size: 12px; font-weight: 700; letter-spacing: 0.3px;
    background: linear-gradient(120deg, {BLUE}, {SUCCESS});
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}}
.topnav-badge {{
    display: flex; align-items: center; gap: 8px; background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3); border-radius: 10px; padding: 8px 14px;
    font-size: 12px; color: {DANGER}; font-weight: 600; position: relative; z-index: 1;
    white-space: nowrap;
}}
.topnav-badge.is-success {{ background: rgba(34,197,94,0.1); border-color: rgba(34,197,94,0.3); color: {SUCCESS}; }}
.live-dot {{
    width: 7px; height: 7px; border-radius: 50%; background: currentColor; flex-shrink: 0;
    box-shadow: 0 0 0 rgba(34,197,94,0.5); animation: softPulse 1.8s ease-in-out infinite;
    --pulse-color: rgba(34,197,94,0.5);
}}

[data-testid="stHorizontalBlock"] div[data-testid="stRadio"] div[role="radiogroup"] {{
    flex-direction: row !important; gap: 6px !important; flex-wrap: nowrap;
}}
div[data-testid="stRadio"] div[role="radiogroup"] label {{
    background: rgba(255,255,255,0.02);
    border: 1px solid transparent;
    border-radius: 10px;
    padding: 9px 16px !important;
    color: {TEXT_SEC} !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    transition: all 0.18s ease;
    margin: 0 !important;
}}
div[data-testid="stRadio"] div[role="radiogroup"] label:hover {{
    background: rgba(79,140,255,0.1); color: {TEXT_PRI} !important;
}}
div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {{
    background: linear-gradient(135deg, rgba(79,140,255,0.22), rgba(79,140,255,0.08));
    border: 1px solid rgba(79,140,255,0.45);
}}
div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p {{
    color: {TEXT_PRI} !important; font-weight: 700 !important;
}}
div[data-testid="stRadio"] svg {{ display: none; }}

/* ── Typography ───────────────────────────────────────────────────────── */
.page-header {{
    padding: 20px 24px; border: 1px solid {BORDER}; border-radius: 16px; margin-bottom: 26px;
    background: linear-gradient(160deg, rgba(17,24,39,0.72), rgba(13,20,32,0.72)); backdrop-filter: blur(20px) saturate(160%); -webkit-backdrop-filter: blur(20px) saturate(160%);
    position: relative; overflow: hidden;
    animation: fadeIn 0.5s ease-out both;
}}
.page-header h2 {{ font-size: 26px; font-weight: 800; color: {TEXT_PRI}; margin: 0 0 6px; letter-spacing: -0.5px; position: relative; z-index: 1; }}
.page-header p {{ font-size: 14px; color: {TEXT_SEC}; margin: 0; font-weight: 400; position: relative; z-index: 1; max-width: 70%; }}
.header-shape {{ position: absolute; top: -10px; right: -10px; width: 170px; height: 170px; animation: floatShape 8s ease-in-out infinite; pointer-events: none; }}

.section-label {{
    font-size: 11px; font-weight: 700; color: {TEXT_SEC};
    text-transform: uppercase; letter-spacing: 1.2px; margin: 30px 0 14px;
    display: flex; align-items: center; gap: 8px;
}}
.section-label:first-child {{ margin-top: 0; }}
.section-label::after {{ content: ''; flex: 1; height: 1px; background: {BORDER}; }}

/* ── Generic Cards ────────────────────────────────────────────────────── */
.metric-card {{
    background: {CARD_GLASS}; backdrop-filter: blur(18px) saturate(160%); -webkit-backdrop-filter: blur(18px) saturate(160%); border: 1px solid {BORDER}; border-radius: 16px;
    padding: 22px 22px 20px; position: relative; overflow: hidden;
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    animation: fadeSlideUp 0.55s ease-out both;
}}
.metric-card::before {{
    content: ''; position: absolute; top: 0; left: 0; width: 3px; height: 100%;
    background: var(--accent); border-radius: 16px 0 0 16px;
}}
.metric-card::after {{
    content: ''; position: absolute; top: -30px; right: -30px; width: 90px; height: 90px;
    border-radius: 50%; background: var(--accent); opacity: 0.06; pointer-events: none;
}}
.metric-card .card-deco {{
    position: absolute; bottom: -14px; right: 10px; font-size: 46px; opacity: 0.06;
    pointer-events: none; transform: rotate(-8deg);
}}
.metric-card:hover {{
    transform: translateY(-5px);
    box-shadow: 0 16px 32px -8px rgba(0,0,0,0.6), 0 0 20px -6px var(--accent);
    border-color: var(--accent);
}}
.metric-icon {{ font-size: 20px; margin-bottom: 14px; opacity: 0.9; position: relative; z-index: 1; }}
.metric-value {{ font-size: 30px; font-weight: 800; color: {TEXT_PRI}; letter-spacing: -1px; line-height: 1; margin-bottom: 8px; position: relative; z-index: 1; }}
.metric-label {{ font-size: 12px; color: {TEXT_SEC}; font-weight: 500; position: relative; z-index: 1; }}
.metric-change {{ font-size: 11.5px; margin-top: 10px; font-weight: 600; position: relative; z-index: 1; }}

.insight-card {{
    background: {CARD_GLASS}; backdrop-filter: blur(18px) saturate(160%); -webkit-backdrop-filter: blur(18px) saturate(160%); border: 1px solid {BORDER}; border-radius: 14px;
    padding: 18px 20px; margin: 8px 0; display: flex; gap: 14px; align-items: flex-start;
    transition: transform 0.15s ease, border-color 0.15s ease;
    animation: fadeSlideUp 0.5s ease-out both;
    position: relative; overflow: hidden;
}}
.insight-card:hover {{ transform: translateY(-2px) translateX(3px); border-color: rgba(79,140,255,0.4); }}
.insight-icon {{ font-size: 17px; line-height: 1.4; }}
.insight-text {{ font-size: 13.5px; color: {TEXT_SEC}; line-height: 1.65; }}
.insight-text strong {{ color: {TEXT_PRI}; font-weight: 600; }}

.chart-card {{
    background: {CARD_GLASS}; backdrop-filter: blur(18px) saturate(160%); -webkit-backdrop-filter: blur(18px) saturate(160%); border: 1px solid {BORDER}; border-radius: 16px;
    padding: 20px 20px 8px; margin-bottom: 4px;
    animation: fadeSlideUp 0.55s ease-out both;
    transition: border-color 0.2s ease;
    position: relative; overflow: hidden;
}}
.chart-card::after {{
    content: ''; position: absolute; top: -20px; left: -20px; width: 70px; height: 70px;
    border-radius: 50%; background: {BLUE}; opacity: 0.05; pointer-events: none;
}}
.chart-card:hover {{ border-color: rgba(79,140,255,0.35); }}
.chart-title {{ font-size: 11.5px; font-weight: 700; color: {TEXT_SEC}; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }}

/* ── Prediction Result ────────────────────────────────────────────────── */
.result-card {{
    border-radius: 18px; padding: 30px 30px 26px; text-align: center;
    border: 1px solid var(--rborder); background: var(--rbg);
    animation: popIn 0.5s cubic-bezier(0.34,1.56,0.64,1) both, softPulse 2.6s ease-in-out infinite;
    --pulse-color: var(--rborder); position: relative; overflow: hidden;
}}
.result-card::before {{
    content: ''; position: absolute; top: -40px; right: -40px; width: 140px; height: 140px;
    border-radius: 50%; background: var(--racc); opacity: 0.08; pointer-events: none;
}}
.result-card::after {{
    content: ''; position: absolute; bottom: -50px; left: -30px; width: 120px; height: 120px;
    border-radius: 50%; background: var(--racc); opacity: 0.06; pointer-events: none;
}}
.result-eyebrow {{ font-size: 11.5px; color: {TEXT_SEC}; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 600; }}
.result-pct {{ font-size: 58px; font-weight: 800; letter-spacing: -2.5px; line-height: 1; margin: 10px 0; color: var(--racc); }}
.result-label {{ font-size: 14.5px; font-weight: 600; margin-top: 6px; color: var(--racc); }}
.progress-wrap {{ background: {BORDER}; border-radius: 99px; height: 8px; margin-top: 22px; overflow: hidden; position: relative; z-index: 1; }}
.progress-bar {{ height: 100%; border-radius: 99px; background: var(--rgrad); animation: growBar 0.9s ease-out both; transition: width 0.9s ease; }}

.empty-state {{
    background: {CARD_GLASS}; backdrop-filter: blur(18px) saturate(160%); -webkit-backdrop-filter: blur(18px) saturate(160%); border: 1px dashed {BORDER}; border-radius: 18px;
    padding: 56px 32px; text-align: center; margin-top: 8px;
    animation: fadeIn 0.5s ease-out both;
    position: relative; overflow: hidden;
}}
.empty-state::before {{
    content: ''; position: absolute; top: -30px; left: -30px; width: 100px; height: 100px;
    border-radius: 50%; background: {BLUE}; opacity: 0.05; pointer-events: none;
}}
.empty-state::after {{
    content: ''; position: absolute; bottom: -30px; right: -30px; width: 100px; height: 100px;
    border-radius: 50%; background: {PURPLE}; opacity: 0.05; pointer-events: none;
}}
.empty-state .icon {{ font-size: 42px; margin-bottom: 16px; opacity: 0.7; animation: floatShape 5s ease-in-out infinite; display: inline-block; position: relative; z-index: 1; }}
.empty-state .text {{ font-size: 14px; color: {TEXT_SEC}; line-height: 1.7; position: relative; z-index: 1; }}
.empty-state .text strong {{ color: {TEXT_PRI}; }}

/* ── Retention Cards ──────────────────────────────────────────────────── */
.ret-card {{
    background: {CARD_GLASS}; backdrop-filter: blur(18px) saturate(160%); -webkit-backdrop-filter: blur(18px) saturate(160%); border: 1px solid {BORDER}; border-radius: 14px;
    padding: 18px 20px; margin: 10px 0; border-left: 4px solid var(--racc);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    animation: fadeSlideUp 0.5s ease-out both;
    position: relative; overflow: hidden;
}}
.ret-card::after {{
    content: ''; position: absolute; top: -20px; right: -20px; width: 60px; height: 60px;
    border-radius: 50%; background: var(--racc); opacity: 0.06; pointer-events: none;
}}
.ret-card:hover {{ transform: translateX(4px); box-shadow: -4px 6px 18px -6px rgba(0,0,0,0.5); }}
.ret-title {{ font-size: 14px; font-weight: 700; color: {TEXT_PRI}; margin-bottom: 8px; display: flex; gap: 8px; align-items: center; position: relative; z-index: 1; }}
.ret-tag {{ font-size: 10.5px; font-weight: 700; padding: 3px 10px; border-radius: 99px; flex-shrink: 0; letter-spacing: 0.4px; }}
.ret-desc {{ font-size: 13px; color: {TEXT_SEC}; line-height: 1.7; position: relative; z-index: 1; }}

.risk-score-box {{
    background: {CARD_GLASS}; backdrop-filter: blur(18px) saturate(160%); -webkit-backdrop-filter: blur(18px) saturate(160%); border: 1px solid {BORDER}; border-radius: 16px; padding: 24px 20px; text-align: center;
    animation: popIn 0.5s cubic-bezier(0.34,1.56,0.64,1) both;
    position: relative; overflow: hidden;
}}
.risk-score-box::before {{
    content: ''; position: absolute; top: -40px; left: 50%; transform: translateX(-50%); width: 160px; height: 160px;
    border-radius: 50%; background: radial-gradient(circle, rgba(79,140,255,0.10), transparent 70%); pointer-events: none;
}}
.risk-score-num {{ font-size: 50px; font-weight: 800; letter-spacing: -2px; position: relative; z-index: 1; }}
.risk-label-text {{ font-size: 12.5px; color: {TEXT_SEC}; margin-top: 6px; position: relative; z-index: 1; }}

.score-bar-wrap {{ background: {BORDER}; border-radius: 99px; height: 8px; margin: 4px 0 12px; overflow: hidden; position: relative; z-index: 1; }}
.score-bar {{ height: 100%; border-radius: 99px; animation: growBar 0.8s ease-out both; transition: width 0.8s ease; }}

.summary-bar {{
    background: {CARD_GLASS}; backdrop-filter: blur(18px) saturate(160%); -webkit-backdrop-filter: blur(18px) saturate(160%); border: 1px solid {BORDER}; border-radius: 14px;
    padding: 18px 20px; margin-top: 16px; display: flex; gap: 30px; align-items: center;
    animation: fadeSlideUp 0.5s ease-out both;
}}
.summary-stat {{ text-align: center; }}
.summary-stat .num {{ font-size: 26px; font-weight: 800; }}
.summary-stat .lbl {{ font-size: 10.5px; color: {TEXT_SEC}; text-transform: uppercase; letter-spacing: 0.6px; margin-top: 2px; }}
.summary-note {{ flex: 1; font-size: 13px; color: {TEXT_SEC}; text-align: right; }}

/* ── Loader ───────────────────────────────────────────────────────────── */
.loader-box {{
    background: {CARD_GLASS}; backdrop-filter: blur(18px) saturate(160%); -webkit-backdrop-filter: blur(18px) saturate(160%); border: 1px solid {BORDER}; border-radius: 18px;
    padding: 56px 32px; text-align: center; margin-top: 8px;
    animation: fadeIn 0.3s ease-out both;
}}
.loader-spin {{
    width: 34px; height: 34px; margin: 0 auto 16px; border-radius: 50%;
    border: 3px solid {BORDER}; border-top-color: {BLUE}; border-right-color: {SUCCESS};
    animation: spinLoader 0.8s linear infinite;
}}
.loader-text {{ font-size: 13px; color: {TEXT_SEC}; }}

/* ── Inputs ───────────────────────────────────────────────────────────── */
[data-testid="stSlider"] label, [data-testid="stSelectbox"] label {{
    color: {TEXT_SEC} !important; font-size: 13px !important; font-weight: 500 !important;
}}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{ background-color: {BLUE} !important; }}

.input-section-title {{
    font-size: 12.5px; font-weight: 700; color: {TEXT_PRI};
    margin: 22px 0 12px; display: flex; align-items: center; gap: 8px;
}}
.input-section-title:first-child {{ margin-top: 0; }}

/* ── Buttons ──────────────────────────────────────────────────────────── */
.stButton > button {{
    background: linear-gradient(135deg, {BLUE}, #2f6fe0) !important;
    background-size: 220% 220% !important;
    color: white !important; border: none !important; border-radius: 12px !important;
    padding: 15px !important; font-size: 14.5px !important; font-weight: 600 !important;
    width: 100% !important; box-shadow: 0 8px 20px -6px rgba(79,140,255,0.45) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease, background-position 0.5s ease !important;
    letter-spacing: 0.2px;
}}
.stButton > button:hover {{
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 12px 30px -6px rgba(79,140,255,0.65) !important;
    background-position: 100% 50% !important;
}}
.stButton > button:active {{ transform: translateY(0) scale(0.99) !important; }}

/* ── Focus glow for sliders / selects ─────────────────────────────────── */
[data-baseweb="select"]:focus-within, [data-testid="stSlider"]:focus-within {{
    box-shadow: 0 0 0 3px rgba(79,140,255,0.18); border-radius: 10px;
}}

/* ── Skeleton shimmer (premium loading state) ─────────────────────────── */
@keyframes shimmerMove {{ 0% {{ background-position: -300px 0; }} 100% {{ background-position: 300px 0; }} }}
.skeleton-line {{
    height: 14px; border-radius: 8px; margin: 10px 0;
    background: linear-gradient(90deg, rgba(255,255,255,0.03) 25%, rgba(255,255,255,0.09) 37%, rgba(255,255,255,0.03) 63%);
    background-size: 300px 100%;
    animation: shimmerMove 1.4s ease-in-out infinite;
}}

/* ── Circular gauge (SVG ring) ────────────────────────────────────────── */
.gauge-wrap {{ position: relative; width: 168px; height: 168px; margin: 6px auto 4px; }}
.gauge-wrap svg {{ transform: rotate(-90deg); width: 100%; height: 100%; }}
.gauge-track {{ fill: none; stroke: {BORDER}; stroke-width: 11; }}
.gauge-fill {{
    fill: none; stroke-width: 11; stroke-linecap: round;
    transition: stroke-dashoffset 1.1s cubic-bezier(0.34,1.2,0.4,1);
    animation: gaugeIn 1.1s cubic-bezier(0.34,1.2,0.4,1) both;
    filter: drop-shadow(0 0 6px var(--gaccent, {BLUE}));
}}
@keyframes gaugeIn {{ from {{ stroke-dashoffset: var(--gcirc); }} }}
.gauge-center {{ position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; }}
.gauge-center .val {{ font-size: 30px; font-weight: 800; letter-spacing: -1px; line-height: 1; }}
.gauge-center .tag {{ font-size: 10px; color: {TEXT_SEC}; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }}

/* Stagger reveal per column so cards don't pop all at once */
[data-testid="column"]:nth-child(1) .metric-card {{ animation-delay: 0.02s; }}
[data-testid="column"]:nth-child(2) .metric-card {{ animation-delay: 0.08s; }}
[data-testid="column"]:nth-child(3) .metric-card {{ animation-delay: 0.14s; }}
[data-testid="column"]:nth-child(4) .metric-card {{ animation-delay: 0.2s; }}
</style>

<div class="bg-orb bg-orb-1"></div>
<div class="bg-orb bg-orb-2"></div>
<div class="bg-orb bg-orb-3"></div>
<div class="bg-orb bg-orb-4"></div>
<div class="bg-orb bg-orb-5"></div>

<div class="shape-ring" style="width:120px;height:120px;top:14%;right:6%;border-color:rgba(79,140,255,0.15);"></div>
<div class="shape-ring" style="width:70px;height:70px;bottom:10%;left:4%;border-color:rgba(34,197,94,0.18);"></div>
<div class="shape-ring" style="width:44px;height:44px;top:62%;left:12%;border-color:rgba(167,139,250,0.2);"></div>
<div class="shape-orbit" style="width:260px;height:260px;top:8%;right:-40px;border-color:{BLUE};"></div>
<div class="shape-orbit" style="width:180px;height:180px;bottom:2%;left:-20px;border-color:{PURPLE};"></div>

<div class="shape-dot" style="width:8px;height:8px;top:22%;left:18%;background:rgba(167,139,250,0.4);"></div>
<div class="shape-dot" style="width:6px;height:6px;bottom:20%;right:16%;background:rgba(79,140,255,0.4);"></div>
<div class="shape-dot" style="width:5px;height:5px;top:38%;right:26%;background:rgba(34,197,94,0.45);animation-delay:1s;"></div>
<div class="shape-dot" style="width:7px;height:7px;bottom:34%;left:10%;background:rgba(245,158,11,0.4);animation-delay:2s;"></div>
<div class="shape-dot" style="width:4px;height:4px;top:10%;left:38%;background:rgba(79,140,255,0.5);animation-delay:0.5s;"></div>

<div class="shape-square" style="width:16px;height:16px;top:30%;right:10%;border:1.5px solid rgba(79,140,255,0.25);border-radius:4px;transform:rotate(20deg);"></div>
<div class="shape-square" style="width:12px;height:12px;bottom:28%;right:30%;border:1.5px solid rgba(245,158,11,0.25);border-radius:3px;transform:rotate(-15deg);animation-delay:3s;"></div>
<div class="shape-triangle" style="top:48%;left:6%;border-left:9px solid transparent;border-right:9px solid transparent;border-bottom:14px solid rgba(34,197,94,0.18);"></div>
<div class="shape-triangle" style="bottom:14%;right:20%;border-left:7px solid transparent;border-right:7px solid transparent;border-bottom:11px solid rgba(167,139,250,0.2);animation-delay:2s;"></div>

<div class="shape-plus" style="width:16px;height:16px;top:16%;left:8%;color:rgba(79,140,255,0.4);"></div>
<div class="shape-plus" style="width:12px;height:12px;bottom:8%;right:38%;color:rgba(34,197,94,0.35);animation-delay:1.5s;"></div>
""", unsafe_allow_html=True)


# ── Premium interaction layer (count-up numbers + button ripple) ─────────────
# Runs inside a zero-height component iframe and reaches into the parent
# document (the actual app page) to animate elements Streamlit just rendered.
# A MutationObserver keeps re-scanning as Streamlit swaps HTML in/out so
# newly-appeared KPI cards or buttons pick up the same effects automatically.
def inject_premium_js():
    components.html("""
    <script>
    const doc = window.parent.document;

    function animateCounters() {
        const els = doc.querySelectorAll('.counter-value:not([data-animated])');
        els.forEach(function(el) {
            el.setAttribute('data-animated', '1');
            const target = parseFloat(el.getAttribute('data-target')) || 0;
            const decimals = parseInt(el.getAttribute('data-decimals')) || 0;
            const useComma = el.getAttribute('data-format') === 'comma';
            const duration = 1100;
            const start = performance.now();
            function frame(now) {
                const p = Math.min((now - start) / duration, 1);
                const eased = 1 - Math.pow(1 - p, 3);
                const val = target * eased;
                let out = val.toFixed(decimals);
                if (useComma) out = Number(out).toLocaleString(undefined, {minimumFractionDigits: decimals, maximumFractionDigits: decimals});
                el.textContent = out;
                if (p < 1) requestAnimationFrame(frame);
            }
            requestAnimationFrame(frame);
        });
    }

    function attachRipple() {
        const buttons = doc.querySelectorAll('.stButton > button:not([data-ripple])');
        buttons.forEach(function(btn) {
            btn.setAttribute('data-ripple', '1');
            btn.style.position = 'relative';
            btn.style.overflow = 'hidden';
            btn.addEventListener('click', function(e) {
                const rect = btn.getBoundingClientRect();
                const ripple = doc.createElement('span');
                const size = Math.max(rect.width, rect.height) * 2;
                ripple.style.position = 'absolute';
                ripple.style.left = (e.clientX - rect.left - size / 2) + 'px';
                ripple.style.top = (e.clientY - rect.top - size / 2) + 'px';
                ripple.style.width = ripple.style.height = size + 'px';
                ripple.style.borderRadius = '50%';
                ripple.style.background = 'rgba(255,255,255,0.35)';
                ripple.style.transform = 'scale(0)';
                ripple.style.opacity = '1';
                ripple.style.pointerEvents = 'none';
                ripple.style.transition = 'transform 0.6s ease, opacity 0.6s ease';
                btn.appendChild(ripple);
                requestAnimationFrame(function() {
                    ripple.style.transform = 'scale(1)';
                    ripple.style.opacity = '0';
                });
                setTimeout(function() { ripple.remove(); }, 650);
            });
        });
    }

    function tick() { animateCounters(); attachRipple(); }
    tick();
    const observer = new MutationObserver(function() { tick(); });
    observer.observe(doc.body, {childList: true, subtree: true});
    </script>
    """, height=0)

inject_premium_js()


# ── Load Data & Model ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, "data/processed/clean_data_sample.csv"))

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, "models/lgbm_model.pkl"))

df = load_data()
model = load_model()


# ── UI Helper Functions ────────────────────────────────────────────────────────
def section_label(text):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)

def page_header(title, subtitle, accent=BLUE):
    html = (
        f'<div class="page-header">'
        f'{page_shape_svg(accent)}'
        f'<h2>{title}</h2><p>{subtitle}</p>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def kpi_card(icon, value, label, change, accent, deco=""):
    deco_html = f'<div class="card-deco">{deco}</div>' if deco else ""
    return f"""<div class="metric-card" style="--accent:{accent}">
        {deco_html}
        <div class="metric-icon">{icon}</div>
        <div class="metric-value">{counter_span(value)}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-change" style="color:{accent}">{change}</div>
    </div>"""

def insight_card(icon, html_text):
    return f"""<div class="insight-card">
        <div class="insight-icon">{icon}</div>
        <div class="insight-text">{html_text}</div>
    </div>"""

def chart_card_open(title):
    st.markdown(f'<div class="chart-card"><div class="chart-title">{title}</div>', unsafe_allow_html=True)

def chart_card_close():
    st.markdown('</div>', unsafe_allow_html=True)

def gauge_svg(pct, accent, size=168, stroke=11, tag="RISK SCORE"):
    """Animated circular gradient gauge used for the prediction % and risk score."""
    r = (size - stroke) / 2
    circ = 2 * 3.14159265 * r
    offset = circ * (1 - max(0, min(pct, 100)) / 100)
    cx = cy = size / 2
    grad_id = f"gaugeGrad{abs(hash((accent, pct))) % 100000}"
    return f"""<div class="gauge-wrap" style="--gcirc:{circ:.2f}px;--gaccent:{accent}">
        <svg viewBox="0 0 {size} {size}">
            <defs>
                <linearGradient id="{grad_id}" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="{accent}" stop-opacity="0.55"/>
                    <stop offset="100%" stop-color="{accent}" stop-opacity="1"/>
                </linearGradient>
            </defs>
            <circle class="gauge-track" cx="{cx}" cy="{cy}" r="{r}"></circle>
            <circle class="gauge-fill" cx="{cx}" cy="{cy}" r="{r}"
                stroke="url(#{grad_id})"
                stroke-dasharray="{circ:.2f}" stroke-dashoffset="{offset:.2f}"></circle>
        </svg>
        <div class="gauge-center">
            <div class="val" style="color:{accent}">{pct:.1f}%</div>
            <div class="tag">{tag}</div>
        </div>
    </div>"""

def result_card(pct, high_risk):
    if high_risk:
        rbg = "linear-gradient(160deg, rgba(239,68,68,0.14), rgba(239,68,68,0.03))"
        rborder, racc, rgrad = "rgba(239,68,68,0.35)", DANGER, f"linear-gradient(90deg, {DANGER}, #f87171)"
        label = "⚠️ High Risk — Immediate action needed"
    else:
        rbg = "linear-gradient(160deg, rgba(34,197,94,0.14), rgba(34,197,94,0.03))"
        rborder, racc, rgrad = "rgba(34,197,94,0.35)", SUCCESS, f"linear-gradient(90deg, {SUCCESS}, #4ade80)"
        label = "✅ Low Risk — Customer is stable"
    return f"""<div class="result-card" style="--rbg:{rbg};--rborder:{rborder};--racc:{racc};--rgrad:{rgrad}">
        <div class="result-eyebrow">Churn Probability</div>
        {gauge_svg(pct, racc, tag="CHURN RISK")}
        <div class="result-label">{label}</div>
        <div class="progress-wrap"><div class="progress-bar" style="width:{int(pct)}%"></div></div>
    </div>"""

def ret_card(icon, title, tag_label, tag_bg, tag_color, accent, desc, delay=0):
    return f"""<div class="ret-card" style="--racc:{accent};animation-delay:{delay}s">
        <div class="ret-title">
            <span>{icon}</span><span>{title}</span>
            <span class="ret-tag" style="background:{tag_bg};color:{tag_color}">{tag_label}</span>
        </div>
        <div class="ret-desc">{desc}</div>
    </div>"""

def dark_fig():
    fig, ax = plt.subplots(facecolor=CARD)
    ax.set_facecolor(CARD)
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    ax.tick_params(colors=TEXT_SEC, labelsize=11)
    ax.yaxis.set_tick_params(length=0)
    ax.yaxis.grid(True, color=BORDER, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


# ── Top Navigation Bar (stays fixed at the top of the page, not a sidebar) ─────
nav_l, nav_r, nav_s = st.columns([1, 2.2, 0.7])
with nav_l:
    st.markdown(
        '<div class="topnav-brand" style="margin-top:6px;">'
        '<div class="icon">🔮</div>'
        '<div class="titles">'
        '<h1>Churn Intelligence</h1>'
        '<div class="team-name">🚀 5aliha 3la Allah</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
with nav_r:
    page = st.radio("", PAGES, horizontal=True, label_visibility="collapsed")
with nav_s:
    st.markdown(
        '<div class="topnav-badge is-success" style="margin-top:6px;">'
        '<span class="live-dot"></span> Model Online'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div style="height:22px;"></div>', unsafe_allow_html=True)


# ── PAGE 1: OVERVIEW ──────────────────────────────────────────────────────────
if page == "📊 Overview":
    page_header("Overview", "High-level summary of customer churn across the dataset", accent=BLUE)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("👥", "999,999", "Total Customers",      "",                    BLUE,    "👥"),
        ("📉", "99,227",  "Churned",              "▲ 9.92% churn rate",  DANGER,  "📉"),
        ("✅", "900,772", "Retained",             "▲ 90.08% retention",  SUCCESS, "✅"),
        ("⚠️", "26.5%",   "Month-to-Month Churn", "Highest risk segment", WARNING, "⚠️"),
    ]
    for col, (icon, val, label, change, accent, deco) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(kpi_card(icon, val, label, change, accent, deco), unsafe_allow_html=True)

    section_label("Key Insights")
    i1, i2 = st.columns(2)
    with i1:
        st.markdown(insight_card("📋", "<strong>Contract Type</strong> is the strongest churn predictor. Month-to-Month churn at <strong>26.5%</strong> vs <strong>5.6%</strong> for Two-Year."), unsafe_allow_html=True)
        st.markdown(insight_card("😤", "<strong>Complaints</strong> are a strong early warning signal. Churned customers submit significantly more complaints."), unsafe_allow_html=True)
    with i2:
        st.markdown(insight_card("⏱️", "<strong>New customers</strong> (0–10 months) are at highest risk. Churn drops significantly after the first year."), unsafe_allow_html=True)
        st.markdown(insight_card("💰", "<strong>High monthly charges</strong> correlate with higher churn. Very High segment shows <strong>12%</strong> vs 8% for Medium."), unsafe_allow_html=True)


# ── PAGE 2: EDA ───────────────────────────────────────────────────────────────
elif page == "📈 EDA":
    page_header("Exploratory Analysis", "Visual breakdown of churn patterns across key features", accent=PURPLE)

    c1, c2 = st.columns(2)
    with c1:
        chart_card_open("Churn Distribution")
        fig, ax = dark_fig()
        vals = df['churn'].value_counts()
        ax.bar(['No Churn', 'Churn'], vals.values, color=[BLUE, DANGER], width=0.5, edgecolor='none')
        st.pyplot(fig); plt.close()
        chart_card_close()

    with c2:
        chart_card_open("Churn Rate by Contract Type")
        fig, ax = dark_fig()
        df['contract_label'] = df['contract'].map({0: 'Month-to-Month', 1: 'One Year', 2: 'Two Year'})
        rates = df.groupby('contract_label')['churn'].mean() * 100
        bars = ax.bar(rates.index, rates.values, color=[DANGER, WARNING, SUCCESS], width=0.5, edgecolor='none')
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f'{b.get_height():.1f}%',
                    ha='center', va='bottom', color=TEXT_SEC, fontsize=10, fontweight='600')
        st.pyplot(fig); plt.close()
        chart_card_close()

    c3, c4 = st.columns(2)
    with c3:
        chart_card_open("Avg Complaints by Churn")
        fig, ax = dark_fig()
        comp = df.groupby('churn')['num_complaints'].mean()
        ax.bar(['No Churn', 'Churn'], comp.values, color=[BLUE, DANGER], width=0.5, edgecolor='none')
        st.pyplot(fig); plt.close()
        chart_card_close()

    with c4:
        chart_card_open("Avg Satisfaction by Churn")
        fig, ax = dark_fig()
        sat = df.groupby('churn')['customer_satisfaction'].mean()
        margin = abs(sat.values.max() - sat.values.min()) * 0.5
        bars = ax.bar(['No Churn', 'Churn'], sat.values, color=[BLUE, DANGER], width=0.5, edgecolor='none')
        ax.set_ylim(sat.values.min() - margin, sat.values.max() + margin)
        for b, v in zip(bars, sat.values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + margin * 0.1, f'{v:.2f}',
                    ha='center', va='bottom', color=TEXT_SEC, fontsize=10, fontweight='600')
        st.pyplot(fig); plt.close()
        chart_card_close()


# ── PAGE 3: PREDICTION ────────────────────────────────────────────────────────
elif page == "🤖 Prediction":
    page_header("Churn Prediction", "Enter customer details to predict churn probability", accent=SUCCESS)

    col_in, col_out = st.columns([1.4, 1])
    with col_in:
        st.markdown('<div class="input-section-title">👤 Customer Profile</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age             = st.slider("Age", 18, 90, 35)
            tenure          = st.slider("Tenure (Months)", 0, 72, 12)
            monthly_charges = st.slider("Monthly Charges ($)", 20, 120, 65)
        with c2:
            num_complaints        = st.slider("Complaints", 0, 7, 1)
            customer_satisfaction = st.slider("Satisfaction (1-9)", 1, 9, 5)
            credit_score          = st.slider("Credit Score", 300, 850, 650)

        st.markdown('<div class="input-section-title">📄 Account Details</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            contract     = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
            num_services = st.slider("Number of Services", 0, 8, 3)
        with c4:
            late_payments = st.slider("Late Payments", 0, 10, 0)

        st.markdown("<br>", unsafe_allow_html=True)
        predict = st.button("🔮  Run Prediction", use_container_width=True)

    with col_out:
        section_label("Prediction Result")

        result_placeholder = st.empty()

        if predict:
            result_placeholder.markdown(f"""<div class="loader-box">
                <div class="loader-spin"></div>
                <div class="loader-text">Analyzing customer profile…</div>
                <div style="margin-top:22px;text-align:left;">
                    <div class="skeleton-line" style="width:60%;"></div>
                    <div class="skeleton-line" style="width:85%;"></div>
                    <div class="skeleton-line" style="width:45%;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

            input_data = pd.DataFrame([{col: 0 for col in df.drop(columns=['churn']).columns if col != 'contract_label'}])
            input_data['age']                   = age
            input_data['tenure']                = tenure
            input_data['monthlycharges']        = monthly_charges
            input_data['num_complaints']        = num_complaints
            input_data['customer_satisfaction'] = customer_satisfaction
            input_data['credit_score']          = credit_score
            input_data['contract']              = {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}[contract]
            input_data['num_services']          = num_services
            input_data['late_payments']         = late_payments

            prob = model.predict_proba(input_data)[0][1]
            pct  = prob * 100

            with result_placeholder.container():
                st.markdown(result_card(pct, high_risk=prob > 0.5), unsafe_allow_html=True)

                risks = []
                if num_complaints > 3:           risks.append(("😤", "<strong>High complaint count</strong> — Major churn driver"))
                if contract == "Month-to-Month": risks.append(("📋", "<strong>Month-to-Month contract</strong> — Highest risk segment"))
                if tenure < 12:                  risks.append(("⏱️", "<strong>New customer</strong> — First year is critical"))
                if customer_satisfaction < 4:    risks.append(("😞", "<strong>Low satisfaction</strong> — Needs immediate follow-up"))

                if risks:
                    section_label("Risk Factors")
                    for icon, text in risks:
                        st.markdown(insight_card(icon, text), unsafe_allow_html=True)
        else:
            result_placeholder.markdown(f"""<div class="empty-state">
                <div class="icon">🔮</div>
                <div class="text">Fill in the customer details<br>and click <strong>Run Prediction</strong></div>
            </div>""", unsafe_allow_html=True)


# ── PAGE 4: RETENTION PLAN ────────────────────────────────────────────────────
elif page == "🎯 Retention Plan":
    page_header("Retention Plan", "Analyze customer risk and get a tailored retention strategy", accent=WARNING)

    section_label("Customer Profile")

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

    plan_placeholder = st.empty()

    if analyze_btn:
        with plan_placeholder.container():
            st.markdown(f"""<div class="loader-box">
                <div class="loader-spin"></div>
                <div class="loader-text">Building retention strategy…</div>
                <div style="margin-top:22px;text-align:left;">
                    <div class="skeleton-line" style="width:70%;"></div>
                    <div class="skeleton-line" style="width:50%;"></div>
                    <div class="skeleton-line" style="width:80%;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

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
            risk_level, risk_color, bar_color = "HIGH RISK",   DANGER,  f"linear-gradient(90deg,{DANGER},#f87171)"
        elif risk_score >= 30:
            risk_level, risk_color, bar_color = "MEDIUM RISK", WARNING, f"linear-gradient(90deg,{WARNING},#fbbf24)"
        else:
            risk_level, risk_color, bar_color = "LOW RISK",    SUCCESS, f"linear-gradient(90deg,{SUCCESS},#4ade80)"

        with plan_placeholder.container():
            col_score, col_plan = st.columns([1, 2])

            with col_score:
                st.markdown(f"""<div class="risk-score-box">
                    <div class="section-label" style="margin:0 0 8px;">Churn Probability</div>
                    {gauge_svg(pct, risk_color, tag=risk_level)}
                    <div class="score-bar-wrap" style="margin:14px 0 4px;">
                        <div class="score-bar" style="width:{risk_score}%;background:{bar_color}"></div>
                    </div>
                    <div class="risk-label-text">Based on LightGBM model</div>
                </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                section_label("Risk Factor Scores")

                factors = [
                    ("Complaints",    min(r_complaints / 7, 1),      DANGER),
                    ("Satisfaction",  1 - (r_satisfaction / 9),      WARNING),
                    ("Tenure Risk",   max(0, 1 - r_tenure / 72),     PURPLE),
                    ("Late Payments", min(r_late / 10, 1),           DANGER),
                    ("Contract Risk", [0.9, 0.4, 0.1][contract_val], WARNING),
                ]
                for name, score, color in factors:
                    st.markdown(f"""<div style="margin-bottom:12px;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                            <span style="font-size:12.5px;color:{TEXT_SEC}">{name}</span>
                            <span style="font-size:12.5px;font-weight:700;color:{color}">{int(score*100)}%</span>
                        </div>
                        <div class="score-bar-wrap" style="margin:0;">
                            <div class="score-bar" style="width:{int(score*100)}%;background:{color}"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

            with col_plan:
                section_label("Recommended Retention Actions")

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
                    "urgent": ("URGENT", "rgba(239,68,68,0.15)",  DANGER,  DANGER),
                    "soon":   ("SOON",   "rgba(245,158,11,0.15)", WARNING, WARNING),
                    "normal": ("NORMAL", "rgba(34,197,94,0.15)",  SUCCESS, SUCCESS),
                }

                for idx, (priority, icon, title, desc) in enumerate(actions):
                    tag_label, tag_bg, tag_color, accent = tag_map[priority]
                    st.markdown(ret_card(icon, title, tag_label, tag_bg, tag_color, accent, desc, delay=idx*0.06), unsafe_allow_html=True)

                urgent_count = sum(1 for a in actions if a[0] == "urgent")
                soon_count   = sum(1 for a in actions if a[0] == "soon")
                normal_count = len(actions) - urgent_count - soon_count

                st.markdown(f"""<div class="summary-bar">
                    <div class="summary-stat"><div class="num" style="color:{DANGER}">{urgent_count}</div><div class="lbl">Urgent</div></div>
                    <div class="summary-stat"><div class="num" style="color:{WARNING}">{soon_count}</div><div class="lbl">Soon</div></div>
                    <div class="summary-stat"><div class="num" style="color:{SUCCESS}">{normal_count}</div><div class="lbl">Normal</div></div>
                    <div class="summary-note">{len(actions)} action{"s" if len(actions) != 1 else ""} recommended · Start with URGENT items first</div>
                </div>""", unsafe_allow_html=True)
