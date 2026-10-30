"""
MaintAI — Streamlit Frontend Application
=========================================
Turbofan Jet Engine Remaining Useful Life (RUL) Prognostics:
1. 🏠 Front Page (Project Overview & Executive Summary)
2. 🎛️ Single-Engine Telemetry (Manual Prognostics Console)
3. 📂 Batch Fleet Analysis (CSV Upload & Fleet Health)
4. 📊 Model Benchmarks & Comparative Analytics
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import json
import os

BACKEND_URL = "http://localhost:8000"

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="MaintAI™ | Turbofan Engine Health Prognostics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Theme & Navigation State ────────────────────────────────────────────────
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "🏠 Front Page (Project Overview)"

with st.sidebar:
    st.markdown("### 🎨 Display Theme")
    theme_choice = st.radio(
        "Display Theme",
        ["🌙 Dark Mode", "☀️ Light Mode"],
        horizontal=True,
        key="app_theme_choice",
        label_visibility="collapsed"
    )
    dark_mode = (theme_choice == "🌙 Dark Mode")
    st.markdown("---")

# ─── Dynamic CSS ─────────────────────────────────────────────────────────────
if dark_mode:
    theme_css = """
    .stApp, [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 10% 20%, #0d131f 0%, #080c14 90%) !important;
        color: #f8fafc !important;
    }
    .aero-header {
        background: linear-gradient(135deg, rgba(16, 24, 40, 0.95), rgba(11, 19, 36, 0.9));
        border: 1px solid rgba(56, 189, 248, 0.25);
        box-shadow: 0 10px 30px -10px rgba(2, 132, 199, 0.25);
    }
    .hero-box {
        background: linear-gradient(135deg, rgba(14, 23, 42, 0.95) 0%, rgba(15, 30, 58, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        box-shadow: 0 15px 35px -5px rgba(2, 132, 199, 0.2);
    }
    .feature-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
    }
    .metric-card-pro {
        background: linear-gradient(145deg, #131c2e, #0e1626);
        border: 1px solid rgba(148, 163, 184, 0.15);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
    }
    .section-header {
        color: #38bdf8 !important;
        border-bottom: 1px solid rgba(56, 189, 248, 0.2);
    }
    section[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #0b111e !important;
        border-right: 1px solid rgba(148, 163, 184, 0.12) !important;
    }
    .sidebar-panel {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.12);
    }
    .api-status-online {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34d399;
    }
    .api-status-offline {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #fca5a5;
    }
    p, span, label, div { color: #cbd5e1; }
    """
    chart_template = "plotly_dark"
    gauge_bg = "#0f172a"
    gauge_border = "#1e293b"
    gauge_tick = "#475569"
    gauge_font_color = "#ffffff"
    plot_bg = "rgba(15, 23, 42, 0.6)"
    text_color = "#cbd5e1"
else:
    theme_css = """
    .stApp, [data-testid="stAppViewContainer"] {
        background: #f1f5f9 !important;
        color: #0f172a !important;
    }
    .aero-header {
        background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
        border: 1px solid rgba(2, 132, 199, 0.35);
        box-shadow: 0 8px 24px -6px rgba(2, 132, 199, 0.12);
    }
    .hero-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(2, 132, 199, 0.3);
        box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.1);
    }
    .feature-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
    }
    .metric-card-pro {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
    }
    .section-header {
        color: #0284c7 !important;
        border-bottom: 1px solid rgba(2, 132, 199, 0.25);
    }
    section[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #ffffff !important;
        border-right: 1px solid #cbd5e1 !important;
    }
    .sidebar-panel {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
    }
    .api-status-online {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.5);
        color: #047857;
    }
    .api-status-offline {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.5);
        color: #b91c1c;
    }
    p, span, label, div { color: #1e293b; }
    """
    chart_template = "plotly_white"
    gauge_bg = "#ffffff"
    gauge_border = "#cbd5e1"
    gauge_tick = "#64748b"
    gauge_font_color = "#0f172a"
    plot_bg = "#ffffff"
    text_color = "#0f172a"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}}

.stDeployButton, [data-testid="stToolbar"], #MainMenu, footer {{
    display: none !important;
    visibility: hidden !important;
}}

.block-container {{
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    max-width: 95% !important;
}}

.aero-header {{
    border-radius: 14px;
    padding: 1.3rem 1.6rem;
    margin-bottom: 1.2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}

.hero-box {{
    border-radius: 16px;
    padding: 2.2rem 2.4rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}}

.feature-card {{
    border-radius: 12px;
    padding: 1.2rem 1.3rem;
    margin-bottom: 1rem;
    height: 100%;
    transition: transform 0.2s ease;
}}

.feature-card:hover {{
    transform: translateY(-2px);
}}

.metric-card-pro {{
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    position: relative;
    overflow: hidden;
}}
.metric-card-pro::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}}
.metric-card-pro.green::before {{ background: linear-gradient(90deg, #10b981, #059669); }}
.metric-card-pro.amber::before {{ background: linear-gradient(90deg, #f59e0b, #d97706); }}
.metric-card-pro.red::before   {{ background: linear-gradient(90deg, #ef4444, #dc2626); }}
.metric-card-pro.blue::before  {{ background: linear-gradient(90deg, #38bdf8, #0284c7); }}

.section-header {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 1.1rem;
    font-weight: 700;
    padding-bottom: 0.45rem;
    margin-top: 1.2rem;
    margin-bottom: 1rem;
    letter-spacing: -0.01em;
}}

.sidebar-panel {{
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 1rem;
}}

.api-status-online, .api-status-offline {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.75rem;
    border-radius: 16px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}}

div.stButton > button {{
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
}}

{theme_css}
</style>
""", unsafe_allow_html=True)

# ─── Backend Health Check ─────────────────────────────────────────────────────
@st.cache_data(ttl=15)
def check_backend_health():
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if resp.status_code == 200:
            return True, resp.json()
    except Exception:
        pass
    return False, {}

backend_online, backend_health = check_backend_health()

# ─── Navigation Options ───────────────────────────────────────────────────────
NAV_PAGES = [
    "🏠 Front Page (Project Overview)",
    "🎛️ Single-Engine Telemetry (Manual)",
    "📂 Batch Fleet Analysis (CSV Upload)",
    "📊 Model Benchmarks & Comparison",
]

with st.sidebar:
    st.markdown("### 🧭 Navigation")
    
    # Sync with session state
    try:
        current_idx = NAV_PAGES.index(st.session_state.nav_page)
    except ValueError:
        current_idx = 0

    selected_nav = st.radio(
        "Navigation",
        NAV_PAGES,
        index=current_idx,
        key="sidebar_nav_radio",
        label_visibility="collapsed"
    )
    if selected_nav != st.session_state.nav_page:
        st.session_state.nav_page = selected_nav
        st.rerun()

    st.markdown("---")
    st.markdown("### 🧠 Active Model")
    selected_model_option = st.selectbox(
        "Inference Architecture",
        [
            "⚡ XGBoost Regressor (Tree Ensemble)",
            "🧬 PyTorch LSTM (Deep Recurrent Neural Net)"
        ],
        index=0,
        key="global_model_choice"
    )
    chosen_model_type = "lstm" if "lstm" in selected_model_option.lower() else "xgboost"

    st.markdown("---")

    # Service Links
    st.markdown("### 🔗 Microservices")
    st.markdown(
        f"""
        <div style="display:flex; flex-direction:column; gap:0.5rem; font-size:0.82rem;">
            <a href="http://localhost:8000/landing" target="_blank" style="color:#38bdf8; font-weight:600; text-decoration:none;">🌐 Standalone Landing Page ↗</a>
            <a href="http://localhost:8000/docs" target="_blank" style="color:#38bdf8; font-weight:600; text-decoration:none;">📖 FastAPI Swagger UI ↗</a>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Backend status
    if backend_online:
        models_avail = ", ".join(backend_health.get("models_available", ["XGBoost", "LSTM"]))
        st.markdown(
            f'<div class="api-status-online">🟢 Backend Connected ({models_avail})</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="api-status-offline">🔴 Backend Offline (Heuristic active)</div>',
            unsafe_allow_html=True
        )

# ─── Telemetry Metadata & Presets ─────────────────────────────────────────────
SENSOR_META = {
    'Cycle'           : ('Flight Cycle Age', 'cycles', 'Total operational cycles completed'),
    'SensorMeasure2'  : ('LPC Outlet Temp', '°R', 'Low-pressure compressor outlet total temperature'),
    'SensorMeasure3'  : ('HPC Outlet Temp', '°R', 'High-pressure compressor outlet total temperature'),
    'SensorMeasure4'  : ('LPT Outlet Temp', '°R', 'Low-pressure turbine outlet total temperature'),
    'SensorMeasure7'  : ('HPC Static Pressure', 'psia', 'High-pressure compressor discharge static pressure'),
    'SensorMeasure8'  : ('Fuel Flow Ratio', 'pps/psia', 'Ratio of fuel flow to static pressure'),
    'SensorMeasure11' : ('HPC Bleed Enthalpy', 'Btu/lbm', 'Enthalpy of air extracted from high-pressure compressor'),
    'SensorMeasure12' : ('LP Turbine Coolant', 'lbm/s', 'Low-pressure turbine coolant bleed airflow rate'),
    'SensorMeasure13' : ('HP Turbine Coolant', 'lbm/s', 'High-pressure turbine coolant bleed airflow rate'),
    'SensorMeasure15' : ('Bypass Ratio', 'ratio', 'Engine bypass mass flow ratio'),
    'SensorMeasure17' : ('Bleed Enthalpy', 'Btu/lbm', 'Corrected turbine bleed enthalpy'),
    'SensorMeasure20' : ('HPC Efficiency', 'ratio', 'High-pressure compressor isentropic efficiency'),
    'SensorMeasure21' : ('LPT Efficiency', 'ratio', 'Low-pressure turbine isentropic efficiency'),
}

FEATURES = ['Cycle'] + [f'SensorMeasure{i}' for i in [2, 3, 4, 7, 8, 11, 12, 13, 15, 17, 20, 21]]

BASELINE_HEALTHY = {
    'Cycle': 1,
    'SensorMeasure2': 641.82, 'SensorMeasure3': 1589.70, 'SensorMeasure4': 1400.60,
    'SensorMeasure7': 554.36, 'SensorMeasure8': 2388.02, 'SensorMeasure11': 47.47,
    'SensorMeasure12': 521.66, 'SensorMeasure13': 2388.02, 'SensorMeasure15': 8.4195,
    'SensorMeasure17': 392.0, 'SensorMeasure20': 39.06, 'SensorMeasure21': 23.419
}

PRESETS = {
    "healthy": {
        'label': '🟢 Nominal / Factory Benchmark (Cycle 1)',
        'Cycle': 1,
        'SensorMeasure2' : 641.82,  'SensorMeasure3' : 1589.70,
        'SensorMeasure4' : 1400.60, 'SensorMeasure7' : 554.36,
        'SensorMeasure8' : 2388.02, 'SensorMeasure11': 47.47,
        'SensorMeasure12': 521.66,  'SensorMeasure13': 2388.02,
        'SensorMeasure15': 8.4195,  'SensorMeasure17': 392.0,
        'SensorMeasure20': 39.06,   'SensorMeasure21': 23.419,
    },
    "warning": {
        'label': '🟡 Degradation Phase / Fleet Mid-Life (Cycle 100)',
        'Cycle': 100,
        'SensorMeasure2' : 643.75,  'SensorMeasure3' : 1598.40,
        'SensorMeasure4' : 1418.20, 'SensorMeasure7' : 556.10,
        'SensorMeasure8' : 2388.10, 'SensorMeasure11': 47.10,
        'SensorMeasure12': 520.50,  'SensorMeasure13': 2388.10,
        'SensorMeasure15': 8.4600,  'SensorMeasure17': 394.0,
        'SensorMeasure20': 38.75,   'SensorMeasure21': 23.15,
    },
    "critical": {
        'label': '🔴 Severe Wear / End-of-Life (Cycle 185)',
        'Cycle': 185,
        'SensorMeasure2' : 646.90,  'SensorMeasure3' : 1614.50,
        'SensorMeasure4' : 1447.60, 'SensorMeasure7' : 558.80,
        'SensorMeasure8' : 2388.30, 'SensorMeasure11': 46.50,
        'SensorMeasure12': 518.20,  'SensorMeasure13': 2388.30,
        'SensorMeasure15': 8.5800,  'SensorMeasure17': 398.5,
        'SensorMeasure20': 38.20,   'SensorMeasure21': 22.70,
    },
}

def rul_status_local(rul):
    if rul > 80:
        return 'NOMINAL OPERATION', 'green', '🟢', 'Standard routine monitoring; flight dispatch permitted with no restrictions.'
    elif rul > 30:
        return 'MAINTENANCE ADVISORY', 'amber', '🟡', 'Subsystem wear detected. Schedule non-destructive inspection (NDI) within 15–20 cycles.'
    else:
        return 'CRITICAL EXHAUSTION (AOG)', 'red', '🔴', 'Aircraft On Ground protocol. Immediate overhaul / engine change required before next mission.'

def api_predict_single(sensor_data: dict, model_type: str = "xgboost"):
    m_param = "lstm" if "lstm" in model_type.lower() else "xgboost"
    resp = requests.post(
        f"{BACKEND_URL}/predict",
        params={"model": m_param},
        json={k: (int(v) if k == 'Cycle' else float(v)) for k, v in sensor_data.items()},
        timeout=8,
    )
    resp.raise_for_status()
    d = resp.json()
    status_map = {'green': ('🟢', 'green'), 'amber': ('🟡', 'amber'), 'red': ('🔴', 'red')}
    icon, theme = status_map.get(d['status_color'], ('⚪', 'blue'))
    model_name = d.get('model_used', 'XGBoost Regressor' if m_param == 'xgboost' else 'LSTM Network (PyTorch)')
    return d['rul'], d['status'], theme, icon, d['action_directive'], d['health_pct'], model_name

def api_predict_batch(rows: list, model_type: str = "xgboost"):
    m_param = "lstm" if "lstm" in model_type.lower() else "xgboost"
    payload = [{k: (int(v) if k == 'Cycle' else float(v)) for k, v in row.items()} for row in rows]
    resp = requests.post(
        f"{BACKEND_URL}/predict/batch",
        params={"model": m_param},
        json={"readings": payload},
        timeout=45
    )
    resp.raise_for_status()
    return resp.json()["predictions"]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: 🏠 FRONT PAGE (LANDING PAGE / PROJECT OVERVIEW)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.nav_page == "🏠 Front Page (Project Overview)":

    # Hero Cover
    st.markdown("""
    <div class="hero-box">
        <div style="display:inline-block; padding:0.25rem 0.8rem; background:rgba(56,189,248,0.15); border:1px solid rgba(56,189,248,0.4); border-radius:20px; font-size:0.75rem; font-weight:700; letter-spacing:0.05em; color:#38bdf8; text-transform:uppercase; margin-bottom:0.8rem;">
            AeroPredict RUL™ · Prognostics &amp; Health Management System
        </div>
        <h1 style="font-size:2.4rem; font-weight:900; margin:0 0 0.5rem 0; letter-spacing:-0.03em;">
            ✈️ MaintAI: Turbofan Jet Engine Prognostics
        </h1>
        <p style="font-size:1.15rem; font-weight:500; margin:0; color:#94a3b8; max-width:850px;">
            Predictive Maintenance &amp; Remaining Useful Life (RUL) Estimation for Commercial Aviation Turbofan Engines using Machine Learning and High-Frequency Sensor Telemetry.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Call to Action Buttons
    cta1, cta2, cta3, cta4 = st.columns(4)
    with cta1:
        if st.button("🚀 Launch Live Diagnostics Console", type="primary", use_container_width=True):
            st.session_state.nav_page = "🎛️ Single-Engine Telemetry (Manual)"
            st.rerun()
    with cta2:
        if st.button("📂 Fleet Batch CSV Analytics", use_container_width=True):
            st.session_state.nav_page = "📂 Batch Fleet Analysis (CSV Upload)"
            st.rerun()
    with cta3:
        if st.button("📊 View Model Benchmarks", use_container_width=True):
            st.session_state.nav_page = "📊 Model Benchmarks & Comparison"
            st.rerun()
    with cta4:
        st.markdown(
            f'<a href="http://localhost:8000/landing" target="_blank" style="display:block; text-align:center; padding:0.5rem; background:#0284c7; color:#fff; border-radius:8px; font-weight:600; text-decoration:none;">🌐 Standalone Web Landing ↗</a>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-header">⚡ Core Project Performance Metrics</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
        <div class="metric-card-pro blue">
            <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#94a3b8;">Benchmark Dataset</div>
            <div style="font-size:2rem; font-weight:800; font-family:'JetBrains Mono'; margin:0.3rem 0;">4 Datasets</div>
            <div style="font-size:0.8rem; color:#64748b;">NASA C-MAPSS (FD001–FD004)</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="metric-card-pro green">
            <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#94a3b8;">Top Model Accuracy</div>
            <div style="font-size:2rem; font-weight:800; font-family:'JetBrains Mono'; margin:0.3rem 0;">13.8 RMSE</div>
            <div style="font-size:0.8rem; color:#64748b;">LSTM Network (13.8 cyc) · XGBoost (17.6 cyc)</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="metric-card-pro amber">
            <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#94a3b8;">Sensor Streams</div>
            <div style="font-size:2rem; font-weight:800; font-family:'JetBrains Mono'; margin:0.3rem 0;">13 Sensors</div>
            <div style="font-size:0.8rem; color:#64748b;">Thermodynamic & Flow Channels Isolated</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="metric-card-pro blue">
            <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#94a3b8;">Inference Speed</div>
            <div style="font-size:2rem; font-weight:800; font-family:'JetBrains Mono'; margin:0.3rem 0;">&lt; 15 ms</div>
            <div style="font-size:0.8rem; color:#64748b;">FastAPI Microservice Response Time</div>
        </div>
        """, unsafe_allow_html=True)

    # Problem Statement & System Architecture
    st.markdown('<div class="section-header">🛠️ System Architecture & Workflow Pipeline</div>', unsafe_allow_html=True)

    w1, w2, w3, w4 = st.columns(4)
    with w1:
        st.markdown("""
        <div class="feature-card">
            <h4 style="margin:0 0 0.5rem 0; color:#38bdf8;">1. Data Acquisition</h4>
            <p style="font-size:0.85rem; line-height:1.5; margin:0;">
                Ingestion of 21 sensor channels across operational flight cycles from the NASA Prognostics Center of Excellence.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with w2:
        st.markdown("""
        <div class="feature-card">
            <h4 style="margin:0 0 0.5rem 0; color:#38bdf8;">2. Signal Preprocessing</h4>
            <p style="font-size:0.85rem; line-height:1.5; margin:0;">
                Removed 8 flatline sensors with zero variance. Capped RUL labels at 125 cycles with piecewise linear modeling.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with w3:
        st.markdown("""
        <div class="feature-card">
            <h4 style="margin:0 0 0.5rem 0; color:#38bdf8;">3. Machine Learning</h4>
            <p style="font-size:0.85rem; line-height:1.5; margin:0;">
                Gradient-boosted decision trees (XGBoost) trained with early stopping, subsampling, and tree depth optimization.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with w4:
        st.markdown("""
        <div class="feature-card">
            <h4 style="margin:0 0 0.5rem 0; color:#38bdf8;">4. Fleet Prognostics</h4>
            <p style="font-size:0.85rem; line-height:1.5; margin:0;">
                Automated classification into Nominal (&gt;80), Advisory (30–80), and Mandatory Grounding (AOG &lt;30) alerts.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Subsystems & Critical Operating Ranges
    st.markdown('<div class="section-header">🔍 Monitored Turbofan Subsystems</div>', unsafe_allow_html=True)

    sub1, sub2 = st.columns(2)
    with sub1:
        st.markdown("""
        <div class="feature-card">
            <h4 style="margin:0 0 0.6rem 0; color:#38bdf8;">Compressor Subsystems (Cold Section)</h4>
            <ul style="font-size:0.85rem; line-height:1.7; padding-left:1.2rem; margin:0;">
                <li><strong>LPC (Low Pressure Compressor):</strong> Monitors outlet temperature (SM2) and initial air compression efficiency.</li>
                <li><strong>HPC (High Pressure Compressor):</strong> Tracks HPC outlet temperature (SM3), discharge static pressure (SM7), and bleed enthalpy (SM11).</li>
                <li><strong>Degradation Signatures:</strong> Tip clearance widening causes lower pressure ratios and elevated discharge temperatures.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with sub2:
        st.markdown("""
        <div class="feature-card">
            <h4 style="margin:0 0 0.6rem 0; color:#38bdf8;">Turbine Subsystems (Hot Section)</h4>
            <ul style="font-size:0.85rem; line-height:1.7; padding-left:1.2rem; margin:0;">
                <li><strong>HPT (High Pressure Turbine):</strong> Subject to extreme combustor exit heat; coolant bleed mass flow rate (SM13) tracks thermal degradation.</li>
                <li><strong>LPT (Low Pressure Turbine):</strong> Monitors exhaust outlet temperature (SM4) and coolant bleed flow (SM12).</li>
                <li><strong>Degradation Signatures:</strong> Thermal fatigue and blade erosion cause coolant flow increases and exhaust gas temperature (EGT) surges.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: 🎛️ SINGLE-ENGINE TELEMETRY DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.nav_page == "🎛️ Single-Engine Telemetry (Manual)":

    if 'sensor_vals' not in st.session_state:
        st.session_state.sensor_vals = {k: v for k, v in PRESETS['healthy'].items() if k != 'label'}

    st.markdown('<div class="section-header">⚡ Rapid Diagnostic Telemetry Scenarios</div>', unsafe_allow_html=True)

    p_col1, p_col2, p_col3, p_col4 = st.columns([1, 1, 1, 1.2])
    with p_col1:
        if st.button("🟢 Nominal Engine\n(Cycle 1)", use_container_width=True):
            for k, v in PRESETS['healthy'].items():
                if k != 'label':
                    st.session_state.sensor_vals[k] = v
            st.rerun()
    with p_col2:
        if st.button("🟡 Advisory Wear\n(Cycle 100)", use_container_width=True):
            for k, v in PRESETS['warning'].items():
                if k != 'label':
                    st.session_state.sensor_vals[k] = v
            st.rerun()
    with p_col3:
        if st.button("🔴 Critical Exhaustion\n(Cycle 185)", use_container_width=True):
            for k, v in PRESETS['critical'].items():
                if k != 'label':
                    st.session_state.sensor_vals[k] = v
            st.rerun()
    with p_col4:
        st.markdown("""
        <div style="font-size:0.75rem; color:#94a3b8; padding:0.4rem; border:1px solid rgba(148,163,184,0.2); border-radius:8px;">
            <strong>Prognostic Rule:</strong> Temperatures & coolant demands rise as rotating blade seals deteriorate.
        </div>
        """, unsafe_allow_html=True)

    # Telemetry Input Form
    st.markdown('<div class="section-header">🎛️ Real-Time Telemetry Data Input</div>', unsafe_allow_html=True)

    with st.form("aeropredict_form"):
        tab1, tab2, tab3, tab4 = st.tabs([
            "🌡️ Thermodynamic Temperatures",
            "💨 Pressures & Fluid Dynamics",
            "⚙️ Coolant Flows & Enthalpy",
            "📈 Subsystem Efficiencies & Cycle"
        ])

        sensor_inputs = {}

        with tab1:
            st.caption("Gas path total temperature telemetry captured across compressor and turbine exit stations.")
            t_cols = st.columns(3)
            for idx, feat in enumerate(['SensorMeasure2', 'SensorMeasure3', 'SensorMeasure4']):
                label, unit, desc = SENSOR_META.get(feat, (feat, '-', ''))
                val = float(st.session_state.sensor_vals.get(feat, 0.0))
                with t_cols[idx]:
                    sensor_inputs[feat] = st.number_input(
                        f"{label} ({unit})", value=val, format="%.4f", help=f"{feat}: {desc}", key=f"in_{feat}"
                    )

        with tab2:
            st.caption("Aerodynamic static pressure and bypass airflow ratio metrics.")
            p_cols = st.columns(3)
            for idx, feat in enumerate(['SensorMeasure7', 'SensorMeasure8', 'SensorMeasure15']):
                label, unit, desc = SENSOR_META.get(feat, (feat, '-', ''))
                val = float(st.session_state.sensor_vals.get(feat, 0.0))
                with p_cols[idx]:
                    sensor_inputs[feat] = st.number_input(
                        f"{label} ({unit})", value=val, format="%.4f", help=f"{feat}: {desc}", key=f"in_{feat}"
                    )

        with tab3:
            st.caption("Coolant bleed mass flow and air enthalpy across high and low pressure turbines.")
            c_cols = st.columns(4)
            for idx, feat in enumerate(['SensorMeasure11', 'SensorMeasure12', 'SensorMeasure13', 'SensorMeasure17']):
                label, unit, desc = SENSOR_META.get(feat, (feat, '-', ''))
                val = float(st.session_state.sensor_vals.get(feat, 0.0))
                with c_cols[idx]:
                    sensor_inputs[feat] = st.number_input(
                        f"{label} ({unit})", value=val, format="%.4f", help=f"{feat}: {desc}", key=f"in_{feat}"
                    )

        with tab4:
            st.caption("Cumulative mission cycles and calculated isentropic component efficiencies.")
            e_cols = st.columns(3)
            for idx, feat in enumerate(['Cycle', 'SensorMeasure20', 'SensorMeasure21']):
                label, unit, desc = SENSOR_META.get(feat, (feat, '-', ''))
                val = float(st.session_state.sensor_vals.get(feat, 0.0))
                with e_cols[idx]:
                    if feat == 'Cycle':
                        sensor_inputs[feat] = st.number_input(
                            f"{label} ({unit})", min_value=1, max_value=600, value=int(val), step=1,
                            help=f"{feat}: {desc}", key=f"in_{feat}"
                        )
                    else:
                        sensor_inputs[feat] = st.number_input(
                            f"{label} ({unit})", value=val, format="%.4f", help=f"{feat}: {desc}", key=f"in_{feat}"
                        )

        predict_trigger = st.form_submit_button("🔮 Compute RUL Prognostics", type="primary", use_container_width=True)

    current_data = sensor_inputs if predict_trigger else st.session_state.sensor_vals

    # Prediction execution
    try:
        if backend_online:
            rul_val, status_title, status_theme, status_icon, action_directive, life_pct, active_model_name = api_predict_single(current_data, model_type=chosen_model_type)
        else:
            raise ConnectionError("Backend offline")
    except Exception:
        cycle_cnt = current_data.get('Cycle', 50)
        rul_val = max(0.0, 125.0 - cycle_cnt * 0.65)
        status_title, status_theme, status_icon, action_directive = rul_status_local(rul_val)
        life_pct = (rul_val / 125.0) * 100.0
        active_model_name = "Heuristic Baseline (Offline)"

    st.markdown('<div class="section-header">📊 Diagnostic Prognosis Results</div>', unsafe_allow_html=True)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="metric-card-pro {status_theme}">
            <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase; color:#94a3b8;">Predicted Remaining Useful Life</div>
            <div style="font-size:2.1rem; font-weight:800; font-family:'JetBrains Mono'; margin:0.3rem 0;">{rul_val:.0f} <span style="font-size:1rem; color:#64748b;">cycles</span></div>
            <div style="font-size:0.8rem; color:#64748b;">Model: {active_model_name}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="metric-card-pro {'green' if life_pct>60 else 'amber' if life_pct>25 else 'red'}">
            <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase; color:#94a3b8;">Health Index</div>
            <div style="font-size:2.1rem; font-weight:800; font-family:'JetBrains Mono'; margin:0.3rem 0;">{life_pct:.1f}%</div>
            <div style="font-size:0.8rem; color:#64748b;">Relative to 125-cycle baseline</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="metric-card-pro {status_theme}">
            <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase; color:#94a3b8;">Airworthiness Classification</div>
            <div style="font-size:1.4rem; font-weight:800; margin:0.5rem 0;">{status_icon} {status_title.split()[0]}</div>
            <div style="font-size:0.8rem; color:#64748b;">{status_title}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="metric-card-pro blue">
            <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase; color:#94a3b8;">Cycle Completed</div>
            <div style="font-size:2.1rem; font-weight:800; font-family:'JetBrains Mono'; margin:0.3rem 0;">{int(current_data.get('Cycle', 1))} <span style="font-size:1rem; color:#64748b;">cyc</span></div>
            <div style="font-size:0.8rem; color:#64748b;">Logged operational age</div>
        </div>
        """, unsafe_allow_html=True)

    # Visualizations: Gauge & Sensor Deviations
    v_col1, v_col2 = st.columns([1.1, 1.2])
    with v_col1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=rul_val,
            number={'suffix': " cyc", 'font': {'size': 26, 'color': gauge_font_color, 'family': 'JetBrains Mono'}},
            delta={'reference': 125, 'decreasing': {'color': '#ef4444'}, 'valueformat': ".0f"},
            title={'text': "Prognostic RUL Gauge", 'font': {'size': 14, 'color': text_color}},
            gauge={
                'axis': {'range': [0, 125], 'tickwidth': 1, 'tickcolor': gauge_tick},
                'bar': {'color': '#0284c7' if not dark_mode else '#38bdf8', 'thickness': 0.28},
                'bgcolor': gauge_bg,
                'borderwidth': 1,
                'bordercolor': gauge_border,
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(239, 68, 68, 0.45)'},
                    {'range': [30, 80], 'color': 'rgba(245, 158, 11, 0.45)'},
                    {'range': [80, 125], 'color': 'rgba(16, 185, 129, 0.45)'},
                ],
                'threshold': {'line': {'color': gauge_font_color, 'width': 3}, 'thickness': 0.75, 'value': rul_val}
            }
        ))
        fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': text_color}, height=270, margin=dict(l=20, r=20, t=35, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with v_col2:
        deviations = []
        for feat in ['SensorMeasure2', 'SensorMeasure3', 'SensorMeasure4', 'SensorMeasure7',
                     'SensorMeasure8', 'SensorMeasure11', 'SensorMeasure15', 'SensorMeasure20', 'SensorMeasure21']:
            base = BASELINE_HEALTHY.get(feat, 1.0)
            curr = current_data.get(feat, base)
            pct_diff = ((curr - base) / base) * 100.0
            deviations.append({
                'Sensor': feat.replace('SensorMeasure', 'SM'),
                'Deviation (%)': round(pct_diff, 2),
                'Direction': 'Elevated' if pct_diff > 0 else 'Depressed'
            })
        dev_df = pd.DataFrame(deviations)

        fig_dev = px.bar(
            dev_df, x='Sensor', y='Deviation (%)', color='Direction',
            color_discrete_map={'Elevated': '#ef4444', 'Depressed': '#0284c7' if not dark_mode else '#38bdf8'},
            title="Sensor Drift vs. Nominal Benchmark (%)", template=chart_template
        )
        fig_dev.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=plot_bg, font={'color': text_color, 'size': 11}, height=270, margin=dict(l=15, r=15, t=35, b=15))
        st.plotly_chart(fig_dev, use_container_width=True)

    if status_theme == 'green':
        st.success(f"**{status_title}** — {action_directive}")
    elif status_theme == 'amber':
        st.warning(f"**{status_title}** — {action_directive}")
    else:
        st.error(f"**{status_title}** — {action_directive}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: 📂 BATCH FLEET SENSOR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.nav_page == "📂 Batch Fleet Analysis (CSV Upload)":

    st.markdown('<div class="section-header">📂 Fleet Batch Sensor Analytics</div>', unsafe_allow_html=True)
    st.markdown("""
    Upload multi-engine operational telemetry logs (`.csv`) containing the 13 standard C-MAPSS channels.
    The engine will generate fleet degradation curves, failure risk distribution, and prioritized overhaul rankings.
    """)

    uploaded_file = st.file_uploader("Upload Fleet Telemetry CSV File", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.markdown(f"**Telemetry File Loaded:** `{len(batch_df):,}` rows × `{len(batch_df.columns)}` channels")

        missing_channels = [col for col in FEATURES if col not in batch_df.columns]
        if missing_channels:
            st.error(f"❌ Missing required telemetry channels: `{', '.join(missing_channels)}`")
        else:
            X_batch = batch_df[FEATURES].copy()

            try:
                if backend_online:
                    rows_payload = X_batch.to_dict(orient='records')
                    api_results  = api_predict_batch(rows_payload, model_type=chosen_model_type)
                    preds = [r['rul'] for r in api_results]
                    batch_df['Predicted_RUL']     = np.clip(preds, 0, 125)
                    batch_df['Prognostic_Status']  = [r['status'].split()[0] for r in api_results]
                else:
                    raise ConnectionError("Backend offline")
            except Exception:
                preds = np.maximum(0, 125 - X_batch['Cycle'].values * 0.65)
                batch_df['Predicted_RUL']    = np.clip(preds, 0, 125)
                batch_df['Prognostic_Status'] = batch_df['Predicted_RUL'].apply(
                    lambda r: 'NOMINAL' if r > 80 else ('ADVISORY' if r > 30 else 'CRITICAL')
                )

            st.markdown('<div class="section-header">📈 Fleet Prognostics Overview</div>', unsafe_allow_html=True)

            b1, b2, b3, b4 = st.columns(4)
            unit_id_col = 'ID' if 'ID' in batch_df.columns else ('unit_number' if 'unit_number' in batch_df.columns else None)
            total_engines = len(batch_df[unit_id_col].unique()) if unit_id_col else len(batch_df)

            b1.metric("Monitored Engines", f"{total_engines}")
            b2.metric("Fleet Average RUL", f"{batch_df['Predicted_RUL'].mean():.1f} cyc")
            b3.metric("Minimum Fleet RUL", f"{batch_df['Predicted_RUL'].min():.1f} cyc")
            b4.metric("At-Risk (AOG/Critical)", int((batch_df['Predicted_RUL'] < 30).sum()))

            fc1, fc2 = st.columns([1.3, 1])
            with fc1:
                fig_traj = px.line(
                    batch_df.reset_index(),
                    x='Cycle' if 'Cycle' in batch_df.columns else batch_df.index,
                    y='Predicted_RUL',
                    color=unit_id_col,
                    title='Engine RUL Degradation Trajectories Across Cycles',
                    template=chart_template
                )
                fig_traj.add_hline(y=30, line_dash='dash', line_color='#ef4444', annotation_text='AOG Critical (<30 cyc)')
                fig_traj.add_hline(y=80, line_dash='dash', line_color='#f59e0b', annotation_text='Advisory (<80 cyc)')
                fig_traj.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=plot_bg, font={'color': text_color})
                st.plotly_chart(fig_traj, use_container_width=True)

            with fc2:
                status_summary = batch_df['Prognostic_Status'].value_counts().reset_index()
                status_summary.columns = ['Status', 'Count']
                fig_donut = px.pie(
                    status_summary, names='Status', values='Count', hole=0.45,
                    title='Fleet Airworthiness Distribution',
                    color='Status',
                    color_discrete_map={'NOMINAL': '#10b981', 'ADVISORY': '#f59e0b', 'CRITICAL': '#ef4444'},
                    template=chart_template
                )
                fig_donut.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': text_color})
                st.plotly_chart(fig_donut, use_container_width=True)

            csv_export = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Fleet Airworthiness Diagnostic Report (CSV)",
                data=csv_export,
                file_name="fleet_prognostics_report.csv",
                mime="text/csv",
                use_container_width=True
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: 📊 MODEL BENCHMARKS & COMPARATIVE ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.nav_page == "📊 Model Benchmarks & Comparison":

    st.markdown('<div class="section-header">📊 Machine Learning Model Comparative Evaluation</div>', unsafe_allow_html=True)

    # Metrics from actual evaluation on NASA C-MAPSS FD001 official test set (100 engines)
    benchmark_data = [
        {"Model": "Linear Regression", "Test RMSE": 22.19, "Test MAE": 17.37, "Test R²": 0.6935, "Type": "Baseline Classical"},
        {"Model": "Random Forest", "Test RMSE": 19.76, "Test MAE": 14.95, "Test R²": 0.7568, "Type": "Ensemble Bagging"},
        {"Model": "XGBoost (Tuned - Final)", "Test RMSE": 18.91, "Test MAE": 14.35, "Test R²": 0.7773, "Type": "Gradient Boosted Trees"},
        {"Model": "LSTM Network (PyTorch)", "Test RMSE": 13.23, "Test MAE": 9.45, "Test R²": 0.8910, "Type": "Sequential Recurrent DL"},
    ]
    bench_df = pd.DataFrame(benchmark_data)

    c1, c2 = st.columns(2)
    with c1:
        fig_rmse = px.bar(
            bench_df, x="Model", y="Test RMSE", color="Model",
            title="Model Test Root Mean Squared Error (Lower is Better)",
            template=chart_template, text="Test RMSE"
        )
        fig_rmse.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=plot_bg)
        st.plotly_chart(fig_rmse, use_container_width=True)

    with c2:
        fig_r2 = px.bar(
            bench_df, x="Model", y="Test R²", color="Model",
            title="Model Coefficient of Determination R² (Higher is Better)",
            template=chart_template, text="Test R²"
        )
        fig_r2.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=plot_bg)
        st.plotly_chart(fig_r2, use_container_width=True)

    st.markdown("### 📋 Detailed Benchmark Summary Table")
    st.dataframe(bench_df, use_container_width=True)

    # Show LSTM plots if available
    lstm_loss_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'lstm_loss_curve.png')
    lstm_pred_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'lstm_test_prediction.png')
    if os.path.exists(lstm_loss_path) and os.path.exists(lstm_pred_path):
        st.markdown('<div class="section-header">📈 LSTM Deep Learning Deliverables (Week 5 & 6)</div>', unsafe_allow_html=True)
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(lstm_loss_path, caption="Week 5 Deliverable: Training vs. Validation Loss Curve across 25 Epochs")
        with img_col2:
            st.image(lstm_pred_path, caption="Week 6 Deliverable: Ground Truth vs. LSTM Predicted RUL Across Test Engines")

    st.markdown("""
    <div class="sidebar-panel">
        <h4 style="margin:0 0 0.5rem 0; color:#38bdf8;">Production Model Strategy: XGBoost &amp; PyTorch LSTM</h4>
        <ul style="font-size:0.85rem; line-height:1.6; padding-left:1.2rem; margin:0;">
            <li><strong>XGBoost Regressor (Production Default):</strong> Optimized gradient boosted trees achieving sub-15ms inference latency on CPU, ideal for single-snapshot telemetry scoring and edge microservices.</li>
            <li><strong>PyTorch LSTM (Deep Sequence Model):</strong> 2-Layer recurrent neural network with 30-cycle temporal sliding windows, capturing non-linear dynamic degradation signatures across time and achieving Test RMSE of 13.23 cycles.</li>
            <li><strong>Hybrid Diagnostic Architecture:</strong> Allows fleet managers to toggle between fast snapshot scoring (XGBoost) and deep multi-cycle temporal analysis (LSTM) directly within the dashboard.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
