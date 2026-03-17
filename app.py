import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PeopleIQ · Attrition Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  GLOBAL CSS INJECTION
# ─────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">

<style>
/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

[data-testid="stAppViewContainer"] {
    background: #0a0d14;
    font-family: 'DM Sans', sans-serif;
    color: #e2e8f0;
}

[data-testid="stHeader"] { background: transparent; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f1420 !important;
    border-right: 1px solid #1e2535;
}

[data-testid="stSidebar"] * { color: #94a3b8 !important; }

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label {
    color: #64748b !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'Syne', sans-serif !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] {
    background: #1a2030 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 8px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0d14; }
::-webkit-scrollbar-thumb { background: #1e2d45; border-radius: 4px; }

/* ── Divider ── */
hr { border-color: #1e2535 !important; margin: 2rem 0 !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2535 !important;
    border-radius: 12px !important;
    overflow: hidden;
}

iframe { border-radius: 12px !important; }

/* ── Progress bar ── */
[data-testid="stProgress"] > div {
    background: #1e2535 !important;
    border-radius: 99px !important;
    height: 8px !important;
}

[data-testid="stProgress"] > div > div {
    border-radius: 99px !important;
    background: linear-gradient(90deg, #f97316, #ef4444) !important;
}

/* ── Alert boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: none !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #0f1420;
    border: 1px solid #1e2535;
    border-radius: 14px;
    padding: 1.2rem 1.4rem !important;
    transition: border-color 0.2s;
}

[data-testid="stMetric"]:hover { border-color: #2d4a6e; }

[data-testid="stMetricLabel"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #475569 !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: #f1f5f9 !important;
}

/* ── Selectbox ── */
[data-baseweb="select"] {
    background: #0f1420 !important;
    border: 1px solid #1e2535 !important;
    border-radius: 10px !important;
}

[data-baseweb="select"] * { color: #cbd5e1 !important; }

/* ── Multiselect tags ── */
[data-baseweb="tag"] {
    background: #1e3a5f !important;
    border-radius: 6px !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #3b82f6 !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0f1420 !important;
    border: 1px solid #1e2535 !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] summary {
    color: #94a3b8 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em;
}

/* ── Spinner ── */
[data-testid="stSpinner"] * { color: #3b82f6 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD MODEL & DATA
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load('lgbm_model.pkl')
    features = joblib.load('feature_columns.pkl')
    return model, features

@st.cache_data
def load_data():
    df            = pd.read_csv('employees.csv')
    df_active     = df[df['Termd'] == 0].copy()
    df_terminated = df[df['Termd'] == 1].copy()
    return df, df_active, df_terminated

model, feature_columns = load_model()
df_raw, df_active, df_terminated = load_data()

# ─────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────
def compute_seniority(df):
    df = df.copy()
    today = datetime.today()
    df['DateofHire'] = pd.to_datetime(df['DateofHire'])
    df['Seniority']  = (today - df['DateofHire']).dt.days / 365.25
    return df

def prepare_features(df):
    df = compute_seniority(df)

    string_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in string_cols:
        df[col] = df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)

    perf_map = {'PIP': 1, 'Needs Improvement': 2, 'Fully Meets': 3, 'Exceeds': 4}
    df['PerformanceScore'] = df['PerformanceScore'].map(perf_map)
    df['PerformanceScore'] = pd.to_numeric(df['PerformanceScore'], errors='coerce').fillna(3)

    cat_cols    = ['Position', 'Department', 'ManagerName', 'RecruitmentSource']
    df_encoded  = pd.get_dummies(df, columns=cat_cols)

    drop_cols   = ['EmpID', 'MarriedID', 'GenderID', 'DeptID', 'PositionID',
                   'ManagerID', 'MaritalStatusID', 'EmpStatusID',
                   'DateofHire', 'Employee_Name', 'DateofTermination',
                   'TermReason', 'EmploymentStatus']
    df_encoded  = df_encoded.drop(columns=[c for c in drop_cols if c in df_encoded.columns])
    df_encoded  = df_encoded.reindex(columns=feature_columns, fill_value=0)

    for col in df_encoded.columns:
        df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce').fillna(0)

    return df_encoded

def get_predictions(df):
    X = prepare_features(df)
    return model.predict_proba(X)[:, 1]

def risk_badge(level):
    colors = {
        '🔴 High':   ('#ff4d4d', '#2a0f0f'),
        '🟡 Medium': ('#f59e0b', '#2a1f00'),
        '🟢 Low':    ('#22c55e', '#0a2010'),
    }
    fg, bg = colors.get(level, ('#94a3b8', '#1e2535'))
    label  = level.split(' ', 1)[1]  # strip emoji
    return f'<span style="background:{bg};color:{fg};padding:3px 10px;border-radius:99px;font-size:0.75rem;font-weight:600;border:1px solid {fg}30;font-family:Syne,sans-serif;letter-spacing:0.05em">{label}</span>'

# ─────────────────────────────────────────────
#  RUN PREDICTIONS
# ─────────────────────────────────────────────
with st.spinner("Running inference…"):
    df_active = compute_seniority(df_active)
    df_active['Termination_Probability'] = get_predictions(df_active)
    df_active['Risk_Level'] = pd.cut(
        df_active['Termination_Probability'],
        bins=[0, 0.08, 0.25, 1.0],
        labels=['🟢 Low', '🟡 Medium', '🔴 High']
    )

n_high   = len(df_active[df_active['Risk_Level'] == '🔴 High'])
n_medium = len(df_active[df_active['Risk_Level'] == '🟡 Medium'])
n_low    = len(df_active[df_active['Risk_Level'] == '🟢 Low'])
n_total  = len(df_active)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0.5rem 0 1.5rem">
        <p style="font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;
                  color:#f1f5f9;margin:0;letter-spacing:-0.02em">🧠 PeopleIQ</p>
        <p style="font-size:0.72rem;color:#334155;margin:0.2rem 0 0;
                  letter-spacing:0.08em;text-transform:uppercase">Attrition Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-family:Syne,sans-serif;font-size:0.65rem;color:#334155;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem">Risk Filter</p>', unsafe_allow_html=True)
    risk_filter = st.multiselect(
        "Risk Level", ['🟢 Low', '🟡 Medium', '🔴 High'],
        default=['🟢 Low', '🟡 Medium', '🔴 High'],
        label_visibility="collapsed"
    )

    st.markdown('<p style="font-family:Syne,sans-serif;font-size:0.65rem;color:#334155;text-transform:uppercase;letter-spacing:0.1em;margin-top:1rem;margin-bottom:0.4rem">Department</p>', unsafe_allow_html=True)
    dept_filter = st.multiselect(
        "Department", sorted(df_active['Department'].str.strip().unique()),
        default=sorted(df_active['Department'].str.strip().unique()),
        label_visibility="collapsed"
    )

    st.markdown('<p style="font-family:Syne,sans-serif;font-size:0.65rem;color:#334155;text-transform:uppercase;letter-spacing:0.1em;margin-top:1rem;margin-bottom:0.4rem">Top N At-Risk</p>', unsafe_allow_html=True)
    top_n = st.slider("Top N", 5, 50, 10, label_visibility="collapsed")

    st.markdown("---")

    # Risk donut-style summary
    pct_high = round(n_high / n_total * 100, 1) if n_total else 0
    st.markdown(f"""
    <div style="background:#0a0d14;border:1px solid #1e2535;border-radius:12px;padding:1rem 1.2rem">
        <p style="font-family:Syne,sans-serif;font-size:0.65rem;color:#334155;
                  text-transform:uppercase;letter-spacing:0.1em;margin:0 0 0.8rem">Fleet Overview</p>
        <div style="display:flex;flex-direction:column;gap:0.5rem">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="color:#ff4d4d;font-size:0.8rem">● High Risk</span>
                <span style="font-family:Syne,sans-serif;font-weight:700;color:#f1f5f9">{n_high}</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="color:#f59e0b;font-size:0.8rem">● Medium Risk</span>
                <span style="font-family:Syne,sans-serif;font-weight:700;color:#f1f5f9">{n_medium}</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="color:#22c55e;font-size:0.8rem">● Low Risk</span>
                <span style="font-family:Syne,sans-serif;font-weight:700;color:#f1f5f9">{n_low}</span>
            </div>
        </div>
        <div style="margin-top:1rem;height:4px;background:#1e2535;border-radius:99px;overflow:hidden">
            <div style="height:100%;width:{pct_high}%;background:linear-gradient(90deg,#f97316,#ef4444);border-radius:99px"></div>
        </div>
        <p style="font-size:0.7rem;color:#475569;margin:0.4rem 0 0">{pct_high}% workforce at high risk</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:auto;padding-top:2rem">
        <p style="font-size:0.65rem;color:#1e2535;text-align:center">
            Model · LightGBM · CV AUC 0.930 ± 0.015
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div style="padding:2rem 0 1rem">
    <p style="font-family:Syne,sans-serif;font-size:2.4rem;font-weight:800;
              color:#f1f5f9;margin:0;letter-spacing:-0.03em;line-height:1.1">
        Attrition Risk Dashboard
    </p>
    <p style="color:#475569;font-size:0.95rem;margin:0.4rem 0 0">
        Powered by LightGBM · {n_total} active employees · Last run {datetime.today().strftime('%b %d, %Y')}
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  KPI CARDS
# ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Active Employees", n_total)
with c2:
    st.metric("🔴 High Risk", n_high, delta=f"{round(n_high/n_total*100,1)}% of workforce", delta_color="inverse")
with c3:
    st.metric("🟡 Medium Risk", n_medium)
with c4:
    st.metric("🟢 Low Risk", n_low)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TOP AT-RISK TABLE
# ─────────────────────────────────────────────
st.markdown(f"""
<p style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;
          color:#f1f5f9;letter-spacing:-0.01em;margin-bottom:0.8rem">
    ⚠️ Top {top_n} Employees at Highest Risk
</p>
""", unsafe_allow_html=True)

df_filtered = df_active[
    df_active['Risk_Level'].isin(risk_filter) &
    df_active['Department'].str.strip().isin(dept_filter)
].sort_values('Termination_Probability', ascending=False)

df_table = df_filtered[['Employee_Name','Department','Position',
                         'Seniority','Termination_Probability','Risk_Level']].head(top_n).copy()

# Build styled HTML table
def prob_bar(p):
    color = '#ef4444' if p >= 0.25 else '#f59e0b' if p >= 0.08 else '#22c55e'
    return f"""
    <div style="display:flex;align-items:center;gap:8px">
        <div style="flex:1;background:#1e2535;border-radius:99px;height:6px;overflow:hidden">
            <div style="width:{p*100:.1f}%;height:100%;background:{color};border-radius:99px"></div>
        </div>
        <span style="font-family:Syne,sans-serif;font-weight:600;color:#f1f5f9;min-width:40px;font-size:0.85rem">{p*100:.1f}%</span>
    </div>"""

rows_html = ""
for _, row in df_table.iterrows():
    name  = str(row['Employee_Name']).strip()
    dept  = str(row['Department']).strip()
    pos   = str(row['Position']).strip()
    sen   = f"{row['Seniority']:.1f} yrs"
    prob  = row['Termination_Probability']
    badge = risk_badge(str(row['Risk_Level']))
    bar   = prob_bar(prob)
    rows_html += f"""
    <tr style="border-bottom:1px solid #0f1420;transition:background 0.15s" 
        onmouseover="this.style.background='#111827'" 
        onmouseout="this.style.background='transparent'">
        <td style="padding:0.85rem 1rem;font-weight:500;color:#e2e8f0">{name}</td>
        <td style="padding:0.85rem 1rem;color:#64748b;font-size:0.85rem">{dept}</td>
        <td style="padding:0.85rem 1rem;color:#64748b;font-size:0.85rem">{pos}</td>
        <td style="padding:0.85rem 1rem;color:#94a3b8;font-size:0.85rem;text-align:center">{sen}</td>
        <td style="padding:0.85rem 1.2rem;min-width:180px">{bar}</td>
        <td style="padding:0.85rem 1rem;text-align:center">{badge}</td>
    </tr>"""

st.markdown(f"""
<div style="background:#0f1420;border:1px solid #1e2535;border-radius:14px;overflow:hidden">
    <table style="width:100%;border-collapse:collapse;font-family:'DM Sans',sans-serif">
        <thead>
            <tr style="border-bottom:1px solid #1e2535;background:#0a0d14">
                <th style="padding:0.7rem 1rem;text-align:left;font-family:Syne,sans-serif;
                           font-size:0.65rem;color:#334155;text-transform:uppercase;letter-spacing:0.1em;font-weight:600">Employee</th>
                <th style="padding:0.7rem 1rem;text-align:left;font-family:Syne,sans-serif;
                           font-size:0.65rem;color:#334155;text-transform:uppercase;letter-spacing:0.1em;font-weight:600">Department</th>
                <th style="padding:0.7rem 1rem;text-align:left;font-family:Syne,sans-serif;
                           font-size:0.65rem;color:#334155;text-transform:uppercase;letter-spacing:0.1em;font-weight:600">Position</th>
                <th style="padding:0.7rem 1rem;text-align:center;font-family:Syne,sans-serif;
                           font-size:0.65rem;color:#334155;text-transform:uppercase;letter-spacing:0.1em;font-weight:600">Seniority</th>
                <th style="padding:0.7rem 1rem;text-align:left;font-family:Syne,sans-serif;
                           font-size:0.65rem;color:#334155;text-transform:uppercase;letter-spacing:0.1em;font-weight:600">Risk Score</th>
                <th style="padding:0.7rem 1rem;text-align:center;font-family:Syne,sans-serif;
                           font-size:0.65rem;color:#334155;text-transform:uppercase;letter-spacing:0.1em;font-weight:600">Level</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────
#  INDIVIDUAL LOOKUP
# ─────────────────────────────────────────────
st.markdown("""
<p style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;
          color:#f1f5f9;letter-spacing:-0.01em;margin-bottom:0.8rem">
    🔎 Individual Employee Profile
</p>
""", unsafe_allow_html=True)

employee_name = st.selectbox(
    "Select employee",
    options=sorted(df_active['Employee_Name'].str.strip().unique()),
    label_visibility="collapsed"
)

emp_row = df_active[df_active['Employee_Name'].str.strip() == employee_name].iloc[0]
prob    = float(emp_row['Termination_Probability'])
risk    = str(emp_row['Risk_Level'])

risk_color = '#ef4444' if '🔴' in risk else '#f59e0b' if '🟡' in risk else '#22c55e'
risk_bg    = '#2a0f0f'  if '🔴' in risk else '#2a1f00'  if '🟡' in risk else '#0a2010'
risk_label = risk.split(' ', 1)[1]

# Profile card
perf_raw = emp_row.get('PerformanceScore', 'N/A')
rec_src  = str(emp_row.get('RecruitmentSource', 'N/A')).strip()
absences = emp_row.get('Absences', 'N/A')
projects = emp_row.get('SpecialProjectsCount', 'N/A')

profile_html = (
    '<div style="background:#0f1420;border:1px solid #1e2535;border-radius:16px;padding:1.8rem 2rem;margin-bottom:1rem">'

    '<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1.5rem">'
        '<div>'
            f'<p style="font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;color:#f1f5f9;margin:0;letter-spacing:-0.02em">{employee_name.strip()}</p>'
            f'<p style="color:#475569;font-size:0.85rem;margin:0.2rem 0 0">{str(emp_row.get("Position","N/A")).strip()} &middot; {str(emp_row.get("Department","N/A")).strip()}</p>'
        '</div>'
        f'<div style="background:{risk_bg};border:1px solid {risk_color}40;border-radius:12px;padding:0.6rem 1.2rem;text-align:center">'
            f'<p style="font-family:Syne,sans-serif;font-size:1.6rem;font-weight:800;color:{risk_color};margin:0">{prob*100:.1f}%</p>'
            f'<p style="font-size:0.7rem;color:{risk_color}99;margin:0;letter-spacing:0.08em;text-transform:uppercase;font-family:Syne,sans-serif">{risk_label} Risk</p>'
        '</div>'
    '</div>'

    '<div style="margin-bottom:1.5rem">'
        '<div style="display:flex;justify-content:space-between;margin-bottom:0.4rem">'
            '<span style="font-size:0.72rem;color:#334155;text-transform:uppercase;letter-spacing:0.08em;font-family:Syne,sans-serif">Termination Probability</span>'
            '<span style="font-size:0.72rem;color:#334155">0% &mdash;&mdash;&mdash;&mdash;&mdash;&mdash;&mdash; 100%</span>'
        '</div>'
        '<div style="background:#1e2535;border-radius:99px;height:10px;overflow:hidden">'
            f'<div style="width:{prob*100:.1f}%;height:100%;background:linear-gradient(90deg,{risk_color}88,{risk_color});border-radius:99px"></div>'
        '</div>'
    '</div>'

    '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem">'
        '<div style="background:#0a0d14;border:1px solid #1e2535;border-radius:10px;padding:0.8rem 1rem">'
            '<p style="font-size:0.65rem;color:#334155;text-transform:uppercase;letter-spacing:0.1em;font-family:Syne,sans-serif;margin:0 0 0.3rem">Seniority</p>'
            f'<p style="font-family:Syne,sans-serif;font-size:1.2rem;font-weight:700;color:#f1f5f9;margin:0">{emp_row["Seniority"]:.1f} <span style="font-size:0.75rem;color:#475569">yrs</span></p>'
        '</div>'
        '<div style="background:#0a0d14;border:1px solid #1e2535;border-radius:10px;padding:0.8rem 1rem">'
            '<p style="font-size:0.65rem;color:#334155;text-transform:uppercase;letter-spacing:0.1em;font-family:Syne,sans-serif;margin:0 0 0.3rem">Absences</p>'
            f'<p style="font-family:Syne,sans-serif;font-size:1.2rem;font-weight:700;color:#f1f5f9;margin:0">{absences}</p>'
        '</div>'
        '<div style="background:#0a0d14;border:1px solid #1e2535;border-radius:10px;padding:0.8rem 1rem">'
            '<p style="font-size:0.65rem;color:#334155;text-transform:uppercase;letter-spacing:0.1em;font-family:Syne,sans-serif;margin:0 0 0.3rem">Projects</p>'
            f'<p style="font-family:Syne,sans-serif;font-size:1.2rem;font-weight:700;color:#f1f5f9;margin:0">{projects}</p>'
        '</div>'
        '<div style="background:#0a0d14;border:1px solid #1e2535;border-radius:10px;padding:0.8rem 1rem">'
            '<p style="font-size:0.65rem;color:#334155;text-transform:uppercase;letter-spacing:0.1em;font-family:Syne,sans-serif;margin:0 0 0.3rem">Source</p>'
            f'<p style="font-family:Syne,sans-serif;font-size:0.85rem;font-weight:600;color:#94a3b8;margin:0">{rec_src}</p>'
        '</div>'
    '</div>'

    '</div>'
)
st.markdown(profile_html, unsafe_allow_html=True)

# Recommendation box
if prob >= 0.25:
    st.markdown(f"""
    <div style="background:#2a0f0f;border:1px solid #ef444440;border-radius:12px;padding:1rem 1.4rem;
                display:flex;align-items:flex-start;gap:1rem">
        <span style="font-size:1.4rem">🚨</span>
        <div>
            <p style="font-family:Syne,sans-serif;font-weight:700;color:#ef4444;margin:0 0 0.2rem;font-size:0.9rem">
                Immediate Retention Action Required</p>
            <p style="color:#94a3b8;font-size:0.82rem;margin:0">
                This employee shows high attrition risk. Schedule a 1:1 with their manager, 
                review compensation benchmarks, and consider additional project assignments to improve engagement.
            </p>
        </div>
    </div>""", unsafe_allow_html=True)
elif prob >= 0.08:
    st.markdown(f"""
    <div style="background:#2a1f00;border:1px solid #f59e0b40;border-radius:12px;padding:1rem 1.4rem;
                display:flex;align-items:flex-start;gap:1rem">
        <span style="font-size:1.4rem">⚠️</span>
        <div>
            <p style="font-family:Syne,sans-serif;font-weight:700;color:#f59e0b;margin:0 0 0.2rem;font-size:0.9rem">
                Monitor & Engage</p>
            <p style="color:#94a3b8;font-size:0.82rem;margin:0">
                This employee shows moderate risk signals. Maintain regular check-ins 
                and ensure workload and recognition are balanced.
            </p>
        </div>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div style="background:#0a2010;border:1px solid #22c55e40;border-radius:12px;padding:1rem 1.4rem;
                display:flex;align-items:flex-start;gap:1rem">
        <span style="font-size:1.4rem">✅</span>
        <div>
            <p style="font-family:Syne,sans-serif;font-weight:700;color:#22c55e;margin:0 0 0.2rem;font-size:0.9rem">
                Employee Appears Stable</p>
            <p style="color:#94a3b8;font-size:0.82rem;margin:0">
                Low attrition risk detected. Continue standard engagement practices 
                and periodic performance reviews.
            </p>
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  EXPLAINABILITY BUTTON
# ─────────────────────────────────────────────

# ── Feature → human sentence templates ──────
# Each entry: (condition_fn, risk_sentence, protective_sentence)
# condition_fn receives the raw feature value
FEATURE_TEMPLATES = {
    'Seniority': {
        'label': 'Seniority',
        'unit': 'years',
        'risk':       lambda v: (f"Low seniority ({v:.1f} yrs) is the strongest risk signal — new employees leave at a much higher rate." if v < 5 else f"Seniority ({v:.1f} yrs) is unexpectedly contributing to risk (possible career stagnation)."),
        'protective': lambda v: (f"High seniority ({v:.1f} yrs) is a strong stabilising factor — long-tenured employees rarely leave." if v >= 5 else f"Despite low seniority ({v:.1f} yrs), turnover risk is lowered here."),
    },
    'SpecialProjectsCount': {
        'label': 'Special Projects',
        'unit': 'projects',
        'risk':       lambda v: (f"Only {int(v)} special project(s) assigned — little project involvement often correlates with disengagement." if v <= 1 else f"{int(v)} special project(s) assigned — high workload might be contributing to burnout risk."),
        'protective': lambda v: (f"{int(v)} special project(s) assigned — active project involvement is a strong retention signal." if v > 1 else f"Having {int(v)} special project(s) is acting as a protective factor for this employee."),
    },
    'Salary': {
        'label': 'Salary',
        'unit': '$',
        'risk':       lambda v: (f"Salary of ${int(v):,} is below average — compensation may be a contributing risk factor." if v < 70000 else f"Salary of ${int(v):,} is high, yet evaluated as a risk factor (possible mismatch with market or expectations)."),
        'protective': lambda v: (f"Salary of ${int(v):,} is competitive — compensation helps retain this employee." if v >= 60000 else f"Salary of ${int(v):,} is acting as a protective factor in this context."),
    },
    'EngagementSurvey': {
        'label': 'Engagement Score',
        'unit': '/5',
        'risk':       lambda v: (f"Engagement survey score of {v:.2f}/5 is low — disengaged employees are significantly more likely to leave." if v < 3.5 else f"Engagement survey score of {v:.2f}/5 is unexpectedly acting as a risk factor here."),
        'protective': lambda v: (f"High engagement score of {v:.2f}/5 — this employee is actively invested in their role." if v >= 3.5 else f"Engagement survey score of {v:.2f}/5 is curiously evaluated as a protective factor here."),
    },
    'EmpSatisfaction': {
        'label': 'Job Satisfaction',
        'unit': '/5',
        'risk':       lambda v: (f"Job satisfaction score of {int(v)}/5 is below average — dissatisfaction is a leading predictor of turnover." if v <= 3 else f"Job satisfaction score of {int(v)}/5 is unexpectedly contributing to turnover risk for this profile."),
        'protective': lambda v: (f"Job satisfaction score of {int(v)}/5 is strong — the employee feels valued." if v >= 4 else f"Job satisfaction score of {int(v)}/5 is acting as a protective retention factor."),
    },
    'Absences': {
        'label': 'Absences',
        'unit': 'days',
        'risk':       lambda v: (f"{int(v)} absence days recorded — elevated absenteeism is strongly correlated with disengagement." if v >= 10 else f"{int(v)} absence days is contributing to attrition risk."),
        'protective': lambda v: (f"Only {int(v)} absence day(s) — regular attendance reflects stable commitment." if v < 10 else f"{int(v)} absence days is surprisingly acting as a protective factor."),
    },
    'DaysLateLast30': {
        'label': 'Days Late (Last 30)',
        'unit': 'days',
        'risk':       lambda v: (f"{int(v)} late arrival(s) in the last 30 days — repeated lateness often signals declining commitment." if v > 0 else f"{int(v)} late arrivals is adding to risk."),
        'protective': lambda v: (f"No late arrivals in the last 30 days — punctuality reflects engagement." if v == 0 else f"{int(v)} late arrivals is evaluated as a protective factor."),
    },
    'PerformanceScore': {
        'label': 'Performance Score',
        'unit': '',
        'risk':       lambda v: (f"Performance score of {int(v)}/4 indicates underperformance — low performers are at elevated risk." if v <= 2 else f"Strong performance score of {int(v)}/4 may indicate they are seeking external opportunities."),
        'protective': lambda v: (f"Performance score of {int(v)}/4 — strong performance is a key retention anchor." if v >= 3 else f"Performance score of {int(v)}/4 acts to decrease turnover risk in this context."),
    },
}

# One-hot feature groups
OHE_TEMPLATES = {
    'RecruitmentSource_Diversity Job Fair': {
        'risk':       "Recruited via a Diversity Job Fair — historically, this source shows higher early attrition rates in this dataset.",
        'protective': None,
    },
    'RecruitmentSource_LinkedIn': {
        'risk':       None,
        'protective': "Recruited via LinkedIn — this source is associated with higher retention rates.",
    },
    'RecruitmentSource_Employee Referral': {
        'risk':       None,
        'protective': "Recruited via Employee Referral — referral hires are among the most stable employees.",
    },
    'Department_Production': {
        'risk':       "Works in the Production department — this department shows the highest historical attrition rate.",
        'protective': None,
    },
    'Department_Software Engineering': {
        'risk':       None,
        'protective': "Works in Software Engineering — this department shows strong retention figures.",
    },
}

@st.cache_resource
def get_shap_explainer(_model, _X_background):
    import shap
    return shap.TreeExplainer(_model)

def explain_employee(emp_name):
    import shap

    # Get the feature row for this employee
    idx      = df_active[df_active['Employee_Name'].str.strip() == emp_name].index[0]
    X_emp    = prepare_features(df_active.loc[[idx]])

    explainer   = get_shap_explainer(model, X_emp)
    shap_values = explainer.shap_values(X_emp)

    # For binary classifiers, shap_values may be a list [class0, class1] or a single array
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    feature_names = X_emp.columns.tolist()
    shap_series   = pd.Series(sv, index=feature_names).sort_values(key=abs, ascending=False)

    # Get raw values for this employee (before encoding)
    raw = df_active.loc[idx]

    risk_factors       = []
    protective_factors = []
    used               = set()

    for feat, shap_val in shap_series.items():
        if len(risk_factors) >= 3 and len(protective_factors) >= 3:
            break
        if abs(shap_val) < 0.05:
            continue

        # Numeric features
        if feat in FEATURE_TEMPLATES and feat not in used:
            tmpl    = FEATURE_TEMPLATES[feat]
            raw_val = raw.get(feat, None)
            if raw_val is None:
                continue
            
            if feat == 'PerformanceScore' and isinstance(raw_val, str):
                perf_map = {'PIP': 1, 'Needs Improvement': 2, 'Fully Meets': 3, 'Exceeds': 4}
                raw_val = perf_map.get(raw_val.strip(), 3)
                
            raw_val = float(raw_val)
            used.add(feat)
            if shap_val > 0 and len(risk_factors) < 3:
                risk_factors.append((feat, shap_val, tmpl['risk'](raw_val)))
            elif shap_val < 0 and len(protective_factors) < 3:
                protective_factors.append((feat, shap_val, tmpl['protective'](raw_val)))

        # One-hot features
        elif feat in OHE_TEMPLATES and feat not in used:
            tmpl = OHE_TEMPLATES[feat]
            used.add(feat)
            if shap_val > 0 and tmpl['risk'] and len(risk_factors) < 3:
                risk_factors.append((feat, shap_val, tmpl['risk']))
            elif shap_val < 0 and tmpl['protective'] and len(protective_factors) < 3:
                protective_factors.append((feat, shap_val, tmpl['protective']))

    return risk_factors, protective_factors, shap_series

# ── Button + reveal ─────────────────────────
if st.button("🔍 Explain this prediction", use_container_width=True):
    with st.spinner("Computing SHAP explanations…"):
        try:
            risk_factors, protective_factors, shap_series = explain_employee(employee_name)

            # ── Opening verdict sentence ────────────
            if prob >= 0.25:
                verdict = f"The model assigns <b>{employee_name.strip()}</b> a <span style='color:#ef4444;font-weight:700'>{prob*100:.1f}% termination risk</span> — classified as <b>High Risk</b>. The following factors drive this prediction:"
            elif prob >= 0.08:
                verdict = f"The model assigns <b>{employee_name.strip()}</b> a <span style='color:#f59e0b;font-weight:700'>{prob*100:.1f}% termination risk</span> — classified as <b>Medium Risk</b>. Several mixed signals were detected:"
            else:
                verdict = f"The model assigns <b>{employee_name.strip()}</b> a <span style='color:#22c55e;font-weight:700'>{prob*100:.1f}% termination risk</span> — classified as <b>Low Risk</b>. Protective factors dominate this profile:"

            # ── Build risk factor pills ─────────────
            def shap_bar(val, color):
                width = min(abs(val) * 200, 100)
                return f'<div style="display:inline-block;height:6px;width:{width:.0f}px;background:{color};border-radius:99px;vertical-align:middle;margin-right:8px"></div>'

            risk_html = ""
            for feat, sv, sentence in risk_factors:
                bar = shap_bar(sv, '#ef4444')
                risk_html += (
                    f'<div style="display:flex;align-items:flex-start;gap:0.8rem;'
                    f'background:#1a0a0a;border:1px solid #ef444430;border-radius:10px;'
                    f'padding:0.8rem 1rem;margin-bottom:0.5rem">'
                    f'<span style="color:#ef4444;font-size:1rem;margin-top:2px">↑</span>'
                    f'<div>'
                    f'{bar}'
                    f'<span style="color:#94a3b8;font-size:0.83rem">{sentence}</span>'
                    f'</div></div>'
                )

            prot_html = ""
            for feat, sv, sentence in protective_factors:
                bar = shap_bar(sv, '#22c55e')
                prot_html += (
                    f'<div style="display:flex;align-items:flex-start;gap:0.8rem;'
                    f'background:#0a1a0a;border:1px solid #22c55e30;border-radius:10px;'
                    f'padding:0.8rem 1rem;margin-bottom:0.5rem">'
                    f'<span style="color:#22c55e;font-size:1rem;margin-top:2px">↓</span>'
                    f'<div>'
                    f'{bar}'
                    f'<span style="color:#94a3b8;font-size:0.83rem">{sentence}</span>'
                    f'</div></div>'
                )

            if not risk_html:
                risk_html = '<p style="color:#475569;font-size:0.82rem;font-style:italic">No significant risk factors identified.</p>'
            if not prot_html:
                prot_html = '<p style="color:#475569;font-size:0.82rem;font-style:italic">No significant protective factors identified.</p>'

            html_content = (
                '<div style="background:#0f1420;border:1px solid #1e2535;border-radius:14px;padding:1.5rem 1.8rem;margin-top:0.5rem">'
                '<p style="font-family:Syne,sans-serif;font-size:0.65rem;color:#334155;'
                'text-transform:uppercase;letter-spacing:0.1em;margin:0 0 0.8rem">Decision Explanation · SHAP Analysis</p>'
                f'<p style="color:#cbd5e1;font-size:0.88rem;line-height:1.6;margin-bottom:1.4rem">{verdict}</p>'
                '<p style="font-family:Syne,sans-serif;font-size:0.7rem;color:#ef4444;'
                'text-transform:uppercase;letter-spacing:0.1em;margin:0 0 0.6rem">⬆ Risk-Increasing Factors</p>'
                f'{risk_html}'
                '<p style="font-family:Syne,sans-serif;font-size:0.7rem;color:#22c55e;'
                'text-transform:uppercase;letter-spacing:0.1em;margin:1rem 0 0.6rem">⬇ Protective Factors</p>'
                f'{prot_html}'
                '<p style="font-size:0.7rem;color:#1e2d45;margin:1.2rem 0 0;font-style:italic">'
                'SHAP (SHapley Additive exPlanations) measures each feature\'s marginal contribution to the prediction. '
                '↑ pushes risk up · ↓ pushes risk down. Bar width reflects magnitude.'
                '</p>'
                '</div>'
            )
            st.markdown(html_content, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Could not compute explanation: {e}")

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────
#  TERMINATED EMPLOYEES
# ─────────────────────────────────────────────
with st.expander("📋 Previously Terminated Employees"):
    df_terminated = compute_seniority(df_terminated)
    st.dataframe(
        df_terminated[['Employee_Name','Department','Position','Seniority']]
        .rename(columns={'Employee_Name':'Employee'})
        .sort_values('Employee'),
        use_container_width=True,
        hide_index=True
    )

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="padding:2rem 0 1rem;text-align:center">
    <p style="font-size:0.72rem;color:#1e2535;letter-spacing:0.08em;text-transform:uppercase;
              font-family:Syne,sans-serif">
        PeopleIQ · LightGBM Model · CV ROC AUC 0.930 ± 0.015 · For internal HR use only
    </p>
</div>
""", unsafe_allow_html=True)