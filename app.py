import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# --- Page config ---
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="👥",
    layout="wide"
)

# --- Load model and assets ---
@st.cache_resource
def load_model():
    model = joblib.load('lgbm_model.pkl')
    features = joblib.load('feature_columns.pkl')
    return model, features

@st.cache_data
def load_data():
    df = pd.read_csv('employees.csv')
    
    # Split into active and already terminated
    df_active     = df[df['Termd'] == 0].copy()  # only predict on active employees
    df_terminated = df[df['Termd'] == 1].copy()  # keep for reference
    
    return df, df_active, df_terminated

model, feature_columns = load_model()
df_raw, df_active, df_terminated = load_data()

# --- Seniority computation (must match your notebook exactly) ---
def compute_seniority(df):
    df = df.copy()
    today = datetime.today()
    df['DateofHire'] = pd.to_datetime(df['DateofHire'])
    df['Seniority'] = (today - df['DateofHire']).dt.days / 365.25
    return df

# --- Feature engineering (must match your notebook exactly) ---
def prepare_features(df):
    df = compute_seniority(df)

    # Fix string spaces before one-hot encoding
    string_columns = df.select_dtypes(include=['object', 'string']).columns
    for col in string_columns:
        df[col] = df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)

    # Map PerformanceScore to numeric
    performance_mapping = {'PIP': 1, 'Needs Improvement': 2, 'Fully Meets': 3, 'Exceeds': 4}
    df['PerformanceScore'] = df['PerformanceScore'].map(performance_mapping)

    # IMPORTANT: LightGBM needs explicit types for object -> numeric mappings
    # If there are missing values after mapping or unmapped values, they stay float/NaN causing object type inference later
    df['PerformanceScore'] = pd.to_numeric(df['PerformanceScore'], errors='coerce').fillna(3) # Default to fully meets

    # Calculate DaysSinceLastReview (if it was included in training logic, depending on user's df_raw)
    # The error came from PerformanceScore type, let's make sure it's strictly numeric

    # One-hot encode categorical columns
    categorical_cols = ['Position', 'Department', 'ManagerName', 'RecruitmentSource']
    df_encoded = pd.get_dummies(df, columns=categorical_cols)

    # Drop leaky and ID columns
    cols_to_drop = ['EmpID', 'MarriedID', 'GenderID', 'DeptID', 'PositionID',
                    'ManagerID', 'MaritalStatusID', 'EmpStatusID',
                    'DateofHire', 'Employee_Name', 'DateofTermination', 'TermReason', 'EmploymentStatus']
    df_encoded = df_encoded.drop(columns=[c for c in cols_to_drop if c in df_encoded.columns])

    # Align columns to match training features exactly as loaded from joblib
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)
    
    # Final safety check: force all boolean one-hot columns to standard int/float
    # just in case LightGBM struggles with pandas internal boolean arrays
    for col in df_encoded.columns:
        df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce').fillna(0)

    # Drop leaky and ID columns
    cols_to_drop = ['EmpID', 'MarriedID', 'GenderID', 'DeptID', 'PositionID',
                    'ManagerID', 'MaritalStatusID', 'EmpStatusID',
                    'DateofHire', 'Employee_Name']
    df_encoded = df_encoded.drop(columns=[c for c in cols_to_drop if c in df_encoded.columns])

    # Align columns to match training features
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    return df_encoded

# --- Run predictions ---
def get_predictions(df_raw):
    X = prepare_features(df_raw)
    probabilities = model.predict_proba(X)[:, 1]
    return probabilities

# ============================================================
#                        UI
# ============================================================

st.title("👥 HR Attrition Risk Predictor")
st.markdown("Predict the probability of contract termination for each employee using LightGBM.")
st.divider()

# --- Run predictions ---
with st.spinner("Running predictions..."):
    df_active = compute_seniority(df_active)
    df_active['Termination_Probability'] = get_predictions(df_active)
    df_active['Risk_Level'] = pd.cut(
        df_active['Termination_Probability'],
        bins=[0, 0.08, 0.25, 1.0],
        labels=['🟢 Low', '🟡 Medium', '🔴 High']
    )

# --- Sidebar filters ---
st.sidebar.header("🔍 Filters")

risk_filter = st.sidebar.multiselect(
    "Filter by Risk Level",
    options=['🟢 Low', '🟡 Medium', '🔴 High'],
    default=['🟢 Low', '🟡 Medium', '🔴 High']
)

dept_filter = st.sidebar.multiselect(
    "Filter by Department",
    options=sorted(df_active['Department'].unique()),
    default=sorted(df_active['Department'].unique())
)

top_n = st.sidebar.slider("Show top N at-risk employees", 5, 50, 10)

# --- Apply filters ---
df_filtered = df_active[
    df_active['Risk_Level'].isin(risk_filter) &
    df_active['Department'].isin(dept_filter)
].sort_values('Termination_Probability', ascending=False)

# --- KPI cards ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Active Employees", len(df_active))
col2.metric("🔴 High Risk",   len(df_active[df_active['Risk_Level'] == '🔴 High']))
col3.metric("🟡 Medium Risk", len(df_active[df_active['Risk_Level'] == '🟡 Medium']))
col4.metric("🟢 Low Risk",    len(df_active[df_active['Risk_Level'] == '🟢 Low']))

st.divider()

# --- Top at-risk employees table ---
st.subheader(f"🔴 Top {top_n} Employees at Highest Risk")

display_cols = ['Employee_Name', 'Department', 'Position',
                'Seniority', 'Termination_Probability', 'Risk_Level']

df_display = df_filtered[display_cols].head(top_n).copy()
df_display['Seniority'] = df_display['Seniority'].round(1).astype(str) + ' yrs'
df_display['Termination_Probability'] = (
    df_display['Termination_Probability'] * 100
).round(1).astype(str) + '%'

st.dataframe(
    df_display.rename(columns={
        'Employee_Name': 'Employee',
        'Termination_Probability': 'Risk %',
        'Risk_Level': 'Risk Level'
    }),
    use_container_width=True,
    hide_index=True
)

st.divider()

# --- Individual employee lookup ---
st.subheader("🔎 Individual Employee Lookup")

employee_name = st.selectbox(
    "Select an employee",
    options=sorted(df_active['Employee_Name'].unique())
)

emp_row = df_active[df_active['Employee_Name'] == employee_name].iloc[0]
prob = emp_row['Termination_Probability']

col1, col2, col3 = st.columns(3)
col1.metric("Employee",   emp_row['Employee_Name'])
col2.metric("Department", emp_row['Department'])
col3.metric("Seniority",  f"{emp_row['Seniority']:.1f} years")

# Risk gauge
st.markdown(f"### Termination Risk: `{prob*100:.1f}%`")
st.progress(float(prob))

if prob >= 0.6:
    st.error(f"🔴 **High Risk** — Immediate retention action recommended")
elif prob >= 0.3:
    st.warning(f"🟡 **Medium Risk** — Monitor and engage this employee")
else:
    st.success(f"🟢 **Low Risk** — Employee appears stable")

st.divider()

with st.expander("📋 Already Terminated Employees"):
    df_terminated = compute_seniority(df_terminated)
    st.dataframe(
        df_terminated[['Employee_Name', 'Department', 'Position', 'Seniority']],
        use_container_width=True,
        hide_index=True
    )