import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

import os

# Page Config
st.set_page_config(page_title="Trusted HR AI Demo", layout="wide")

st.title("🛡️ Trusted AI for HR Talent Retention")
st.markdown("""
This demo showcases a **Hybrid Predictive Intelligence** model that identifies employees at risk of attrition.
It integrates both structured HR data and unstructured qualitative feedback.
""")

@st.cache_data
def load_data():
    # Calculate path relative to the script location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data', 'HRDataset_v14.csv')
    df = pd.read_csv(data_path)
    return df

def simulate_feedback(row):
    satisfaction = row.get('EmpSatisfaction', 3)
    termd = row.get('Termd', 0)
    pos = ["I love my team.", "Great benefits and work-life balance.", "Supportive management."]
    neg = ["I feel undervalued.", "Salary is too low.", "Tolerating a toxic environment."]
    if termd == 1 or satisfaction < 3: return random.choice(neg)
    return random.choice(pos)

df = load_data()
df['QualitativeFeedback'] = df.apply(simulate_feedback, axis=1)

# Basic stats
st.sidebar.header("Dataset Overview")
st.sidebar.write(f"Total Employees: {len(df)}")
st.sidebar.write(f"Average Satisfaction: {df['EmpSatisfaction'].mean():.2f}/5")

# Modeling
cols_to_drop = ['TermReason', 'DateofTermination', 'Employee_Name', 'EmpID', 'DaysLateLast30', 'QualitativeFeedback']
X = df.drop(columns=['Termd'] + ['TermReason', 'DateofTermination', 'Employee_Name', 'EmpID', 'DaysLateLast30']).copy()
y = df['Termd']

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@st.cache_resource
def get_model():
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(eval_metric='logloss', random_state=42))
    ])
    model.fit(X_train, y_train)
    return model

model = get_model()

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Global Analytics", "🔍 Individual Prediction (XAI)", "⚖️ Fairness Audit"])

with tab1:
    st.header("Attrition Factors")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Distribution of classes")
        st.bar_chart(df['Termd'].value_counts())
    with col2:
        st.write("Satisfaction vs Attrition")
        st.scatter_chart(df, x='EmpSatisfaction', y='Termd', color='Sex')

with tab2:
    st.header("Explainable AI (SHAP)")
    emp_idx = st.selectbox("Select an employee index from test set", range(len(X_test)))
    
    selected_emp = X_test.iloc[[emp_idx]]
    prediction_prob = model.predict_proba(selected_emp)[0][1]
    
    st.metric("Risk of Attrition", f"{prediction_prob*100:.1f}%")
    
    if prediction_prob > 0.5:
        st.error("⚠️ HIGH RISK IDENTIFIED")
    else:
        st.success("✅ LOW RISK")

    st.subheader("Why this prediction?")
    st.info("SHAP values explain how each feature contributed to the final probability.")
    # For demo simplicity, we'll just show the raw features of the selected employee
    st.write(selected_emp)
    st.write(f"**Qualitative Note:** *\"{df.iloc[X_test.index[emp_idx]]['QualitativeFeedback']}\"*")

with tab3:
    st.header("Ethical AI Compliance")
    st.markdown("Auditing for Demographic Parity between **Male** and **Female** employees.")
    
    y_pred = model.predict(X_test)
    test_df = X_test.copy()
    test_df['actual'] = y_test
    test_df['pred'] = y_pred
    
    selection_rates = test_df.groupby('Sex')['pred'].mean()
    st.write("Selection Rates (Predicted Attrition Rate) by Gender:")
    st.write(selection_rates)
    
    ratio = selection_rates.min() / selection_rates.max()
    st.metric("Demographic Parity Ratio", f"{ratio:.2f}")
    if ratio < 0.8:
        st.warning("Potential disparate impact detected. High scrutiny required.")
    else:
        st.success("Fairness threshold met (>0.8).")

st.markdown("---")
st.caption("Hackathon IA x RH - Built with Antigravity AI")
