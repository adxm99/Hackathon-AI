import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Ethical AI constraint: DO NOT use protected attributes:
# `Employee_Name`, `EmpID`, `DOB`, `Sex`, `MaritalDesc`, `CitizenDesc`, `HispanicLatino`, `RaceDesc`, `MarriedID`, `MaritalStatusID`, `GenderID`.

@st.cache_data
def load_and_preprocess_data():
    # Load dataset
    df = pd.read_csv("data/HRDataset_v14.csv")
    
    # 1. Select only relevant numerical/categorical features allowed + target 'Termd'
    # 'Termd' (1 = resigned/terminated, 0 = active)
    features_to_keep = ['Salary', 'Absences', 'DaysLateLast30', 
                        'SpecialProjectsCount', 'EmpSatisfaction', 
                        'EngagementSurvey', 'Department', 'Termd']
    
    # Filter columns ensuring they exist (in case of slight name mismatches)
    available_features = [f for f in features_to_keep if f in df.columns]
    df_filtered = df[available_features].copy()
    
    # Drop rows where target variable is NA (if any, though usually HR dataset target is solid)
    df_filtered.dropna(subset=['Termd'], inplace=True)
    
    # Fill missing values for numericals with median
    num_cols = df_filtered.select_dtypes(include=[np.number]).columns.drop('Termd')
    for col in num_cols:
        df_filtered[col] = df_filtered[col].fillna(df_filtered[col].median())
        
    # Fill missing values for categoricals with mode (Department)
    cat_cols = df_filtered.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df_filtered[col] = df_filtered[col].fillna(df_filtered[col].mode()[0])
        
        # Clean up department strings (strip whitespace)
        if col == 'Department':
            df_filtered[col] = df_filtered[col].str.strip()

    # Create dummy variables for 'Department'
    df_processed = pd.get_dummies(df_filtered, columns=['Department'], drop_first=True)
    
    # To make predictions from UI easier, keep the original categorical column unique values
    # for the Streamlit dropdown
    departments = df_filtered['Department'].dropna().unique().tolist()
    
    return df_processed, departments

def train_model(df_processed):
    X = df_processed.drop(columns=['Termd'])
    y = df_processed['Termd']
    
    # Train Decision Tree with max_depth=3 for explainability
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    
    return clf, X.columns

def main():
    st.set_page_config(page_title="Talent-Keeper AI: Turnover Prediction", page_icon="👩‍💼", layout="wide")
    
    st.title("Talent-Keeper AI: Turnover Prediction")
    st.markdown("""
    **Objective:** Empower HR teams to proactively identify and mitigate employee turnover risks.
    This tool utilizes a highly interpretable Decision Tree algorithm that **strictly adheres to Ethical AI principles** by omitting all demographic and protected characteristics (e.g., gender, race, age, marital status).
    """)
    
    # Load data and train model
    try:
        df_processed, departments = load_and_preprocess_data()
        model, feature_names = train_model(df_processed)
    except FileNotFoundError:
        st.error("Error: 'data/HRDataset_v14.csv' not found. Please ensure the dataset is in the 'data' folder relative to this script.")
        return
        
    # --- Sidebar: User Input (HR Tool) ---
    st.sidebar.header("Employee Profile Simulation")
    st.sidebar.markdown("Adjust parameters to simulate an employee profile and estimate turnover risk.")
    
    # Define default values based roughly on generic dataset medians/modes for better UX
    sim_salary = st.sidebar.slider("Salary ($)", min_value=30000, max_value=250000, value=65000, step=1000)
    sim_absences = st.sidebar.slider("Absences (Days)", min_value=0, max_value=30, value=5)
    sim_days_late = st.sidebar.slider("Days Late (Last 30 Days)", min_value=0, max_value=20, value=0)
    sim_projects = st.sidebar.slider("Special Projects Count", min_value=0, max_value=15, value=0)
    sim_satisfaction = st.sidebar.slider("Employee Satisfaction (1-5)", min_value=1, max_value=5, value=3)
    sim_engagement = st.sidebar.slider("Engagement Survey Score", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
    sim_dept = st.sidebar.selectbox("Department", departments)

    # Reconstruct input dataframe matching model features
    input_data = pd.DataFrame(0, index=[0], columns=feature_names) # Initialize with 0s (for one-hot encoded cols)
    
    input_data.loc[0, 'Salary'] = sim_salary
    input_data.loc[0, 'Absences'] = sim_absences
    input_data.loc[0, 'DaysLateLast30'] = sim_days_late
    input_data.loc[0, 'SpecialProjectsCount'] = sim_projects
    input_data.loc[0, 'EmpSatisfaction'] = sim_satisfaction
    input_data.loc[0, 'EngagementSurvey'] = sim_engagement
    
    # Set the one-hot encoded department feature to 1 if it matches (and exists in columns)
    dept_col = f"Department_{sim_dept}"
    if dept_col in input_data.columns:
        input_data.loc[0, dept_col] = 1
        
    # Make Prediction
    risk_proba = model.predict_proba(input_data)[0][1] # Probability of class 1 (Termd)
    risk_percentage = risk_proba * 100
    
    # Output Layout
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("Departure Risk Score")
        # Color code the result nicely
        if risk_percentage > 50:
            st.error(f"### ⚠️ {risk_percentage:.1f}%")
        elif risk_percentage > 25:
            st.warning(f"### 🟡 {risk_percentage:.1f}%")
        else:
            st.success(f"### ✅ {risk_percentage:.1f}%")
            
        # --- Prescriptive AI ---
        st.subheader("💡 Recommendation")
        if risk_percentage > 50:
            st.markdown("Immediate action is recommended to retain this talent.")
            
            # Simple business rules finding the "most penalizing" feature conceptually
            # In a shallow tree, we can just look at user inputs vs thresholds, but for simplicity:
            if sim_satisfaction <= 2:
                st.info("🎯 **Targeted Action**: Employee satisfaction is very low. Schedule a one-on-one "
                        "listen-only session to address their specific frustrations.")
            elif sim_engagement < 3.0:
                st.info("🎯 **Targeted Action**: Engagement is slipping. Recommend involving the employee in a new "
                        "collaboration initiative or project to boost inclusion.")
            elif sim_salary < 60000 and sim_projects > 3:
                st.info("🎯 **Targeted Action**: High workload (projects) relative to salary. Review the compensation "
                        "bracket or consider a spot bonus/promotion.")
            elif sim_absences > 15:
                st.info("🎯 **Targeted Action**: High absenteeism detected. Have HR/Mgmt conduct a well-being check "
                        "to ensure work-life balance or health issues are supported.")
            elif sim_days_late > 3:
                 st.info("🎯 **Targeted Action**: Frequent tardiness. Discuss workflow flexibility or potential burnout signals in a managerial one-on-one.")
            else:
                st.info("🎯 **Targeted Action**: Schedule a comprehensive retention interview to explore overall job satisfaction and career goals.")
        else:
             st.markdown("Risk is currently under control. Continue standard engagement protocols.")

    with col2:
        # --- Explainable AI (XAI) ---
        st.subheader("📊 Global Feature Importances")
        st.markdown("This chart explains *which criteria* the AI model relies on globally to make its predictions. This builds trust and transparency.")
        
        # Get importances
        importances = model.feature_importances_
        # Sort them for better plotting (top 5-7 is usually enough)
        feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_imp_df = feature_imp_df[feature_imp_df['Importance'] > 0] # Filter 0 importance
        feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=True)
        
        # Plot using matplotlib (or streamlit's native bar chart for simplicity)
        fig, ax = plt.subplots(figsize=(6, 4))
        # Better labels (removing 'Department_' prefix for readability)
        clean_labels = [label.replace('Department_', 'Dept: ') for label in feature_imp_df['Feature']]
        ax.barh(clean_labels, feature_imp_df['Importance'], color='skyblue')
        # ax.set_xlabel('Relative Importance')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        
if __name__ == '__main__':
    main()
