import pandas as pd
import joblib
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Trusted AI for HR Retention", layout="wide")

DATA_PATH = Path("data/processed/hr_model_ready.csv")
MODEL_PATH = Path("models/logistic_regression.joblib")
FAIRNESS_PATH = Path("models/fairness_metrics.csv")
METRICS_PATH = Path("models/model_metrics.csv")
TARGET = "Termd"
SENSITIVE_COLS = ["Sex", "RaceDesc", "HispanicLatino"]

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    fairness_df = pd.read_csv(FAIRNESS_PATH)
    metrics_df = pd.read_csv(METRICS_PATH)
    return df, fairness_df, metrics_df

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def get_recommendation(probability: float) -> str:
    if probability >= 0.75:
        return "High risk: schedule a retention interview and review engagement, workload, and compensation."
    if probability >= 0.50:
        return "Moderate risk: manager follow-up and targeted retention actions recommended."
    return "Low risk: continue regular monitoring."


def build_local_explanation(model, employee_features: pd.DataFrame) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    x_trans = preprocessor.transform(employee_features)
    feature_names = preprocessor.get_feature_names_out()
    coefs = classifier.coef_[0]

    values = x_trans.toarray()[0] if hasattr(x_trans, "toarray") else x_trans[0]
    contrib = pd.DataFrame({
        "feature": feature_names,
        "contribution": values * coefs
    }).sort_values("contribution", ascending=False)
    return contrib


def main():
    st.title("Trusted AI for Employee Retention")
    st.write("Demo app for HR attrition prediction, explainability, and fairness audit.")

    try:
        df, fairness_df, metrics_df = load_data()
        model = load_model()
    except Exception as e:
        st.error(f"Could not load files: {e}")
        st.info("Make sure these files exist: data/processed/hr_model_ready.csv, models/logistic_regression.joblib, models/fairness_metrics.csv, models/model_metrics.csv")
        return

    X = df.drop(columns=[TARGET] + [c for c in SENSITIVE_COLS if c in df.columns], errors="ignore")

    tab1, tab2, tab3 = st.tabs(["Prediction Demo", "Model Metrics", "Fairness Audit"])

    with tab1:
        st.subheader("Employee-level prediction")
        employee_index = st.number_input("Choose employee index", min_value=0, max_value=len(df)-1, value=0, step=1)

        employee_full = df.iloc[[employee_index]].copy()
        employee_features = X.iloc[[employee_index]].copy()

        col1, col2 = st.columns([1, 1])
        with col1:
            st.write("Selected employee")
            st.dataframe(employee_full, use_container_width=True)

        pred_class = model.predict(employee_features)[0]
        pred_proba = model.predict_proba(employee_features)[0, 1]
        recommendation = get_recommendation(pred_proba)

        with col2:
            st.metric("Predicted class", int(pred_class))
            st.metric("Attrition probability", f"{pred_proba:.1%}")
            st.info(recommendation)

        st.subheader("Local explanation")
        contrib = build_local_explanation(model, employee_features)
        pos = contrib.head(10).copy()
        neg = contrib.tail(10).sort_values("contribution").copy()

        c1, c2 = st.columns(2)
        with c1:
            st.write("Top factors increasing risk")
            st.dataframe(pos, use_container_width=True)
        with c2:
            st.write("Top factors decreasing risk")
            st.dataframe(neg, use_container_width=True)

    with tab2:
        st.subheader("Main model metrics")
        st.dataframe(metrics_df, use_container_width=True)
        st.caption("Recommended main model for the presentation: Logistic Regression.")

    with tab3:
        st.subheader("Fairness audit")
        st.dataframe(fairness_df, use_container_width=True)
        st.caption("Interpret cautiously: some groups have very small sample sizes.")

    st.markdown("---")
    st.write("This tool is a decision-support demo, not an automated HR decision-maker.")


if __name__ == "__main__":
    main()
