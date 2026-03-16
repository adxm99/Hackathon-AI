# Hackathon Trusted AI x HR - Team [Name]

## 1. Objectives & Pitch
Our goal is to assist HR managers in identifying employees at risk of attrition (`Termd`) and understanding the key factors behind this risk. 
This solution adheres to the principles of **Responsible AI**:
1.  **Explainability (XAI):** We avoid "black box" models by integrating SHAP to explain exactly why an employee is flagged as a risk.
2.  **Ethical AI:** We evaluate our model for fairness against sensitive attributes (e.g., Sex) to ensure no algorithmic discrimination occurs.

## 2. Approach & Scope
We tackle both structured and unstructured data:
-   **Structured ML:** An XGBoost model trained on demographic and performance data.
-   **NLP Integration:** Simulation of unstructured HR feedback (e.g., exit interviews, transfer requests) processed via NLP techniques (TF-IDF) and merged with the structured pipeline to create a **Hybrid Model**.

## 3. Persona
**HR Manager / Director:** Needs a transparent dashboard to view at-risk talent, comprehend the reasons (salary, manager, environment), and take preventive actions.

---

## 4. Technical Documentation

### Repository Structure
-   `data/` : Contains the HR dataset (`HRDataset_v14.csv`).
-   `notebooks/` :
    -   `01_EDA.py` : Data exploration and distribution analysis.
    -   `02_Baseline_Model.py` : Base XGBoost predicting `Termd` with global SHAP explanations.
    -   `03_NLP_Hybrid_Model.py` : Creation of simulated HR texts, TF-IDF extraction, and hybrid XGBoost model.
    -   `04_Ethical_AI_Fairness.py` : Audit of the model's predictions using Fairlearn to check Demographic Parity.
-   `docs/` : Contains the Data Card and Model Card.

### Setup Instructions
1.  Initialize a virtual environment (optional but recommended).
2.  Install dependencies: `pip install pandas scikit-learn xgboost shap fairlearn`
3.  Run the notebooks sequentially to reproduce the pipeline.

---

## 5. Architecture Scheme
1.  **Input:** Structured HR Database + Unstructured HR Notes.
2.  **Preprocessing:** 
    -   StandardScaler for numerics.
    -   OneHotEncoder for categoricals.
    -   TF-IDF Vectorizer for text.
3.  **Modeling:** XGBoost Classifier (Hybrid Feature Union).
4.  **Audit Layer:** 
    -   *SHAP Explainer* (Identify top drivers of attrition).
    -   *Fairlearn MetricFrame* (Audit selection rate across demographics).
5.  **Output:** Risk Score + Explanation + Fairness Report.
