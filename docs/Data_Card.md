# Data Card

## Context
This dataset (`HRDataset_v14.csv`) was provided for a hackathon aiming to build a Trusted AI solution for HR retention. It contains roughly ~300 records and captures demographic, compensation, and performance metrics of a fictional company.

## Variables Overview
-   **Target Variable:** `Termd` (1 if the employee resigned or was terminated, 0 if still employed).
-   **Sensitive Attributes:** 
    -   `Sex` (M/F)
    -   `RaceDesc` (e.g., White, Black or African American, Asian, etc.)
    -   `MaritalDesc` (Single, Married, Divorced, etc.)
    -   `HispanicLatino` (Yes/No)
-   **Performance Metrics:** `EmpSatisfaction`, `PerformanceScore`, `Absences`, `EngagementSurvey`.

## Preprocessing Steps
1.  **Irrelevant / Leakage Features:** Removed direct leakage features (`TermReason`, `DateofTermination`) to avoid artificial high performance (because a model shouldn't have access to future data like termination date when predicting attrition). Also removed identifiers (`EmpID`, `Employee_Name`).
2.  **Imputation:** Missing continuous variables were kept as is (XGBoost handles missing data naturally).
3.  **Numerical Variables:** Handled via StandardScaler.
4.  **Categorical Variables:** Handled via OneHotEncoder.
5.  **Simulated Unstructured Text:** We simulated a column (`HR_FeedbackText`) utilizing the existing `EmpSatisfaction` score. A low score simulates negative or transfer request texts, a high score simulates positive texts. We preprocessed this text feature using a TF-IDF vectorizer (max 50 features).

## Bias / Ethical Analysis
A review of the sensitive features (`Sex`) in relation to the attrition target was conducted. The model training and evaluation process was designed to output fairness metrics such as Demographic Parity via Fairlearn to make sure the false positive/negative rates do not skew unfairly against any protected group.
