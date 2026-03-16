# Model Card

## Objective
Predict Employee Attrition (`Termd`) based on structured HR data and unstructured feedback texts.
The final model is a **Hybrid XGBoost Classifier**, taking inputs from basic categorical/numeric features as well as text embeddings (TF-IDF).

## Intended Use
-   **Primary Use Case:** Proactive identification of employees at high risk of quitting to enable HR retention initiatives.
-   **Intended Users:** HR Managers and Directors.
-   **Out-of-Scope:** Not intended to dictate automatic firing or compensation decisions without human review.

## Architecture
1.  **Tabular Pipeline:** Uses a `StandardScaler` for numeric columns and a `OneHotEncoder` for categorical strings.
2.  **NLP Pipeline:** Analyzes a simulated `HR_FeedbackText` feature using a `TfidfVectorizer` (top 50 features, removing English stop words) to extract sentiment and theme intensity.
3.  **Classification:** A `FeatureUnion` structure passing the total array to an `XGBClassifier` optimized for logloss.

## Evaluation & Metrics
Based on the default Train/Test split (80/20):
-   **Classification Metrics:** Evaluates Accuracy, Precision, Recall, and F1-Score focusing on the minority class (`Termd=1`). The model currently exhibits a baseline accuracy (around ~80% depending on the random split).
-   **Fairness Metrics:** Evaluated specifically against the `Sex` attribute using `fairlearn.metrics.demographic_parity_ratio`. The goal is to ensure the ratio is > 0.8 to comply with the four-fifths rule (minimizing disparate impact).

## Ethical Considerations
If the model relies heavily on age, gender, or race, it could perpetuate existing HR biases. That is why **Ethical AI** constraints have been enforced via Fairlearn auditing, and **Explainable AI (XAI)** constraints with SHAP are added to give transparency on feature impact.

## Explainability (SHAP)
The SHAP Tree Explainer demonstrates which factors push the prediction towards Attrition.
For example, if an employee has a lower `EmpSatisfaction`, the XGBoost model outputs a higher logic score towards `1`. If the employee has positive keywords in their feedback text (e.g., "love", "great"), the TF-IDF features act as a counter-weight pulling the probability down.
