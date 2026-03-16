# %% [markdown]
# # Hackathon-AI : Ethical AI / Fairness Evaluation
# In this notebook, we evaluate the fairness of our baseline model using `fairlearn`.
# The goal is to ensure our model does not discriminate based on sensitive attributes.

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, demographic_parity_ratio
from sklearn.metrics import accuracy_score

# %% [markdown]
# ## 1. Load Data and Define Sensitive Feature
# We will evaluate fairness based on `Sex` (Gender).

# %%
df = pd.read_csv('data/HRDataset_v14.csv')

cols_to_drop = ['TermReason', 'DateofTermination', 'Employee_Name', 'EmpID', 'DaysLateLast30']
df_clean = df.drop(columns=cols_to_drop).copy()

# Sensitive feature
sensitive_feature = 'Sex'
A = df_clean[sensitive_feature]

# Normal features and target
X = df_clean.drop(columns=['Termd'])
y = df_clean['Termd']

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# %% [markdown]
# ## 2. Train baseline Model (same as before)

# %%
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# Note: We keep the sensitive feature in the split to evaluate on it
X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
    X, y, A, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Overall Accuracy:", accuracy_score(y_test, y_pred))

# %% [markdown]
# ## 3. Fairness Evaluation with Fairlearn
# We use `MetricFrame` to calculate metrics across the different groups (Male / Female).
# 
# **Key Concepts:**
# - **Selection Rate:** The fraction of predicted positive outcomes (attrition=1). Ideally similar across groups.
# - **Demographic Parity:** A state where the selection rate is independent of the sensitive feature.

# %%
metrics = {
    'accuracy': accuracy_score,
    'selection_rate': selection_rate
}

metric_frame = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=A_test
)

print("Metrics by group:")
print(metric_frame.by_group)

print("\nDemographic Parity Difference (should be close to 0):")
print(demographic_parity_difference(y_test, y_pred, sensitive_features=A_test))

print("\nDemographic Parity Ratio (should be close to 1):")
print(demographic_parity_ratio(y_test, y_pred, sensitive_features=A_test))

# %% [markdown]
# ## Conclusion for the Model Card
# If the Demographic Parity Ratio is < 0.8, the model might be exhibiting disparate impact against a specific gender. We will detail this in the Data/Model Card for the jury to show our Ethical AI assessment.
