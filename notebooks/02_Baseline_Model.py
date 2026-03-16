# %% [markdown]
# # Hackathon-AI : Baseline Predictive Model
# This script builds a baseline model to predict `Termd` (attrition).
# We also implement a basic Ethical AI check to see if the predictions are fair.

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import shap

# %% [markdown]
# ## 1. Data Preparation

# %%
df = pd.read_csv('data/HRDataset_v14.csv')

# Drop columns that are essentially leakage or irrelevant for forward prediction
# - TermReason, DateofTermination are directly tied to the target 'Termd'
# - Employee_Name, EmpID are identifiers
cols_to_drop = ['TermReason', 'DateofTermination', 'Employee_Name', 'EmpID', 'DaysLateLast30'] # Assuming DaysLateLast30 might be missing for older records
df_clean = df.drop(columns=cols_to_drop).copy()

# Target and features
X = df_clean.drop(columns=['Termd'])
y = df_clean['Termd']

# Identify numerical and categorical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# %% [markdown]
# ## 2. Preprocessing Pipeline

# %%
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# %% [markdown]
# ## 3. Train-Test Split & Modeling

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# We use XGBoost as it's typically a strong baseline for tabular data
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# %% [markdown]
# ## 4. Evaluation

# %%
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# %% [markdown]
# ## 5. Explainable AI (XAI) using SHAP (Global Explanation)
# Let's see what structured variables influence the `Termd` prediction most.

# %%
# Extract the preprocessed training data for SHAP
X_train_transformed = model.named_steps['preprocessor'].fit_transform(X_train)
X_test_transformed = model.named_steps['preprocessor'].transform(X_test)

# Get feature names from the preprocessor
cat_encoder = model.named_steps['preprocessor'].named_transformers_['cat']
cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
all_feature_names = numeric_features.tolist() + cat_feature_names.tolist()

# Train a separate XGB model directly on transformed data for SHAP compatibility easily
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_transformed, y_train)

# Initialize explanation object
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_transformed)

# Plot summary
shap.summary_plot(shap_values, X_test_transformed, feature_names=all_feature_names)

# %% [markdown]
# ## Next Steps: Ethical AI & NLP
# - Audit fairness of `y_pred` against protected attributes (e.g., Sex, RaceDesc) via `fairlearn` or `AIF360`.
# - Integrate simulated internal transfer texts/feedback with NLP models for a hybrid prediction approach.
