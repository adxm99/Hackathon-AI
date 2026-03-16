# %% [markdown]
# # Hackathon-AI : NLP and Hybrid Model Architecture
# In this notebook we will:
# 1. Simulate unstructured text data (HR feedbacks, transfer requests) for our employees.
# 2. Extract features from these texts (Sentiment, Topics).
# 3. Merge with structured data to train a Hybrid predictive model.

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
import shap
import random

# %% [markdown]
# ## 1. Load Data and Simulate Textual Feedback
# Since we don't have real textual feedback, we'll simulate basic sentences based on the employee's `EmpSatisfaction` and `Termd` status.

# %%
df = pd.read_csv('data/HRDataset_v14.csv')

# Drop identifying/leakage columns
cols_to_drop = ['TermReason', 'DateofTermination', 'Employee_Name', 'EmpID', 'DaysLateLast30']
df_clean = df.drop(columns=cols_to_drop).copy()

# Simple Simulation Function
def simulate_hr_feedback(row):
    satisfaction = row.get('EmpSatisfaction', 3)
    termd = row.get('Termd', 0)
    
    positive_comments = [
        "Everything is great.", "I love my team.", "The management is supportive.",
        "I see a long future here.", "Great benefits and work-life balance."
    ]
    neutral_comments = [
        "Things are okay.", "No major complaints.", "Work is steady.", 
        "My manager is alright.", "The workload is manageable."
    ]
    negative_comments = [
        "I feel undervalued.", "Salary is too low for the work.", "Manager micro-manages.",
        "Tolerating a toxic environment.", "Looking for better opportunities."
    ]
    transfer_requests = [
        " I requested a transfer to a different department but got denied.",
        " I'd like to work on new projects.", ""
    ]
    
    if termd == 1 or satisfaction < 3:
        # Higher chance of negative
        comment = random.choice(negative_comments) + random.choice(transfer_requests)
    elif satisfaction > 3:
        comment = random.choice(positive_comments)
    else:
        comment = random.choice(neutral_comments)
        
    return comment.strip()

# Apply simulation
np.random.seed(42)
random.seed(42)
df_clean['HR_FeedbackText'] = df_clean.apply(simulate_hr_feedback, axis=1)

print(df_clean[['Termd', 'EmpSatisfaction', 'HR_FeedbackText']].head(10))

# %% [markdown]
# ## 2. Prepare Hybrid Features 
# We will use TF-IDF for the text and standard encoding for the rest.

# %%
X = df_clean.drop(columns=['Termd'])
y = df_clean['Termd']

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# The text column is conceptually object, but we treat it separately from standard categorical
if 'HR_FeedbackText' in categorical_features:
    categorical_features.remove('HR_FeedbackText')

# %% [markdown]
# ## 3. Build Preprocessing Pipeline

# %%
# Pipeline for numerical/categorical
standard_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# We combine structured preprocessing with text feature extraction
hybrid_preprocessor = ColumnTransformer(
    transformers=[
        ('structured', standard_preprocessor, numeric_features + categorical_features),
        ('text_tfidf', TfidfVectorizer(max_features=50, stop_words='english'), 'HR_FeedbackText')
    ])

# %% [markdown]
# ## 4. Train Hybrid Model

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = Pipeline(steps=[
    ('preprocessor', hybrid_preprocessor),
    ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

model.fit(X_train, y_train)

# %% [markdown]
# ## 5. Model Evaluation

# %%
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# %% [markdown]
# ## 6. Explainability (SHAP) for Hybrid Model
# Explaining models with text pipelines can be tricky. We extract the transformed dataset first.

# %%
# Transform train and test
X_train_transformed = model.named_steps['preprocessor'].fit_transform(X_train)
X_test_transformed = model.named_steps['preprocessor'].transform(X_test)

# Get feature names
cat_encoder = model.named_steps['preprocessor'].named_transformers_['structured'].named_transformers_['cat']
cat_feature_names = cat_encoder.get_feature_names_out(categorical_features).tolist()

tfidf = model.named_steps['preprocessor'].named_transformers_['text_tfidf']
text_feature_names = ["tfidf_" + w for w in tfidf.get_feature_names_out()]

all_feature_names = numeric_features + cat_feature_names + text_feature_names

# Train standalone XGB on transformed data for SHAP
xgb_hybrid = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_hybrid.fit(X_train_transformed, y_train)

explainer = shap.TreeExplainer(xgb_hybrid)
shap_values = explainer.shap_values(X_test_transformed)

# You will notice TF-IDF words appearing in the SHAP summary plot!
# shap.summary_plot(shap_values, X_test_transformed, feature_names=all_feature_names)
print("SHAP explanation ready to plot.")
