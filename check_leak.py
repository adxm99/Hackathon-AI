import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df_original = pd.read_csv('data/HRDataset_v14.csv')
df = df_original.copy()

string_columns = df.select_dtypes(include=['object', 'string']).columns
for col in string_columns:
    df[col] = df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)

colonnes_a_garder = [
    'Salary', 'Position', 'Department', 'ManagerName', 'PerformanceScore', 
    'EngagementSurvey', 'EmpSatisfaction', 'SpecialProjectsCount', 'Absences', 
    'DaysLateLast30', 'RecruitmentSource', 'Termd'
]
df_features = df[colonnes_a_garder].copy()

df_features['DateofHire'] = pd.to_datetime(df_original['DateofHire'])
df_features['DateofTermination'] = pd.to_datetime(df_original['DateofTermination'])
date_extraction = pd.Timestamp.today()
df_features['Anciennete_Annees'] = np.where(df_features['Termd'] == 1,
                                  (df_features['DateofTermination'] - df_features['DateofHire']).dt.days / 365.25,
                                  (date_extraction - df_features['DateofHire']).dt.days / 365.25)
df_features['Anciennete_Annees'] = df_features['Anciennete_Annees'].round(2)
df_features = df_features.drop(columns=['DateofHire', 'DateofTermination'])

df_features['LastPerformanceReview_Date'] = pd.to_datetime(df_original['LastPerformanceReview_Date'])
df_features['DaysSinceLastReview'] = (date_extraction - df_features['LastPerformanceReview_Date']).dt.days
median_days = df_features['DaysSinceLastReview'].median()
df_features['DaysSinceLastReview'] = df_features['DaysSinceLastReview'].fillna(median_days)
df_features = df_features.drop(columns=['LastPerformanceReview_Date'])

performance_mapping = {'PIP': 1, 'Needs Improvement': 2, 'Fully Meets': 3, 'Exceeds': 4}
df_features['PerformanceScore'] = df_features['PerformanceScore'].map(performance_mapping)

categorical_cols_to_encode = ['Position', 'Department', 'ManagerName', 'RecruitmentSource']
df_features = pd.get_dummies(df_features, columns=categorical_cols_to_encode, drop_first=True)

X = df_features.drop('Termd', axis=1)
y = df_features['Termd']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

acc = accuracy_score(y_test, xgb_model.predict(X_test))
print(f"Accuracy: {acc}")

importances = list(zip(X.columns, xgb_model.feature_importances_))
importances.sort(key=lambda x: x[1], reverse=True)
for feat, imp in importances[:5]:
    print(f"{feat}: {imp}")
