import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

df_original = pd.read_csv('data/HRDataset_v14.csv')
df = df_original.copy()

df['DateofHire'] = pd.to_datetime(df['DateofHire'])
df['DateofTermination'] = pd.to_datetime(df['DateofTermination'])
date_extraction = pd.to_datetime('2019-12-31')
df['Seniority'] = np.where(df['Termd'] == 1,
                                  (df['DateofTermination'] - df['DateofHire']).dt.days / 365.25,
                                  (date_extraction - df['DateofHire']).dt.days / 365.25)
df['Seniority'] = df['Seniority'].round(2)

df['LastPerformanceReview_Date'] = pd.to_datetime(df['LastPerformanceReview_Date'])
df['DaysSinceLastReview'] = np.where(df['Termd'] == 1,
                                    (df['DateofTermination'] - df['LastPerformanceReview_Date']).dt.days,
                                    (date_extraction - df['LastPerformanceReview_Date']).dt.days)
df['DaysSinceLastReview'] = df['DaysSinceLastReview'].fillna(df['DaysSinceLastReview'].median())

colonnes_a_garder = [
    'Salary', 'Position', 'Department', 'ManagerName', 'PerformanceScore', 
    'EngagementSurvey', 'EmpSatisfaction', 'SpecialProjectsCount', 'Absences', 
    'DaysLateLast30', 'RecruitmentSource', 'Termd', 'Seniority', 'DaysSinceLastReview'
]

df_features = df.copy()
string_columns = df_features.select_dtypes(include=['object', 'string']).columns
for col in string_columns:
    df_features[col] = df_features[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)

df_features = df_features[colonnes_a_garder].copy()

performance_mapping = {'PIP': 1, 'Needs Improvement': 2, 'Fully Meets': 3, 'Exceeds': 4}
df_features['PerformanceScore'] = df_features['PerformanceScore'].map(performance_mapping)

categorical_cols_to_encode = ['Position', 'Department', 'ManagerName', 'RecruitmentSource']
df_features = pd.get_dummies(df_features, columns=categorical_cols_to_encode, drop_first=True)

X = df_features.drop('Termd', axis=1)
y = df_features['Termd']

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X, y)

print(export_text(tree, feature_names=list(X.columns)))
