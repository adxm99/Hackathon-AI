# %% [markdown]
# # Hackathon-AI : Exploratory Data Analysis (EDA)
# This notebook explores the dataset `HRDataset_v14.csv` to understand the features, 
# target variable (`Termd`), and sensitive attributes for our Trusted AI approach.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
pd.set_option('display.max_columns', None)
sns.set_theme(style="whitegrid")

# %% [markdown]
# ## 1. Loading the Data

# %%
# Load the dataset
df = pd.read_csv('data/HRDataset_v14.csv')
print(f"Dataset shape: {df.shape}")
df.head()

# %% [markdown]
# ## 2. Basic Information and Missing Values

# %%
df.info()

# %%
# Check for missing values
missing_values = df.isnull().sum()
missing_values[missing_values > 0]

# %% [markdown]
# ## 3. Target Variable Analysis: Attrition (`Termd`)
# `Termd` = 1 indicates the employee has left the company.

# %%
# Distribution of the target variable
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Termd')
plt.title('Distribution of Attrition (Termd)')
plt.show()

print(df['Termd'].value_counts(normalize=True))

# %% [markdown]
# ## 4. Sensitive Attributes Analysis (Ethical AI)
# Let's look at `Sex`, `RaceDesc`, `MaritalDesc`, `CitizenDesc`, `HispanicLatino`.

# %%
sensitive_cols = ['Sex', 'RaceDesc', 'MaritalDesc']

for col in sensitive_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, y=col, hue='Termd')
    plt.title(f'Attrition by {col}')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 5. Performance and Engagement

# %%
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='PerformanceScore', hue='Termd')
plt.title('Attrition by Performance Score')
plt.show()

# %%
plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x='Termd', y='EmpSatisfaction')
plt.title('Attrition by Employee Satisfaction')
plt.show()

# %% [markdown]
# ## Next steps:
# - Clean text columns (e.g. `TermReason`)
# - Handle missing data
# - Correlational matrix
# - Prepare data for AIF360 fairness evaluation
