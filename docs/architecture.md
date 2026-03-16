# Architecture

## Pipeline Overview

Our solution follows this pipeline:

HR Dataset → Data Preprocessing → Attrition Prediction Model → Fairness Audit → Explainability Module → HR Decision Support

## Modules

### 1. Data Ingestion
The raw HR dataset is loaded from the repository.

### 2. Preprocessing
The dataset is cleaned, date variables are transformed, leakage columns are removed, and relevant features are prepared for modeling.

### 3. Predictive Model
A supervised classification model estimates the probability that an employee may leave the company.

### 4. Fairness Audit
Model predictions are evaluated across sensitive groups such as sex and ethnicity in order to detect possible disparities.

### 5. Explainability
The solution provides both:
- global explanations: which variables matter the most,
- local explanations: why a given employee is predicted as high-risk.

### 6. HR Decision Support
The final output is intended to support HR managers in identifying high-risk employees and planning preventive retention actions.

## Positioning
This system is designed as a transparent and responsible HR support tool, not as an automatic decision-maker.