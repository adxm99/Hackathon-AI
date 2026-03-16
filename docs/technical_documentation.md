# Technical Documentation

## Objective
Build a trusted AI solution for employee attrition prediction.

## Problem Formulation
We model attrition as a binary classification task using `Termd`.

## Main Steps
1. Load the raw dataset
2. Remove identifiers and leakage columns
3. Convert dates and engineer features
4. Split train/test data
5. Train baseline and comparison models
6. Evaluate predictive performance
7. Audit fairness across sensitive groups
8. Produce explainability outputs

## Leakage Prevention
Columns such as `DateofTermination`, `TermReason`, and `EmploymentStatus` are removed because they reveal the target and would artificially inflate performance.

## Feature Engineering
Derived features include:
- age
- tenure
- days since last review

## Modeling Strategy
Two models are trained:
- Logistic Regression
- Random Forest

## Evaluation Metrics
- accuracy
- precision
- recall
- F1-score
- ROC-AUC

## Fairness Audit
The model is audited across:
- sex
- race / ethnicity

Metrics compared by group include:
- accuracy
- recall
- false positive rate
- false negative rate

## Explainability
Global and local explanations are generated in order to support HR interpretability.

## Intended Use
Decision support for HR managers.

## Non-Intended Use
Fully automated HR decisions without human review.