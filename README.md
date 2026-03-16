# Trusted AI for Employee Retention

## Project Overview
This project aims to build a trusted AI solution for Human Resources in order to identify employees at risk of leaving the company, explain the factors behind this risk, and support preventive HR actions.

## Objective
Our main objective is to predict employee attrition using the `Termd` variable and provide a transparent and fair decision-support tool for HR teams.

## Chosen Themes
- Ethical AI
- Explainable AI

## Business Use Case
HR managers need to:
- identify employees at risk of leaving,
- understand the main drivers of attrition,
- make better retention decisions,
- ensure the model does not produce unfair outcomes across sensitive groups.

## Persona
**HR Manager**
- Needs early alerts on employee attrition risk
- Wants understandable and defensible model outputs
- Must avoid unfair or biased decisions

## Dataset
We use a synthetic HR dataset containing employee information such as salary, department, performance, satisfaction, lateness, absences, and attrition status.

## Target Variable
- `Termd = 1`: employee left the company
- `Termd = 0`: employee is still active

## Responsible AI Scope
This project focuses on:
- **Ethics**: checking whether the model behaves fairly across sensitive groups such as sex and ethnicity
- **Explainability**: explaining why a given employee is predicted as high-risk

## Repository Structure
- `data/`: raw and processed datasets
- `src/`: preprocessing, training, fairness and explainability scripts
- `notebooks/`: exploration notebook
- `docs/`: technical documentation, architecture, data card, model card, executive summary
- `demo/`: demo support
- `slides/`: slide structure
- `assets/`: visual outputs

## Main Deliverables
- README
- Technical Documentation
- Architecture Scheme
- Data Card
- Model Card
- Executive Summary
- Demo
- Slides

## How to Run
1. Install dependencies
2. Place the dataset in `data/raw/HRDataset_v14.csv`
3. Run preprocessing
4. Train the model
5. Run fairness analysis
6. Run explainability analysis

## Team Members
- Add names here

## Important Note
This solution is a **decision-support tool** for HR teams. It must not be used as a fully automated decision-maker.