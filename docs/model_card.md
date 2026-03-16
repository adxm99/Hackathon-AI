# Model Card

## Model Purpose
The model estimates the probability that an employee may leave the company.

## Task
Binary classification:
- 1 = employee left
- 0 = employee remains active

## Inputs
Structured HR variables such as:
- salary
- department
- position
- engagement score
- satisfaction score
- absences
- lateness
- tenure
- age

## Output
A probability of attrition and a binary predicted class.

## Selected Model
We use a baseline Logistic Regression and a Random Forest classifier for comparison.

## Why These Models
- Logistic Regression is simple, interpretable, and robust.
- Random Forest can capture non-linear effects and interactions.
- This combination offers a good trade-off between performance and explainability.

## Ethical Considerations
We do not rely on the model as the sole basis for HR decisions. Sensitive attributes are mainly used for fairness auditing rather than to drive decision-making.

## Explainability
The solution provides:
- global feature importance,
- local employee-level explanations.

## Limitations
- Predictions are probabilistic, not certain
- The model may inherit patterns present in the data
- Outputs should be used by HR professionals with human oversight