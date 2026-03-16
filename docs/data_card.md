# Data Card

## Dataset Description
The dataset is a synthetic HR dataset containing employee demographic, professional, and performance-related information.

## Target Variable
The main target is `Termd`, indicating whether the employee left the company.

## Main Variable Families
- Demographics: age-related fields, marital status, citizenship
- Job information: department, position, salary, manager
- Performance and engagement: performance score, engagement survey, satisfaction
- Attendance: absences, lateness
- Employment events: hire date, termination information

## Sensitive Attributes
The dataset includes potentially sensitive features such as:
- `Sex`
- `RaceDesc`
- `HispanicLatino`
- `CitizenDesc` (to be treated carefully)

## Data Processing
The preprocessing pipeline:
- removes direct identifiers,
- removes leakage variables,
- converts dates,
- creates derived variables such as age and tenure,
- prepares a modeling dataset.

## Excluded Variables
The following variables are excluded from training because they directly identify employees or leak the target:
- `Employee_Name`
- `EmpID`
- `ManagerName`
- `ManagerID`
- `DateofTermination`
- `TermReason`
- `EmploymentStatus`

## Risks and Limitations
- Synthetic data may not reflect all real-world HR complexity
- Historical and structural biases may still exist
- Small sample size may limit generalization
- Sensitive variables require careful treatment in fairness analysis