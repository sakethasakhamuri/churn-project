# EDA findings

## Telco (churn 26.5%)
- By contract: month-to-month 42.7%, one year 11.3%, two year 2.8%.
- By tenure: 0-12 months 47%, 13-24 29%, 25-48 20%, 49-72 9.5%. Median tenure of
  leavers 10 months vs 38 for stayers. Customers are most at risk in their first year.
- Correlations: tenure-TotalCharges 0.83 (collinearity), tenure-churn -0.35,
  MonthlyCharges-churn +0.19.

## Bank (churn 20.4%)
- Geography: Germany 32% vs France / Spain about 16-17%.
- Gender: female 25% vs male 17% (association, not cause; handle carefully for fairness).
- Number of products: 1 -> 28%, 2 -> 8%, 3 -> 83% (n = 266), 4 -> 100% (n = 60).
  Small groups, treat with caution.
- Active member: inactive 27% vs active 14%. HasCrCard: no effect (21% vs 20%).
- Age: 30 or under 7.5%, 31-40 12%, 41-50 34%, 51-60 56%, over 60 25% (non-linear).
- Balance: with balance 24% vs zero balance 14% (36% of customers have zero balance).
- Correlation with churn: Age 0.29, Balance 0.12; CreditScore, Tenure, EstimatedSalary
  about 0. Minimum age is 18 (no minors).