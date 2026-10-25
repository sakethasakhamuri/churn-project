# Decisions log

## 1. Data cleaning (notebooks/01_data_cleaning.ipynb)
- Raw files in data/raw are never edited; cleaned versions go to data/processed.
- Telco TotalCharges: stored as text. 11 blanks, all with tenure = 0 and none churned
  (new customers not yet billed). Converted to numeric and filled with 0, the true
  amount billed so far. Alternatives: drop the 11 rows (0.16%); mean imputation
  rejected because it would invent a charge.
- Merged "No internet service" / "No phone service" into "No" in 7 Telco columns.
  Crosstab showed these labels occur only when InternetService = No (1,526 rows) or
  PhoneService = No (682 rows), so they duplicate another column. Keeping them would
  cause collinearity and split SHAP credit.
- Dropped identifiers: customerID (Telco); RowNumber, CustomerId, Surname (bank).
- Target encoded 1 = left, 0 = stayed, named "churn" in both datasets.
- Shapes after cleaning: Telco 7,043 x 20 (churn 26.5%); bank 10,000 x 11 (churn 20.4%).
- Bank EstimatedSalary is uncorrelated with everything, so it is not used for cost
  assumptions (limitation).

## 2. Train/test split
- 80/20 stratified, random_state = 42, done before modelling. Test set not used until
  final evaluation; model choices use 5-fold CV on training data only.

## 3. Metrics
- PR-AUC is the main metric, with precision, recall, F1, ROC-AUC. Accuracy is
  misleading under imbalance (bank: always predicting "stays" scores 79.6%).

## 4. Model comparison (5-fold CV on training data)
- Models: Logistic Regression (baseline), Random Forest, XGBoost, LightGBM, SVM (RBF,
  no probability output, comparison only).
- Telco PR-AUC: XGBoost 0.666, LogReg 0.661, RF 0.659, LightGBM 0.656, SVM 0.637.
  The top four are within noise; complex models add little.
- Bank PR-AUC: LightGBM 0.702, XGBoost 0.701, RF 0.684, SVM 0.657, LogReg 0.469.
  Boosting helps where relationships are non-linear (age, number of products).
- Decision: XGBoost is the lead model for cost, SHAP and DiCE (tied-best on both
  datasets, well supported by SHAP and DiCE).

## 5. Imbalance handling
- Compared class weights, SMOTE and random undersampling, applied only inside the
  training part of each CV fold (imblearn pipeline) to avoid leakage.
- Recall rose a lot (bank XGBoost 0.478 -> 0.756 with undersampling) while precision
  fell (0.753 -> 0.483). ROC-AUC and PR-AUC stayed roughly flat, so these methods
  mostly shift the decision threshold. Undersampling slightly lowered PR-AUC.
- Decision: use unweighted XGBoost and choose the threshold by cost, because the cost
  calculation needs honest probabilities. Class-weighted and SMOTE versions are kept
  as comparison rows.

## 6. Cost-benefit analysis (labelled assumptions; no real campaign data)
- Telco: V = 777 (12 x mean monthly charge 64.76), offer = 78 (10% of V), success 30%.
- Bank: illustrative V = 1000, offer = 100, success 30%.
- Costs: TP = offer + (1 - s) V; FP = offer; FN = V; TN = 0.
- Break-even threshold = offer / (s x V), about 0.33.
- Out-of-fold XGBoost probabilities: best threshold Telco 0.36 (188.1 per customer),
  bank 0.32 (185.7). At 0.5: 190.2 / 187.4. Flag nobody: 206.2 / 203.8. Flag
  everybody: 222.3 / 242.6. Saving vs flag nobody about 8.8%; vs 0.5 about 1%.
- Cost curves are flat near the minimum (about 0.25-0.45).

## 7. Sensitivity analysis
- Varied success rate (10-50%) and offer cost (5-20% of V).
- Best threshold is close to offer / (s x V), ranging from 0.05 to 0.95. Saving vs flag
  nobody ranges from 0% (offers costly or ineffective) to about 35% (cheap, effective).
- Conclusion: the value of the model depends on the real offer success rate, which
  should be measured with a pilot.
- Limitation: Telco and bank use the same cost ratios, so their similarity is by
  construction.

## 8. Pending decisions
- XGBoost tuning settings; final test-set result; number of clusters (k); DiCE
  actionable features; retention strategy mapping.