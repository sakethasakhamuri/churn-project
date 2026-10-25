# Decisions log

Every choice I may be asked to defend in the viva: what, why, alternatives, date.

## 2026-10-10 - Topic approved with four enhancements
- Guide (Dr. Swaminathan V.) approved the topic and public Kaggle datasets.
- Added: (1) cluster on SHAP values (SHAP-to-Segment), (2) financial cost-benefit matrix in evaluation, (3) DiCE counterfactuals,(4) bank dataset is mandatory (validate on telco AND banking).
- Consequence: pipeline is written as reusable code in src/ so it runs on both datasets.

## 2026-10-10 - Imbalance handling only inside training folds
- Why: resampling before the split leaks information into validation data and inflates scores.

## Template for new entries
### YYYY-MM-DD - Title
- Decision:
- Why:
- Alternatives considered:
- Impact / limitation:

## 2026-10-10
Decision 1: Blank TotalCharges set to 0 (Telco)

Problem: TotalCharges was stored as text, because 11 rows had a blank.
Finding: all 11 had tenure = 0 (new customers, not yet billed), and none had churned.
Decision: converted the column to numbers, and filled the blanks with 0.
Why: 0 is the true amount billed so far. This keeps all 7,043 rows. Mean imputation would invent a charge that never happened.
Alternative: drop the 11 rows (only 0.16% of the data).

Decision 2: “No internet service” / “No phone service” merged into “No”

Finding: a crosstab showed “No internet service” (1,526 rows) occurs only when InternetService = No, and “No phone service” (682 rows) only when PhoneService = No.
Decision: merged both labels into “No” in the 7 affected columns.
Why: they repeat information already in another column. Keeping them causes collinearity and splits SHAP credit across redundant features.
Alternative: keep them separate. Nothing is lost by merging, because the parent columns still record who has no service.

Decision 3: Identifier columns dropped

Telco: customerID. Bank: RowNumber, CustomerId, Surname.
Why: they’re labels, not behaviour. A model could memorise them instead of learning real patterns.

Decision 4: Target encoded as 1 = left, 0 = stayed, and named churn in both datasets

Why: models need numbers. One shared name lets the same code run on both datasets.

Decision 5: Raw data never edited

Cleaning happens in code and writes to data/processed/. This keeps the work reproducible.

Facts to record: Telco 7,043 rows, 26.5% churn. Bank 10,000 rows, 20.4% churn, no missing values or duplicates in either file after cleaning.

## Train/test split
- Decision: 80/20 stratified split, random_state=42, done before any modelling.
- Why: a held-out test set gives an honest score on customers the model has never
  seen. Stratifying keeps the churn rate the same in both parts (Telco 26.5%,
  Bank 20.4%).
- Rule: the test set is not used until final evaluation. Model choices are made
  with 5-fold cross-validation on the training data only.

## Main metric
- Decision: PR-AUC is the main metric, alongside precision, recall, F1, ROC-AUC.
- Why: churn is the minority class, so accuracy is misleading (Bank: always
  guessing "stays" scores 79.6% accuracy). PR-AUC focuses on how well the model
  finds the leavers.

## Model comparison (5-fold CV on training data)
- Models: Logistic Regression (baseline), Random Forest, XGBoost.
- Telco: all three tied (PR-AUC 0.661 / 0.659 / 0.666). Simple relationships
  (contract, tenure), so a complex model adds little.
- Bank: tree models clearly better (PR-AUC LogReg 0.469, Random Forest 0.684,
  XGBoost 0.701). Consistent with EDA: churn rises then falls with age and
  depends non-linearly on number of products, which a straight-line model
  cannot capture.
- XGBoost is best or tied-best on both datasets, so it is the lead candidate.

## Imbalance handling
- Tried: class weights, SMOTE, random undersampling, each applied only inside the
  training part of each CV fold (imblearn pipeline) to avoid leakage.
- Result: recall rose a lot (Bank XGBoost 0.478 -> 0.756 with undersampling) but
  precision fell (0.753 -> 0.483). ROC-AUC and PR-AUC stayed about the same, and
  undersampling slightly lowered PR-AUC. So these methods mostly shift the
  decision threshold rather than improve how well the model ranks customers.

## Model carried forward
- Decision: carry forward XGBoost without resampling or class weights, and choose
  the decision threshold using the cost-benefit matrix.
- Why: class weights and resampling distort the predicted probabilities, and the
  cost calculation needs honest probabilities (expected cost = probability x cost).
- Plan: keep class-weighted and SMOTE versions as comparison rows in the report.
- Limitation: differences of about 0.005 in PR-AUC are within CV noise.

## Cost-benefit assumptions (labelled assumptions, no real campaign data)
- Telco: V = 777 (12 x mean monthly charge 64.76), offer cost = 78 (10% of V),
  offer success rate s = 30%.
- Bank: illustrative units V = 1000, offer = 100, s = 30%. EstimatedSalary not used
  (uncorrelated with all other columns in EDA).
- Cost per outcome: TP = offer + (1-s)V; FP = offer; FN = V; TN = 0.
- Break-even threshold = offer / (s x V) ~ 0.33.

## Cost-optimal threshold (out-of-fold XGBoost probabilities, unweighted)
- Telco 0.36 (188.1 per customer), bank 0.32 (185.7). At 0.5: 190.2 / 187.4.
  Flag nobody: 206.2 / 203.8. Flag everybody: 222.3 / 242.6.
- Saving vs flag nobody about 8.8% on both; vs threshold 0.5 about 1%.
- Cost curves are flat near the minimum (about 0.25-0.45), so the exact threshold
  is not critical.

## Sensitivity analysis
- Varied success rate (10-50%) and offer cost (5-20% of V).
- Best threshold is close to offer/(s x V); ranges from 0.05 to 0.95.
- Saving vs flag nobody ranges from 0% (offers costly or ineffective, ratio >= 1)
  to about 35% (cheap, effective offers).
- Conclusion: the value of the model depends on the real success rate of retention
  offers, which should be measured with a pilot.
- Limitation: Telco and bank use the same cost ratios, so their similarity is by
  construction.