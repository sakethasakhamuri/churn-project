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