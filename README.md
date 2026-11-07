# Customer Churn Prediction and Segmentation-Based Retention Strategy

Using Machine Learning and Explainable AI. MSc Data Science (Semester IV), INTOL500 Project
Work and Viva-Voce, SASTRA Deemed University.

Author: Saketha Sakhamuri. Guide: Dr. V. Swaminathan.

## What the project does
Predicts which customers are likely to leave, explains why, groups them by the reasons behind
their predicted risk, and recommends a retention action for each group. Two public datasets are
used so the framework is checked in two domains: telecom (Telco) and banking.

1. **Prediction:** Logistic Regression, Random Forest, XGBoost, LightGBM, SVM, CatBoost and EBM,
   compared with 5-fold cross-validation (main metric: PR-AUC). XGBoost is the lead model.
2. **Cost-benefit evaluation:** the decision threshold is chosen by expected cost, with a
   sensitivity analysis on the (assumed) cost figures.
3. **Explanation (SHAP):** out-of-fold SHAP values, aggregated per original feature.
4. **SHAP-to-Segment clustering:** K-Means on SHAP values, compared with clustering on raw
   features; segments are mapped to retention actions.
5. **Counterfactuals (DiCE):** the smallest actionable changes that would flip a churn prediction.
6. **Error analysis:** profile of the churners the model misses (false negatives).
7. **Survival analysis (Telco extension):** Cox, Random Survival Forest and XGBoost-AFT, scored
   by concordance index and integrated Brier score; gives a customer-specific lifetime value
   that replaces the fixed value in the cost matrix.
8. **Streamlit demo:** probability, SHAP explanation, segment, counterfactual and recommended
   action for one customer.

## Structure
- `data/raw/` original Kaggle files (not committed); `data/processed/` cleaned data (not committed)
- `notebooks/` run in order:
  `01_data_cleaning`, `02_eda`, `03_modelling`, `04_explainability_shap`, `05_segmentation`,
  `06_counterfactuals`, `07_error_analysis`, `08_survival`, `09_report_figures`
- `src/` reusable code shared by both datasets
  (`data`, `models`, `evaluate`, `explain`, `segment`, `survival`)
- `app/` Streamlit demo (`build_app_artifacts.py`, `streamlit_app.py`)
- `results/` result tables and arrays; `figures/` charts and demo screenshots
- `notes/decisions_log.md` every methodological decision and why; `notes/eda_findings.md`
- `reports/` literature review, drafts and the final report

## Datasets (download separately from Kaggle into `data/raw/`)
- Telco Customer Churn -> `data/raw/telco_churn.csv` (7,043 customers, 20 columns after cleaning)
- Bank Customer Churn / Churn Modelling -> `data/raw/bank_churn.csv` (10,000 customers, 11 columns
  after cleaning)

## Reproducing the results
Python 3.12 is used.
```
python -m venv .venv
.venv\Scripts\activate            # Windows (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```
Select the `.venv` kernel in Jupyter or VS Code, then run notebooks 01 to 09 in order
(Restart and Run All in each). Fixed seeds (42) are used throughout; the train/test split is
80/20 stratified and the test set is used once, for the final evaluation.

## Running the demo
From the project root, with the virtual environment active:
```
python app/build_app_artifacts.py
streamlit run app/streamlit_app.py
```
The first command refits the model and segments and saves them to `models/` (not committed). The
second opens the demo in the browser. Demo screenshots are in `figures/`.

## Key results (training-data cross-validation unless stated)
- PR-AUC, Telco: EBM 0.672, CatBoost 0.667, XGBoost 0.666, LogReg 0.661. Bank: CatBoost 0.711,
  LightGBM 0.702, XGBoost 0.701, EBM 0.701. No model beats XGBoost by more than 0.01 on both
  datasets, so XGBoost is kept for SHAP, segmentation and counterfactuals.
- Final test set (XGBoost, cost-chosen threshold): Telco PR-AUC 0.654, ROC-AUC 0.842; bank PR-AUC
  0.718, ROC-AUC 0.866. Expected cost about 8.7-8.8% below flagging nobody in both datasets.
- SHAP-based segments separate churn better than raw-feature segments (churn-rate spread Telco
  0.654 vs 0.498, bank 0.808 vs 0.191).
- Survival (Telco): concordance index Cox 0.867, XGBoost-AFT 0.865, RSF 0.852; integrated Brier
  score 0.082-0.092 against 0.175 with no features. A customer-specific rule saves 9.4%, against
  8.7% for the best single threshold.

## Limitations
No real campaign data exists, so the cost figures (offer cost, success rate, customer value for
the bank) are labelled assumptions. Results show associations learned by the models, not causes.
Segments are a practical partition of a continuum (low silhouette). Counterfactuals show what the
model responds to, not what real customers will do. See `notes/decisions_log.md` for details.

## AI assistance disclosure
This project was developed with the assistance of an AI model (Claude, by Anthropic). The AI
assistant helped plan the work, write and debug code, interpret outputs, and draft the decisions
log, the literature review and the report text. All code was run, and all results were produced
and checked, by the author on the author's own computer. References were verified against
publisher, arXiv or index pages before inclusion. The author is responsible for the content.
