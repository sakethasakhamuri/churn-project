# Churn Project

Customer churn prediction and segmentation-based retention strategy using
machine learning and explainable AI (MSc Data Science, INTOL500).

## Structure
- `data/raw/` original Kaggle files (never edited, not committed)
- `data/processed/` cleaned data produced by code
- `notebooks/` 01_data_cleaning, 02_eda, 03_modelling, 04_explainability_shap, 05_segmentation
- `src/` reusable code shared by both datasets
- `models/`, `results/`, `figures/` outputs
- `reports/` literature, drafts, final
- `notes/decisions_log.md` every methodological decision and why

## Datasets (download separately from Kaggle into data/raw/)
- Telco Customer Churn -> `data/raw/telco_churn.csv`
- Bank Customer Churn / Churn Modelling -> `data/raw/bank_churn.csv`

## Setup
```
python -m venv .venv
.venv\Scripts\activate        # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```
