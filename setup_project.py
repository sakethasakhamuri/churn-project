"""
setup_project.py - run ONCE from inside the churn-project/ folder:

    python setup_project.py

Safe to re-run: it never overwrites files that already exist
(except it appends missing lines to .gitignore and replaces the
starter README if it is still the default one).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- folders
FOLDERS = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "models",
    "figures",
    "results",
    "app",
    "reports/literature",
    "reports/drafts",
    "reports/final",
    "notes",
]

# ---------------------------------------------------------------- files
FILES = {}

FILES["requirements.txt"] = """\
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
lightgbm
shap
dice-ml
matplotlib
seaborn
jupyter
joblib
"""

FILES["src/__init__.py"] = ""

FILES["src/data.py"] = '''\
"""Loading and cleaning. One function per dataset, same output contract:
returns a DataFrame of features plus a binary target column named "churn" (1 = churned).
"""
'''

FILES["src/models.py"] = '''\
"""Model definitions and leakage-safe pipelines (preprocessing + optional
resampling live INSIDE the pipeline so they are fit on training folds only).
"""
'''

FILES["src/evaluate.py"] = '''\
"""Metrics (precision, recall, F1, ROC-AUC, PR-AUC) and the cost-benefit
matrix: expected cost vs decision threshold, and sensitivity analysis.
"""
'''

FILES["src/explain.py"] = '''\
"""SHAP (global + out-of-fold per-customer values) and DiCE counterfactuals
restricted to actionable features.
"""
'''

FILES["src/segment.py"] = '''\
"""SHAP-to-Segment clustering, choice of k, and the raw-feature baseline
clustering used for comparison.
"""
'''

FILES["notes/decisions_log.md"] = """\
# Decisions log

Every choice I may be asked to defend in the viva: what, why, alternatives, date.

## 2026-10-10 - Topic approved with four enhancements
- Guide (Dr. Swaminathan V.) approved the topic and public Kaggle datasets.
- Added: (1) cluster on SHAP values (SHAP-to-Segment), (2) financial
  cost-benefit matrix in evaluation, (3) DiCE counterfactuals,
  (4) bank dataset is mandatory (validate on telco AND banking).
- Consequence: pipeline is written as reusable code in src/ so it runs on both datasets.

## 2026-10-10 - Imbalance handling only inside training folds
- Why: resampling before the split leaks information into validation data
  and inflates scores.

## Template for new entries
### YYYY-MM-DD - Title
- Decision:
- Why:
- Alternatives considered:
- Impact / limitation:
"""

README = """\
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
.venv\\Scripts\\activate        # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```
"""

GITIGNORE_LINES = [
    "# data and model artefacts",
    "data/raw/",
    "data/processed/",
    "models/*.pkl",
    "models/*.joblib",
    ".DS_Store",
]


def main():
    for folder in FOLDERS:
        p = ROOT / folder
        p.mkdir(parents=True, exist_ok=True)
        keep = p / ".gitkeep"
        if folder in ("data/raw", "data/processed", "models", "results",
                      "figures", "app", "reports/literature", "reports/drafts",
                      "reports/final", "notebooks") and not keep.exists():
            keep.touch()

    for rel, content in FILES.items():
        path = ROOT / rel
        if path.exists():
            print(f"skip (exists): {rel}")
        else:
            path.write_text(content, encoding="utf-8")
            print(f"created:       {rel}")

    # README: replace only the untouched starter version
    readme = ROOT / "README.md"
    if (not readme.exists()) or "Project workspace for churn analysis" in readme.read_text(encoding="utf-8"):
        readme.write_text(README, encoding="utf-8")
        print("updated:       README.md")
    else:
        print("skip (customised): README.md")

    # .gitignore: append missing lines only
    gi = ROOT / ".gitignore"
    existing = gi.read_text(encoding="utf-8") if gi.exists() else ""
    present = {line.strip() for line in existing.splitlines()}
    missing = [l for l in GITIGNORE_LINES if l.strip() not in present]
    if missing:
        with gi.open("a", encoding="utf-8") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            f.write("\n".join(missing) + "\n")
        print(f"updated:       .gitignore (+{len(missing)} lines)")

    print("\nDone. Next: put the two Kaggle CSVs in data/raw/ and run:")
    print("  git add . && git commit -m 'Add project structure' && git push")


if __name__ == "__main__":
    main()