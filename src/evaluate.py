"""Cross-validated evaluation shared by all models."""
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

SCORING = {"precision": "precision", "recall": "recall", "f1": "f1",
           "roc_auc": "roc_auc", "pr_auc": "average_precision"}
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def cv_report(model, X, y):
    res = cross_validate(model, X, y, cv=CV, scoring=SCORING)
    return {k[5:]: round(v.mean(), 3) for k, v in res.items() if k.startswith("test_")}