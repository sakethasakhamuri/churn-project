"""Cross-validated evaluation shared by all models."""
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
import numpy as np
from sklearn.model_selection import cross_val_predict

SCORING = {"precision": "precision", "recall": "recall", "f1": "f1",
           "roc_auc": "roc_auc", "pr_auc": "average_precision"}
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def cv_report(model, X, y):
    res = cross_validate(model, X, y, cv=CV, scoring=SCORING)
    return {k[5:]: round(v.mean(), 3) for k, v in res.items() if k.startswith("test_")}

def oof_proba(model, X, y):
    """Honest out-of-fold probability of churn for every training customer."""
    return cross_val_predict(model, X, y, cv=CV, method="predict_proba")[:, 1]

def confusion_counts(y, proba, threshold):
    y = np.asarray(y)
    pred = (proba >= threshold).astype(int)
    return {"TP": int(((pred == 1) & (y == 1)).sum()),
            "FP": int(((pred == 1) & (y == 0)).sum()),
            "FN": int(((pred == 0) & (y == 1)).sum()),
            "TN": int(((pred == 0) & (y == 0)).sum())}

# ---- LABELLED ASSUMPTIONS (no real campaign data; used for the cost-benefit analysis) ----
ASSUMPTIONS = {
    "telco": {"V": 777.0,  "offer": 78.0,  "s": 0.30},   # V = 12 x mean monthly charge; offer = 10% of V
    "bank":  {"V": 1000.0, "offer": 100.0, "s": 0.30},   # illustrative units
}

def total_cost(c, V, offer, s):
    return c["TP"] * (offer + (1 - s) * V) + c["FP"] * offer + c["FN"] * V

def cost_curve(y, proba, V, offer, s, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)
    rows = []
    for t in thresholds:
        c = confusion_counts(y, proba, t)
        rows.append({"threshold": round(float(t), 2), **c,
                     "total_cost": total_cost(c, V, offer, s)})
    return pd.DataFrame(rows)

def sensitivity(y, proba, V, s_list=(0.1, 0.2, 0.3, 0.4, 0.5),
                offer_fracs=(0.05, 0.10, 0.15, 0.20)):
    rows = []
    for s in s_list:
        for f in offer_fracs:
            offer = f * V
            curve = cost_curve(y, proba, V, offer, s)
            best = curve.loc[curve["total_cost"].idxmin()]
            nobody = total_cost(confusion_counts(y, proba, 1.01), V, offer, s)
            at05 = total_cost(confusion_counts(y, proba, 0.5), V, offer, s)
            rows.append({"success_rate": s, "offer_pct_of_V": f,
                         "best_threshold": best["threshold"],
                         "saving_vs_nobody_%": round(100 * (nobody - best["total_cost"]) / nobody, 1),
                         "saving_vs_0.5_%": round(100 * (at05 - best["total_cost"]) / at05, 1)})
    return pd.DataFrame(rows)