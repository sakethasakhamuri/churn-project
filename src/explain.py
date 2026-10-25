"""SHAP utilities: out-of-fold SHAP values (leakage-safe)."""
import numpy as np, pandas as pd, shap
from sklearn.base import clone

def oof_shap(model, X, y, cv):
    shap_vals = np.zeros((len(X), 0))
    for tr, va in cv.split(X, y):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        Xv = m.named_steps["prep"].transform(X.iloc[va])
        sv = shap.TreeExplainer(m.named_steps["model"]).shap_values(Xv)
        if shap_vals.shape[1] == 0:
            shap_vals = np.zeros((len(X), sv.shape[1]))
        shap_vals[va] = sv
    full = clone(model).fit(X, y)
    names = full.named_steps["prep"].get_feature_names_out()
    Xt = pd.DataFrame(full.named_steps["prep"].transform(X), columns=names, index=X.index)
    return shap_vals, Xt

class ThresholdedModel:
    """Wraps a fitted pipeline so that our cost-optimal threshold becomes the 0.5 cut-off
    that DiCE uses. Also converts columns back to their original types, because DiCE
    passes categorical values as text."""
    def __init__(self, pipe, threshold, dtypes=None):
        self.pipe, self.t, self.dtypes = pipe, threshold, dtypes
    def _fix(self, X):
        if self.dtypes is not None:
            X = X[list(self.dtypes.index)].astype(self.dtypes.to_dict())
        return X
    def predict_proba(self, X):
        p = self.pipe.predict_proba(self._fix(X))[:, 1]
        odds = (p / (1 - p + 1e-12)) / (self.t / (1 - self.t))
        p2 = odds / (1 + odds)
        return np.column_stack([1 - p2, p2])
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)