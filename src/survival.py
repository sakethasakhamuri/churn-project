"""Survival helpers for the Telco extension: time/event arrays, predictors, lifetime value."""
import numpy as np
from sksurv.util import Surv

H = 12   # months ahead used for the lifetime value


def to_surv(X, y):
    """time = tenure (tenure 0 gets half a month), event = churn (non-churners are censored)."""
    return Surv.from_arrays(event=y.astype(bool), time=X["tenure"].clip(lower=0.5))


def predictors(X):
    """Tenure is the time variable and TotalCharges is roughly tenure x charge, so both are dropped."""
    return X.drop(columns=["tenure", "TotalCharges"])


def expected_months(model, X, tenure, H=H):
    """Expected number of the next H months a customer is still active, given tenure so far."""
    out = []
    for fn, t0 in zip(model.predict_survival_function(X), tenure):
        lo, hi = fn.domain
        s0 = fn(min(max(t0, lo), hi))
        s = [fn(min(t0 + k, hi)) for k in range(1, H + 1)]
        out.append(np.sum(s) / s0)
    return np.array(out)
