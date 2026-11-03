"""Build the files the Streamlit demo needs. Run once from the project root:

    python app/build_app_artifacts.py

It refits the unweighted XGBoost model on the training data, rebuilds the SHAP-based
K-Means segments (checked against results/segments_*.csv) and, for Telco, fits the Cox
model used for the customer-specific lifetime value. Output: models/app_<dataset>.joblib
(ignored by git, so rebuild it after cloning).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sksurv.linear_model import CoxPHSurvivalAnalysis

from src.data import load_dataset, split_data
from src.models import make_xgb, make_preprocessor
from src.survival import to_surv, predictors

THRESH = {"telco": 0.36, "bank": 0.32}
K = {"telco": 6, "bank": 4}

(ROOT / "models").mkdir(exist_ok=True)

for name in ["telco", "bank"]:
    X_train, X_test, y_train, y_test = split_data(load_dataset(name))
    pipe = make_xgb(X_train).fit(X_train, y_train)

    Z = pd.read_csv(ROOT / f"results/shap_by_feature_{name}.csv")
    km = KMeans(n_clusters=K[name], n_init=10, random_state=42).fit(Z)
    saved = pd.read_csv(ROOT / f"results/segments_{name}.csv", index_col=0)["shap_segment"].values
    print(name, "| segments reproduced, ARI vs saved (should be 1.0):",
          round(adjusted_rand_score(km.labels_, saved), 3),
          "| identical segment numbers:", bool(np.array_equal(km.labels_, saved)))

    art = {
        "pipe": pipe,
        "km": km,
        "z_cols": list(Z.columns),
        "threshold": THRESH[name],
        "orig_cols": list(X_train.columns),
        "dtypes": X_train.dtypes,
        "train": X_train.assign(churn=y_train),
        "profile": pd.read_csv(ROOT / f"results/segment_profile_{name}.csv", index_col=0),
    }

    # a few test-set customers across the risk range, for the demo's example picker
    pt = pd.Series(pipe.predict_proba(X_test)[:, 1], index=X_test.index).sort_values()
    pick = pt.iloc[(np.linspace(0.03, 0.97, 8) * (len(pt) - 1)).astype(int)].index
    art["examples"] = X_test.loc[pick]
    art["example_proba"] = pt.loc[pick]

    if name == "telco":
        P = predictors(X_train)
        prep2 = make_preprocessor(P)
        cox = CoxPHSurvivalAnalysis(alpha=0.1).fit(prep2.fit_transform(P), to_surv(X_train, y_train))
        art["surv_prep"], art["cox"] = prep2, cox

    joblib.dump(art, ROOT / f"models/app_{name}.joblib")
    print(name, "saved models/app_%s.joblib" % name)
