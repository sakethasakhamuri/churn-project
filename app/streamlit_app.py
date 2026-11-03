"""Churn retention demo. Run from the project root:

    python app/build_app_artifacts.py      (once)
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dice_ml
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.explain import ThresholdedModel
from src.segment import aggregate_shap
from src.survival import expected_months, predictors

CONT = {"telco": ["tenure", "MonthlyCharges", "TotalCharges"],
        "bank": ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]}
VARY = {"telco": ["Contract", "PaymentMethod", "PaperlessBilling", "OnlineSecurity", "OnlineBackup",
                  "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "MonthlyCharges"],
        "bank": ["NumOfProducts", "IsActiveMember", "HasCrCard"]}
# Cost assumptions (labelled assumptions, as in the report)
OFFER = {"telco": 78.0, "bank": 100.0}
SUCCESS = 0.3
BANK_V = 1000.0

# Retention actions per segment. These are hypotheses to be tested in a pilot, not proven effects.
ACTIONS = {
    "telco": {
        4: "New month-to-month fibre customers paying by electronic check. Offer a one-year contract "
           "with an incentive and encourage automatic payment; add an early onboarding call.",
        2: "Established, expensive month-to-month fibre customers. Offer a contract-commitment discount "
           "and a bundle of security and support add-ons.",
        0: "Short-tenure month-to-month DSL customers (segment average is below break-even). Offer a "
           "one-year contract only if this customer's own risk is above break-even.",
        5: "Long-tenure customers on one-year contracts. Low average risk; keep service quality high "
           "and approach at renewal if risk rises.",
        3: "Long-tenure two-year customers. Very low risk; no offer needed, protect satisfaction.",
        1: "Long-tenure two-year customers with few services. Very low risk; no offer needed.",
    },
    "bank": {
        2: "Customers with three or more products. Very high risk: review their experience with a "
           "relationship-manager call and a product-bundle review (do not simply remove products).",
        0: "Older, mostly single-product customers. Run an activity and re-engagement programme and "
           "offer a second product.",
        3: "Single-product customers with low average risk. No offer needed; monitor.",
        1: "Younger customers with about two products. Very low risk; no offer needed.",
    },
}

# Counterfactuals that take something away are not realistic retention actions, so they are not shown.
NO_REMOVE = {"telco": ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                       "StreamingTV", "StreamingMovies"],
             "bank": ["IsActiveMember", "HasCrCard"]}


def sensible(old, new, name):
    for c in NO_REMOVE[name]:
        o, n = str(old[c]), str(new[c])
        if (name == "telco" and o == "Yes" and n == "No") or (name == "bank" and o == "1" and n == "0"):
            return False
    return True


st.set_page_config(page_title="Churn retention demo", layout="wide")


@st.cache_resource
def load(name):
    return joblib.load(ROOT / f"models/app_{name}.joblib")


@st.cache_resource
def shap_explainer(name):
    return shap.TreeExplainer(load(name)["pipe"].named_steps["model"])


@st.cache_resource
def dice_explainer(name):
    art = load(name)
    model = ThresholdedModel(art["pipe"], art["threshold"], art["dtypes"])
    data = dice_ml.Data(dataframe=art["train"], continuous_features=CONT[name], outcome_name="churn")
    return dice_ml.Dice(data, dice_ml.Model(model=model, backend="sklearn"), method="random")


st.title("Customer churn: prediction, explanation and retention action")
st.caption("Demo of the project pipeline (XGBoost, SHAP, SHAP-based segments, DiCE counterfactuals, "
           "survival-based customer value). Cost figures are assumptions, not real campaign data.")

name = st.sidebar.radio("Dataset", ["telco", "bank"], format_func=lambda s: {"telco": "Telecom (Telco)", "bank": "Bank"}[s])
try:
    art = load(name)
except FileNotFoundError:
    st.error("Model files not found. Run:  python app/build_app_artifacts.py")
    st.stop()

pipe, dtypes, cols = art["pipe"], art["dtypes"], art["orig_cols"]
train = art["train"]

ex = art["examples"]
labels = [f"Customer {i} (model risk {art['example_proba'][i]:.2f})" for i in ex.index]
choice = st.sidebar.selectbox("Start from an example customer", range(len(ex)), format_func=lambda i: labels[i])
base = ex.iloc[choice]
st.sidebar.markdown("**Customer details** (edit any value)")

vals = {}
for c in cols:
    key = f"{name}_{choice}_{c}"
    if pd.api.types.is_numeric_dtype(dtypes[c]):
        lo, hi = train[c].min(), train[c].max()
        if pd.api.types.is_integer_dtype(dtypes[c]):
            vals[c] = st.sidebar.number_input(c, int(lo), int(hi), int(base[c]), step=1, key=key)
        else:
            vals[c] = st.sidebar.number_input(c, float(lo), float(hi * 1.5), float(base[c]), key=key)
    else:
        opts = sorted(train[c].astype(str).unique())
        vals[c] = st.sidebar.selectbox(c, opts, index=opts.index(str(base[c])), key=key)

q = pd.DataFrame([vals])[cols].astype(dtypes.to_dict())

# 1. Churn probability --------------------------------------------------------------------
p = float(pipe.predict_proba(q)[0, 1])
thr = art["threshold"]
c1, c2, c3 = st.columns(3)
c1.metric("Churn probability", f"{p:.0%}")
c2.metric("Cost-optimal threshold", f"{thr:.2f}")
c3.metric("Flagged as at risk", "Yes" if p >= thr else "No")

# 2. Segment -----------------------------------------------------------------------------
prep, mdl = pipe.named_steps["prep"], pipe.named_steps["model"]
sv = shap_explainer(name).shap_values(prep.transform(q))
Z = aggregate_shap(sv, list(prep.get_feature_names_out()), art["orig_cols"])[art["z_cols"]]
seg = int(art["km"].predict(Z.astype("float64"))[0])
prof = art["profile"].loc[seg]

left, right = st.columns(2)
with left:
    st.subheader("Why this prediction (SHAP)")
    s = Z.iloc[0].sort_values(key=np.abs, ascending=False).head(8)[::-1]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.barh(s.index, s.values, color=["#c0392b" if v > 0 else "#2e7d32" for v in s.values])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("pushes risk up (red) / down (green)")
    st.pyplot(fig)
with right:
    st.subheader(f"Segment {seg}")
    st.write(f"{int(prof['size']):,} training customers, average churn **{prof['churn']:.1%}**.")
    st.write(ACTIONS[name][seg])

# 3. Recommended action ---------------------------------------------------------------------
st.subheader("Recommended action")
if name == "telco":
    months = float(expected_months(art["cox"], art["surv_prep"].transform(predictors(q)), q["tenure"].values)[0])
    V = months * float(q["MonthlyCharges"].iloc[0])
else:
    months, V = None, BANK_V
breakeven = OFFER[name] / (SUCCESS * V)
if name == "telco":
    st.write(f"Expected active months in the next 12: **{months:.1f}** → customer value V = **{V:,.0f}** "
             f"(fixed V in the baseline analysis: 777).")
st.write(f"Break-even risk = offer {OFFER[name]:.0f} ÷ (success rate {SUCCESS:.0%} × V {V:,.0f}) = **{breakeven:.2f}**")
if p >= breakeven:
    st.success("Make a retention offer. " + ACTIONS[name][seg])
else:
    st.info("No offer: the expected saving is smaller than the cost of the offer for this customer.")

# 4. Counterfactual ---------------------------------------------------------------------
st.subheader("What would change the prediction (DiCE counterfactuals)")
if p < thr:
    st.write("This customer is below the threshold, so no change is needed.")
elif st.button("Generate counterfactual options"):
    rng = {}
    if name == "telco":
        price = float(q["MonthlyCharges"].iloc[0])
        rng = {"MonthlyCharges": [round(0.8 * price, 2), round(price, 2)]}   # at most a 20% discount
    else:
        rng = {"NumOfProducts": ["1", "2", "3", "4"]}
    try:
        with st.spinner("Searching for options..."):
            cf = dice_explainer(name).generate_counterfactuals(
                q, total_CFs=8, desired_class="opposite", features_to_vary=VARY[name], permitted_range=rng)
        cfdf = cf.cf_examples_list[0].final_cfs_df.drop(columns="churn")
        shown = 0
        for _, row in cfdf.reset_index(drop=True).iterrows():
            if not sensible(q.iloc[0], row, name):      # skip options that remove something good
                continue
            new = pd.DataFrame([row]).astype(dtypes.to_dict())
            changes = {c: f"{q[c].iloc[0]} → {row[c]}" for c in VARY[name] if str(row[c]) != str(q[c].iloc[0])}
            shown += 1
            st.write(f"**Option {shown}** (new risk {float(pipe.predict_proba(new[cols])[0, 1]):.0%}): "
                     + ("; ".join(f"{k}: {v}" for k, v in changes.items()) or "no change found"))
            if shown == 3:
                break
        if shown == 0:
            st.warning("Only options that remove a service or benefit were found, so none are shown.")
    except Exception as e:
        st.warning(f"No counterfactual found within the allowed changes ({type(e).__name__}).")
    st.caption("Counterfactuals show what the model responds to, not what a real customer will do.")
