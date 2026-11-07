# Decisions log

## 1. Data cleaning (notebooks/01_data_cleaning.ipynb)
- Raw files in data/raw are never edited; cleaned versions go to data/processed.
- Telco TotalCharges: stored as text. 11 blanks, all with tenure = 0 and none churned
  (new customers not yet billed). Converted to numeric and filled with 0, the true
  amount billed so far. Alternatives: drop the 11 rows (0.16%); mean imputation
  rejected because it would invent a charge.
- Merged "No internet service" / "No phone service" into "No" in 7 Telco columns.
  Crosstab showed these labels occur only when InternetService = No (1,526 rows) or
  PhoneService = No (682 rows), so they duplicate another column. Keeping them would
  cause collinearity and split SHAP credit.
- Dropped identifiers: customerID (Telco); RowNumber, CustomerId, Surname (bank).
- Target encoded 1 = left, 0 = stayed, named "churn" in both datasets.
- Shapes after cleaning: Telco 7,043 x 20 (churn 26.5%); bank 10,000 x 11 (churn 20.4%).
- Bank EstimatedSalary is uncorrelated with everything, so it is not used for cost
  assumptions (limitation).

## 2. Train/test split
- 80/20 stratified, random_state = 42, done before modelling. Test set not used until
  final evaluation; model choices use 5-fold CV on training data only.

## 3. Metrics
- PR-AUC is the main metric, with precision, recall, F1, ROC-AUC. Accuracy is
  misleading under imbalance (bank: always predicting "stays" scores 79.6%).

## 4. Model comparison (5-fold CV on training data)
- Models: Logistic Regression (baseline), Random Forest, XGBoost, LightGBM, SVM (RBF,
  no probability output, comparison only).
- Telco PR-AUC: XGBoost 0.666, LogReg 0.661, RF 0.659, LightGBM 0.656, SVM 0.637.
  The top four are within noise; complex models add little.
- Bank PR-AUC: LightGBM 0.702, XGBoost 0.701, RF 0.684, SVM 0.657, LogReg 0.469.
  Boosting helps where relationships are non-linear (age, number of products).
- Decision: XGBoost is the lead model for cost, SHAP and DiCE (tied-best on both
  datasets, well supported by SHAP and DiCE).

## 5. Imbalance handling
- Compared class weights, SMOTE and random undersampling, applied only inside the
  training part of each CV fold (imblearn pipeline) to avoid leakage.
- Recall rose a lot (bank XGBoost 0.478 -> 0.756 with undersampling) while precision
  fell (0.753 -> 0.483). ROC-AUC and PR-AUC stayed roughly flat, so these methods
  mostly shift the decision threshold. Undersampling slightly lowered PR-AUC.
- Decision: use unweighted XGBoost and choose the threshold by cost, because the cost
  calculation needs honest probabilities. Class-weighted and SMOTE versions are kept
  as comparison rows.

## 6. Cost-benefit analysis (labelled assumptions; no real campaign data)
- Telco: V = 777 (12 x mean monthly charge 64.76), offer = 78 (10% of V), success 30%.
- Bank: illustrative V = 1000, offer = 100, success 30%.
- Costs: TP = offer + (1 - s) V; FP = offer; FN = V; TN = 0.
- Break-even threshold = offer / (s x V), about 0.33.
- Out-of-fold XGBoost probabilities: best threshold Telco 0.36 (188.1 per customer),
  bank 0.32 (185.7). At 0.5: 190.2 / 187.4. Flag nobody: 206.2 / 203.8. Flag
  everybody: 222.3 / 242.6. Saving vs flag nobody about 8.8%; vs 0.5 about 1%.
- Cost curves are flat near the minimum (about 0.25-0.45).

## 7. Sensitivity analysis
- Varied success rate (10-50%) and offer cost (5-20% of V).
- Best threshold is close to offer / (s x V), ranging from 0.05 to 0.95. Saving vs flag
  nobody ranges from 0% (offers costly or ineffective) to about 35% (cheap, effective).
- Conclusion: the value of the model depends on the real offer success rate, which
  should be measured with a pilot.
- Limitation: Telco and bank use the same cost ratios, so their similarity is by
  construction.

## 8. Pending decisions
- XGBoost tuning settings; final test-set result; number of clusters (k); DiCE
  actionable features; retention strategy mapping.

## 9. Tuning (5.5)
- Random search, 25 settings x 5-fold CV, scored by PR-AUC, on training data only.
- Telco PR-AUC 0.666 -> 0.671 (+0.005); bank 0.701 -> 0.700. Cost per customer
  changed by under 0.5%. Best thresholds unchanged (0.36 Telco, 0.32 bank).
- Decision: keep original XGBoost settings (gain within noise). One fixed setup for
  both datasets also keeps SHAP and DiCE consistent.
- Tuned settings saved in results/xgb_best_params_*.json for reference.

## 10. Final test-set evaluation (run once)
- XGBoost (original settings), thresholds from training CV (Telco 0.36, bank 0.32).
- Telco: precision 0.569, recall 0.703, F1 0.629, ROC-AUC 0.842, PR-AUC 0.654.
- Bank: precision 0.616, recall 0.641, F1 0.628, ROC-AUC 0.866, PR-AUC 0.718.
- Cost per customer vs flag nobody: Telco 188.3 vs 206.2 (8.7% saving), bank 185.6 vs
  203.5 (8.8%). Test scores close to CV, so no sign of overfitting.
- Test set not used again for any decision.

## 11. Choosing k for SHAP-based K-Means
- Rule set in advance: highest silhouette among k = 3-6 with stability ARI above 0.8;
  smaller k if close; k = 2 excluded as too coarse for distinct actions.
- Bank: k = 4 (silhouette about 0.36, tied with k = 5; elbow bend at 4).
- Telco: silhouettes low (0.23-0.26), weak cluster structure. Rule gives k = 3; chose
  k = 6 for more granular retention actions (differences among k = 3-6 are small),
  with k = 3 checked as a sensitivity run.
- Sensitivity run for Telco with k = 3: segments of 2,335 / 1,581 / 1,718 customers with
  churn of 5.7% / 44.0% / 38.9% (spread 0.383 vs 0.654 for k = 6; ARI between the two
  groupings 0.457). k = 3 puts the new, very high-risk customers (67% churn in k = 6) in the
  same group as longer-tenure customers at about 38-44%, so k = 6 is retained for its finer
  separation of risk and retention actions. Because silhouette is low (about 0.23-0.26), the
  segments are a practical partition of a continuum, not sharply separated natural clusters.

## 12. SHAP-based segments and retention mapping
- Telco k = 6: churn 1.7% to 67.1%; riskiest are new month-to-month fiber customers
  paying by electronic check (seg 4, 67%) and expensive long-tenure month-to-month
  fiber customers (seg 2, 38%).
- Bank k = 4: churn 4.7% to 85.5%; riskiest are customers with 3+ products (seg 2,
  85.5%, n = 256) and older single-product customers (seg 0, 43.5%).
- SHAP-based segments separate churn far better than raw-feature clustering (spread
  Telco 0.654 vs 0.498, bank 0.808 vs 0.191; ARI 0.23 and 0.10, so the groupings
  differ). Caveat: SHAP is risk-aware by construction.
- Retention actions are targeted at segments whose average churn exceeds the
  break-even (about 0.33). Actions are hypotheses to be tested in a pilot.
- Limitations: bank seg 2 is small; gender and geography appear as drivers
  (fairness); associations are not causes.

## 13. Counterfactuals with DiCE (design choices)
- Tool: DiCE (dice-ml), random-search method, 3 counterfactuals per customer, applied to
  the unweighted XGBoost model.
- Threshold alignment: a wrapper rescales predicted probabilities so that the cost-optimal
  threshold (Telco 0.36, bank 0.32) acts as DiCE's decision boundary. The wrapper also
  converts categorical values back to their original data types.
- Actionable features (allowed to change): Telco - contract, payment method, paperless
  billing, add-on services (online security, backup, device protection, tech support,
  streaming), and monthly charge limited to at most a 20% discount per customer; bank -
  number of products, active-member status, credit-card holding.
- Fixed features: demographics, tenure, country, balance, credit score, salary, total
  charges.
- Customers explained: one typical flagged customer (middle of the risk range) from each of
  the two riskiest SHAP segments per dataset.
- Limitations: counterfactuals show what the model responds to, not what real customers will
  do; monthly charge is changed independently of the services chosen; the bank has only three
  actionable levers; DiCE's random method does not guarantee minimal changes and can
  return duplicate or redundant options.

## 14. Counterfactual results (DiCE)
- First Telco run allowed the monthly charge to vary freely (18-120), producing unrealistic
  price cuts of 50-77% and a price increase. The final run limits it to at most a 20% discount.
- Telco, segment 4 (customer 5037, risk 0.70, new fiber customer, month-to-month): in the saved
  run the options are a two-year contract with the streaming TV service switched off, a one-year
  contract with automatic credit-card payment, and a two-year contract with a small discount
  (about 2%). An earlier run of the same search also found a one-year contract alone and a
  two-year contract with a discount of about 19%.
- Telco, segment 2 (customer 5750, risk 0.51, established month-to-month fiber customer): in the
  saved run all three options switch to a one-year contract, combined with added tech support,
  added streaming movies, or a discount of about 13%. An earlier run found a two-year contract
  with a discount of about 9% and a one-year contract with automatic credit-card payment.
- The options differ between runs because DiCE's random search is not seeded; options that
  switch a service off (such as streaming TV) are model artefacts, not recommendations. What is
  stable across runs is the contract: every Telco option moves the customer to a one- or two-year
  contract.
- Main lever in Telco (stable across both runs): contract commitment, consistent with SHAP (contract is the top driver)
  and EDA (month-to-month churn 42.7%). Automatic payment is a secondary lever.
- Bank, segment 0 (customer 5341, 1 product, inactive, risk 0.63): becoming an active member,
  with a credit card or a second product. Bank segment 2 (customer 2210, 3 products, risk
  0.95): reducing to 1-2 products flips the prediction. This is read as a signal about the
  experience of multi-product customers, not as a recommendation to remove products.
- Limitations: "becoming active" is an outcome that needs its own actions; only four customers
  were analysed in detail; the results are associations learned by the model.

## 15. Literature review method
- Searched for sources on churn prediction (telecom and banking), class imbalance and
  metrics, cost-sensitive evaluation, SHAP, counterfactual explanations, and segmentation
  with explanations. Each reference's title, authors, venue, year, pages and DOI were
  checked against publisher, arXiv or index pages before inclusion; 11 references retained.
- Closest prior work: El Attar and El-Hajj (2026), SHAP with autoencoder-based segmentation
  on telecom data; Lundberg et al. (2020), combining local explanations to find subgroups.
- Novelty claim is limited to the combination (out-of-fold SHAP clustering, comparison with
  raw-feature clustering, link to cost-optimal thresholds and counterfactuals, validation on
  telecom and banking data) and reflects a limited search, not an exhaustive review.

## 16. Guide feedback and scope extension
- Guide reviewed the methodology and results and recommended four additions: (1) CatBoost
  and EBM in the model comparison, (2) survival analysis on Telco (Cox, Random Survival
  Forest, XGBoost-AFT; concordance index and integrated Brier score; customer-specific
  lifetime value in place of the fixed value V = 777 in the cost matrix), (3) error analysis
  of false negatives, (4) a Streamlit demo.
- Survival design notes: tenure is the time variable and churn the event (non-churners are
  right-censored); tenure and TotalCharges are removed from the predictors because
  TotalCharges is a function of tenure; customers with tenure 0 need a small positive time.
- DeepSurv is optional because it requires PyTorch.

## 17. Error analysis: false negatives (Telco, out-of-fold, threshold 0.36)
- Of 1,495 churners in the training data, the model caught 1,030 (recall 0.689) and missed 465.
- Misses are concentrated outside the highest-risk segment: segment 4 (new month-to-month
  fiber customers) is almost fully caught (1.1% missed), while 32.6% of churners in segment 2,
  62.9% in segment 0 and 93.7% in segment 5 are missed. Segments 1 and 3 have very few
  churners (18 and 24), so their 100% miss rates are not reliable.
- Missed churners are long-tenure customers (mean 32.6 months vs 11.9 for caught churners),
  with slightly lower monthly charges, and 93.8% of one-year-contract churners are missed
  (22.9% of month-to-month churners).
- The mean predicted probability of missed churners is 0.19, well below the threshold, so
  lowering the threshold would not recover them cheaply; the features available do not
  identify them.
- Hypothesis tested: that long-tenure one-year-contract churners leave at contract renewal.
  Their tenures were not concentrated near multiples of 12 months (20.8% within one month of
  a multiple of 12, against about 25% expected by chance; n = 130; two-year churners 33.3%,
  n = 39, inflated by the dataset's tenure cap at 72). The test is inconclusive, because the
  data holds the current contract, not its start date, so tenure is not time since contract
  start. Conclusion: no evidence of a renewal effect; the data cannot answer the question.
- Recommended: collect contract start and end dates, complaint and usage data, and use
  time-to-event (survival) models.
- Bank (out-of-fold, threshold 0.32): of 1,630 churners the model caught 1,036 (recall 0.636)
  and missed 594.
- Bank misses are concentrated in two segments: segment 1 (144 churners, 97.9% missed) and
  segment 3 (485 churners, 68.2% missed); segments 0 and 2 are mostly caught (14.8% and 2.7%
  missed).
- Missed bank churners are younger (mean age 38.6 vs 48.3 for caught churners), more often
  active members (0.51 vs 0.28), and hold fewer products (1.35 vs 1.54). Their mean predicted
  probability is 0.15 (caught: 0.70). Misses are higher in France (44.3%) and Spain (45.2%)
  than in Germany (24.6%), and slightly higher for men (40.9%) than for women (32.9%).
- Reading: in both datasets, missed churners look like loyal customers (long tenure in Telco;
  young and active in the bank), with predicted probabilities far below the threshold, so the
  available features cannot identify them. Lowering the threshold would not recover them cheaply.
  A real company would need additional data (complaints, usage trends, contract dates) or a
  separate low-cost, broad retention measure for these groups.
- Limitation: the miss rates for the gender and geography groups are associations, not evidence
  of unfair treatment; a formal fairness check was not carried out.

## 18. Adding CatBoost and EBM to the model comparison (5-fold CV on training data)
- Added as requested by the guide. CatBoost: 300 iterations, learning rate 0.05, depth 4, same
  preprocessing as the other models. EBM (Explainable Boosting Machine, a glass-box additive
  model): default settings on the raw table, random_state 42.
- PR-AUC, Telco: EBM 0.672, CatBoost 0.667, XGBoost 0.666. Bank: CatBoost 0.711, XGBoost 0.701,
  EBM 0.701, LightGBM 0.702.
- Rule set in advance: XGBoost remains the lead model unless another model beats it by more than
  0.01 PR-AUC on both datasets. The largest gains are +0.006 (Telco, EBM) and +0.010 (bank,
  CatBoost, exactly at the margin), and no model passes on both datasets, so XGBoost is kept.
  This also keeps the SHAP, segmentation and DiCE results valid, as they were built on XGBoost.
- EBM matches XGBoost on both datasets while being directly interpretable, which supports the
  conclusion that the churn signal is mostly additive and that complex models add little.
- Results saved in results/model_comparison_all.csv.

## 19. Guide's priorities (reply of 2 Nov)
- Model comparison and error analysis are the prime items; survival analysis is the second
  priority ("extension"); the Streamlit app is for the demo. Review dates, viva date and report
  template are still to be communicated by the guide.
- Consequence: the report presents model comparison and error analysis in the main results;
  survival analysis is written as an extension chapter; Streamlit is shown at the viva.

## 20. Survival analysis on Telco (extension; 5-fold CV on training data)
- Setup: time = tenure in months (the 8 training customers with tenure 0 were given 0.5 months),
  event = churn, non-churners are right-censored. Tenure and TotalCharges were removed from the
  predictors, since tenure is now the time variable and TotalCharges is roughly tenure times
  monthly charge. 17 predictors remain; 1,495 events among 5,634 training customers; median time
  to churn among churners 10 months; maximum observed time 72 (the dataset's cap).
- Models: Cox proportional hazards (ridge penalty 0.1), Random Survival Forest (200 trees,
  minimum leaf 15), XGBoost-AFT (normal distribution, scale 1.2, 300 rounds, depth 3). DeepSurv
  was not run (requires PyTorch; optional in the guide's request).
- Concordance index: Cox 0.867 (sd 0.003), XGBoost-AFT 0.865 (0.004), RSF 0.852 (0.004).
  Cox and AFT are tied within fold-to-fold variation; RSF is slightly lower. As in the
  classification comparison, the simple model matches the complex ones.
- Integrated Brier score (months 3-60; lower is better): no features (Kaplan-Meier) 0.175,
  Cox 0.084, RSF 0.092, XGBoost-AFT 0.082. All three models roughly halve the error of the
  no-features baseline.
- Limitations: the proportional-hazards assumption of the Cox model was not tested; the AFT
  survival curve assumes a normal distribution of log lifetime with a fixed scale; tenure is
  capped at 72; the test set was not used for the survival models.

## 21. Customer-specific lifetime value and cost matrix
- Definition: V_i = monthly charge x expected number of active months in the next 12, from the
  out-of-fold Cox survival curve, conditional on the customer's tenure so far. A 12-month
  horizon keeps V_i comparable with the fixed V = 777 (12 x mean monthly charge).
- Result: mean V_i = 688.8 (range 41 to 1,425) against the fixed 777; mean expected months 10.74
  of 12. Churners have a higher mean V_i (741) than stayers (670) because they pay higher bills.
- Cost rerun (offer 78, success rate 30%, out-of-fold XGBoost probabilities, cost per customer):
  fixed V reproduces the earlier numbers (flag nobody 206.18; 188.11 at threshold 0.36).
  With V_i: flag nobody 196.74; best single threshold 0.41 gives 179.60 (8.7% saving); the
  customer-specific rule (flag if probability > offer / (success rate x V_i)) gives 178.32
  (9.4% saving) while flagging fewer customers (1,673 against 1,897 under fixed V).
- Reading: the customer-specific rule skips low-value customers for whom the 78 offer is not
  worth it and flags high-value customers at lower risk. The rule has no tuned parameter, while
  the single threshold was tuned on the same data. The gain over the best single threshold is
  small (about 1.3 per customer, 0.7 percentage points).
- Limitations: V_i is an estimate and is revenue, not profit; the offer cost and success rate
  are assumptions; results are out-of-fold on training data.

## 22. Streamlit demo (app/streamlit_app.py)
- Purpose: a working demonstration (guide's request; presented at the viva). A user picks one
  of eight example customers or edits any value, and sees the churn probability against the
  cost-optimal threshold, the SHAP explanation, the assigned segment, a recommended action and
  DiCE counterfactual options.
- Build: app/build_app_artifacts.py refits the unweighted XGBoost model on the training data,
  rebuilds the SHAP-based K-Means segments (reproduced exactly: ARI 1.0 against the saved
  segments) and, for Telco, fits the Cox model used for customer value. The saved models are not
  committed and are rebuilt with one command.
- Recommended action: offer if the churn probability exceeds the break-even offer / (success rate
  x V). For Telco V is customer-specific (survival-based); for the bank it is the assumed
  V = 1,000, so break-even is fixed at 0.33 because the bank data has no tenure-to-churn timeline
  usable for survival analysis. The segment actions are hypotheses to be tested in a pilot.
- Counterfactual filter: DiCE's random search sometimes proposes changes that remove something
  (becoming inactive, dropping the credit card, switching off an add-on service) because the
  model's risk falls. These are hidden in the demo as unrealistic retention actions; eight options
  are generated and the first three realistic ones are shown.
- Limitations: segment assignment for a new customer uses SHAP values from the model fitted on
  all training data, whereas the segments were built from out-of-fold SHAP values; the offer cost
  and success rate are assumptions; counterfactuals show what the model responds to, not what a
  real customer will do.
