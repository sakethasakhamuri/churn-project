"""SHAP-to-Segment clustering utilities."""
import numpy as np, pandas as pd
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

def aggregate_shap(sv, names, orig_cols):
    """Sum SHAP values of one-hot pieces back to their original feature."""
    groups = {}
    for j, n in enumerate(names):
        base = n.split("__", 1)[1]
        match = [c for c in orig_cols if base == c or base.startswith(c + "_")]
        groups.setdefault(max(match, key=len), []).append(j)
    return pd.DataFrame({c: sv[:, idx].sum(axis=1) for c, idx in groups.items()})

def fit_kmeans(Z, k, seed=42):
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)

def stability(Z, k, seeds=range(5)):
    labs = [fit_kmeans(Z, k, s) for s in seeds]
    return float(np.mean([adjusted_rand_score(a, b) for a, b in combinations(labs, 2)]))

def k_scan(Z, ks=range(2, 9), seed=42):
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(Z, km.labels_, sample_size=3000, random_state=seed),
                     "stability_ARI": stability(Z, k)})
    return pd.DataFrame(rows)