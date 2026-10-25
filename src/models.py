"""Preprocessing + model pipelines, shared by both datasets."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.svm import SVC

def make_preprocessor(X):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary",
                              sparse_output=False), cat_cols),
    ])

def make_logreg(X, class_weight=None):
    return Pipeline([
        ("prep", make_preprocessor(X)),
        ("model", LogisticRegression(max_iter=1000, class_weight=class_weight)),
    ])

def make_rf(X, class_weight=None):
    return Pipeline([
        ("prep", make_preprocessor(X)),
        ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                         class_weight=class_weight,
                                         random_state=42, n_jobs=-1)),
    ])

def make_xgb(X, scale_pos_weight=1):
    return Pipeline([
        ("prep", make_preprocessor(X)),
        ("model", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                subsample=0.8, colsample_bytree=0.8,
                                scale_pos_weight=scale_pos_weight,
                                eval_metric="logloss", random_state=42, n_jobs=-1)),
    ])

def make_balanced(kind, X, y, strategy):
    """kind: 'logreg' or 'xgb'.  strategy: 'none', 'class_weight', 'smote', 'undersample'."""
    ratio = (y == 0).sum() / (y == 1).sum()
    if kind == "logreg":
        clf = LogisticRegression(max_iter=1000,
                                 class_weight="balanced" if strategy == "class_weight" else None)
    else:
        clf = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                            subsample=0.8, colsample_bytree=0.8,
                            scale_pos_weight=ratio if strategy == "class_weight" else 1,
                            eval_metric="logloss", random_state=42, n_jobs=-1)
    steps = [("prep", make_preprocessor(X))]
    if strategy == "smote":
        steps.append(("resample", SMOTE(random_state=42)))
    if strategy == "undersample":
        steps.append(("resample", RandomUnderSampler(random_state=42)))
    steps.append(("model", clf))
    return ImbPipeline(steps)

def make_lgbm(X):
    return Pipeline([
        ("prep", make_preprocessor(X)),
        ("model", LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                 random_state=42, n_jobs=-1, verbose=-1)),
    ])

def make_svm(X):
    return Pipeline([
        ("prep", make_preprocessor(X)),
        ("model", SVC(kernel="rbf", C=1.0, gamma="scale")),
    ])

def make_xgb_tuned(X, params):
    base = dict(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=-1)
    base.update(params)
    return Pipeline([("prep", make_preprocessor(X)), ("model", XGBClassifier(**base))])