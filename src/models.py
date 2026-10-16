"""Preprocessing + model pipelines, shared by both datasets."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

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