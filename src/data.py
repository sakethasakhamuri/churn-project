"""Loading and splitting. Same functions work for both datasets."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    "telco": ROOT / "data/processed/telco_clean.csv",
    "bank": ROOT / "data/processed/bank_clean.csv",
}

def load_dataset(name):
    return pd.read_csv(PATHS[name])

def split_data(df, test_size=0.2, seed=42):
    X = df.drop(columns="churn")      # the inputs (features)
    y = df["churn"]                   # the answer (target)
    return train_test_split(X, y, test_size=test_size,
                            stratify=y, random_state=seed)