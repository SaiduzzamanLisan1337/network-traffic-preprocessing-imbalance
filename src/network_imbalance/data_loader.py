from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def load_csv(path: str | Path, target_column: str, drop_columns=None, max_rows=None):
    path = Path(path)
    df = pd.read_csv(path, nrows=max_rows)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. Columns: {list(df.columns)[:20]}")
    drop_columns = drop_columns or []
    df = df.drop(columns=[c for c in drop_columns if c in df.columns], errors="ignore")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").drop_duplicates()
    y = df[target_column].astype(str).str.strip()
    X = df.drop(columns=[target_column])
    # Keep numeric features only for a transparent baseline; categorical encoders can be added later.
    X = X.select_dtypes(include=[np.number]).copy()
    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns remain after preprocessing.")
    return X, y, {"source": str(path), "rows": len(df), "features": X.shape[1]}


def make_demo_dataset(n_samples=12000, random_state=42):
    """Create a reproducible imbalanced binary IDS-like dataset for smoke tests/demo runs."""
    weights = [0.93, 0.07]
    X, y = make_classification(
        n_samples=n_samples,
        n_features=40,
        n_informative=18,
        n_redundant=8,
        n_repeated=2,
        n_classes=2,
        weights=weights,
        class_sep=1.0,
        flip_y=0.01,
        random_state=random_state,
    )
    columns = [f"flow_feature_{i:02d}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=columns), pd.Series(np.where(y == 1, "ATTACK", "BENIGN"), name="Label"), {
        "source": "sklearn.make_classification demo",
        "rows": n_samples,
        "features": X.shape[1],
    }
