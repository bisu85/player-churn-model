"""Ensure a model file exists before any test imports the API.

CI checks out from git where models/ is gitignored, so we generate a tiny
throwaway model from synthetic data. Tests only need a *valid* model to load,
not an accurate one.
"""
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("models/churn_model.joblib")


def _make_dummy_model():
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # Minimal synthetic training data matching the feature contract
    X = pd.DataFrame({
        "events_day1":     [1, 30, 5, 50, 2, 40],
        "purchases_day1":  [0, 2, 0, 5, 0, 3],
        "levels_day1":     [1, 15, 3, 25, 1, 20],
        "max_level_day1":  [1, 20, 4, 30, 2, 25],
        "player_segment":  ["free", "spender", "free", "whale", "free", "spender"],
    })
    y = [1, 0, 1, 0, 1, 0]

    preprocess = ColumnTransformer([
        ("num", StandardScaler(),
         ["events_day1", "purchases_day1", "levels_day1", "max_level_day1"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["player_segment"]),
    ])
    pipe = Pipeline([("preprocess", preprocess),
                     ("clf", LogisticRegression(max_iter=1000, random_state=42))])
    pipe.fit(X, y)
    return pipe


def pytest_configure(config):
    """Runs once before tests are collected."""
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(_make_dummy_model(), MODEL_PATH)