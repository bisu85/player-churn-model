import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from player_churn_model.features import build_features

MODEL_PATH = Path("models/churn_model.joblib")


def build_pipeline() -> Pipeline:
    """A single object that turns raw features into a prediction."""
    # Numeric columns get scaled; the categorical column gets one-hot encoded.
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(),
             ["events_day1", "purchases_day1", "levels_day1", "max_level_day1"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"),
             ["player_segment"]),
        ]
    )
    return Pipeline([
        ("preprocess", preprocess),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def main() -> None:
    X, y = build_features()
    print(f"Training on {len(X)} players, {X.shape[1]} raw features")

    pipeline = build_pipeline()
    pipeline.fit(X, y)          # train on ALL data — this is the production artifact

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()