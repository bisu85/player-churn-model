from player_churn_model.db import load_features
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn


FEATURE_QUERY = """
SELECT
    player_id,
    lifetime_events,
    lifetime_revenue,
    player_segment,
    EXTRACT(EPOCH FROM (last_seen_ts - first_seen_ts)) / 86400.0 AS tenure_days,
    CASE
        WHEN last_seen_ts < (SELECT MAX(last_seen_ts) FROM public.dim_player) - INTERVAL '14 days'
        THEN 1 ELSE 0
    END AS churned
FROM public.dim_player
"""


def main() -> None:
    df = load_features(FEATURE_QUERY)
    print("Shape:", df.shape)
    print("\nColumn types:")
    print(df.dtypes)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nChurn balance:")
    print(df["churned"].value_counts())

    # --- Piece 2: build X (features) and y (target) ---
    y = df["churned"]
    X = df.drop(columns=["player_id", "churned"])

    # Convert the text 'player_segment' column into numeric 0/1 columns.
    X = pd.get_dummies(X, columns=["player_segment"])

    print("\n--- X and y built ---")
    print("X shape:", X.shape)
    print("X columns:", list(X.columns))
    print("y shape:", y.shape)
    print("\nX first 3 rows:")
    print(X.head(3))

    # --- Piece 3: train the baseline model ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    print("\n--- data split ---")
    print("train rows:", len(X_train), "| test rows:", len(X_test))

    model = DummyClassifier(strategy="stratified", random_state=42)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    print("\n--- baseline result ---")
    print(f"ROC-AUC: {auc:.4f}")

    # --- Piece 3 + 4: train the baseline and track it in MLflow ---
    mlflow.set_experiment("player-churn")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    print("\n--- data split ---")
    print("train rows:", len(X_train), "| test rows:", len(X_test))

    with mlflow.start_run(run_name="baseline-dummy"):
        model = DummyClassifier(strategy="stratified", random_state=42)
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)

        # Log the settings (params), the result (metric), and the model itself.
        mlflow.log_param("model_type", "DummyClassifier")
        mlflow.log_param("strategy", "stratified")
        mlflow.log_param("test_size", 0.25)
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(model, name="model")

        print("\n--- baseline result (logged to MLflow) ---")
        print(f"ROC-AUC: {auc:.4f}")


if __name__ == "__main__":
    main()