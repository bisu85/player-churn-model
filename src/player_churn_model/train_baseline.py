from player_churn_model.db import load_features
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


FEATURE_QUERY = """
WITH player_first AS (
    SELECT player_id, MIN(event_ts) AS first_ts, MAX(event_ts) AS last_ts
    FROM public.fct_events
    GROUP BY player_id
),
day1 AS (
    -- behaviour in the first 24 hours only: a window nearly everyone survives,
    -- so it measures early intensity, not lifespan
    SELECT
        f.player_id,
        COUNT(*) AS events_day1,
        COUNT(*) FILTER (WHERE f.event_type = 'purchase')       AS purchases_day1,
        COUNT(*) FILTER (WHERE f.event_type = 'level_complete') AS levels_day1,
        MAX(f.level) AS max_level_day1
    FROM public.fct_events f
    JOIN player_first pf ON f.player_id = pf.player_id
    WHERE f.event_ts < pf.first_ts + INTERVAL '1 day'
    GROUP BY f.player_id
)
SELECT
    d.player_id,
    d.events_day1,
    d.purchases_day1,
    d.levels_day1,
    d.max_level_day1,
    dp.player_segment,
    CASE
        WHEN pf.last_ts < (SELECT MAX(event_ts) FROM public.fct_events) - INTERVAL '14 days'
        THEN 1 ELSE 0
    END AS churned
FROM day1 d
JOIN player_first pf      ON d.player_id = pf.player_id
JOIN public.dim_player dp ON d.player_id = dp.player_id
"""


def train_and_log(run_name, model, params, X_train, X_test, y_train, y_test):
    """Train one model inside its own MLflow run and log everything."""
    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)

        mlflow.log_params(params)
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(model, name="model")

        print(f"{run_name:>22}   ROC-AUC: {auc:.4f}")
        return auc


def main() -> None:
    df = load_features(FEATURE_QUERY)
    print("Shape:", df.shape)
    print("\nChurn balance:")
    print(df["churned"].value_counts())

    # --- build X (features) and y (target) ---
    y = df["churned"]
    X = df.drop(columns=["player_id", "churned"])
    X = pd.get_dummies(X, columns=["player_segment"])

    print("\n--- X and y built ---")
    print("X shape:", X.shape)
    print("X columns:", list(X.columns))

    # --- split once, use for every model ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    print("\n--- data split ---")
    print("train rows:", len(X_train), "| test rows:", len(X_test))

    # --- train + track each model in its own MLflow run ---
    mlflow.set_experiment("player-churn")
    print("\n--- training runs ---")

    train_and_log(
        "baseline-dummy",
        DummyClassifier(strategy="stratified", random_state=42),
        {"model_type": "DummyClassifier", "strategy": "stratified", "test_size": 0.25},
        X_train, X_test, y_train, y_test,
    )

    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    train_and_log(
        "logistic-regression",
        logreg,
        {"model_type": "LogisticRegression", "scaled": True, "test_size": 0.25},
        X_train, X_test, y_train, y_test,
    )
    train_and_log(
        "random-forest",
        RandomForestClassifier(n_estimators=200, random_state=42),
        {"model_type": "RandomForest", "n_estimators": 200, "test_size": 0.25},
        X_train, X_test, y_train, y_test,
    )


if __name__ == "__main__":
    main()