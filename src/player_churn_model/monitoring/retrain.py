# src/player_churn_model/monitoring/retrain.py
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from player_churn_model.evaluate import evaluate
from player_churn_model.features import build_features
from player_churn_model.train import build_pipeline
from player_churn_model.monitoring.promote import consider_promotion

mlflow.set_tracking_uri("http://127.0.0.1:5000")
MODEL_NAME = "player-churn"


def retrain_challenger() -> None:
    # 1. Honest score on a held-out split
    auc = evaluate()

    # 2. Refit on ALL data for the shipped artifact (honest score + max-trained model)
    X, y = build_features()
    mlflow.set_experiment("player-churn-registry")
    with mlflow.start_run(run_name="challenger-retrain") as run:
        pipeline = build_pipeline()
        pipeline.fit(X, y)
        mlflow.log_metric("roc_auc", auc)          # the held-out score, not a full-data score
        mlflow.log_param("trigger", "drift-retrain")
        mlflow.sklearn.log_model(
            pipeline, name="model", registered_model_name=MODEL_NAME,
        )

    # 3. Find the version just created, run it through the guard
    client = MlflowClient()
    latest = max(client.search_model_versions(f"name='{MODEL_NAME}'"),
                 key=lambda v: int(v.version))
    consider_promotion(latest.version, auc)


if __name__ == "__main__":
    retrain_challenger()