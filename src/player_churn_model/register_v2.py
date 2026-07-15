import mlflow
import mlflow.sklearn

from player_churn_model.features import build_features
from player_churn_model.train import build_pipeline

mlflow.set_tracking_uri("http://127.0.0.1:5000")
MODEL_NAME = "player-churn"


def main() -> None:
    X, y = build_features()          # ALL players, no split — full-data model

    mlflow.set_experiment("player-churn-registry")

    with mlflow.start_run(run_name="logreg-full-data") as run:
        pipeline = build_pipeline()
        pipeline.fit(X, y)           # trained on everything

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("trained_on", "full_dataset")

        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name=MODEL_NAME,   # existing name -> becomes version 2
        )
        print(f"Run {run.info.run_id} logged and registered as a new version.")


if __name__ == "__main__":
    main()