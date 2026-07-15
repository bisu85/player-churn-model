import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from player_churn_model.features import build_features
from player_churn_model.train import build_pipeline

# Point all MLflow calls at the tracking server (not local files)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_NAME = "player-churn"


def main() -> None:
    X, y = build_features()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    mlflow.set_experiment("player-churn-registry")

    with mlflow.start_run(run_name="logreg-registered") as run:
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)

        auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_param("model_type", "LogisticRegression")

        # Log AND register in one call: registered_model_name does the registration
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name=MODEL_NAME,
        )

        print(f"Run {run.info.run_id} logged. AUC={auc:.4f}")
        print(f"Registered as '{MODEL_NAME}' (a new version).")


if __name__ == "__main__":
    main()