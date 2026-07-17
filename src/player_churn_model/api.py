from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from pathlib import Path
from player_churn_model.monitoring.predict_log import log_prediction
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram


app = FastAPI(title="Player Churn API")


class PlayerFeatures(BaseModel):
    """The features required to predict churn for one player — the train/serve contract."""
    events_day1: int = Field(ge=0, description="Total events in the player's first 24h")
    purchases_day1: int = Field(ge=0, description="Purchases in the first 24h")
    levels_day1: int = Field(ge=0, description="Levels completed in the first 24h")
    max_level_day1: int = Field(ge=0, description="Highest level reached in the first 24h")
    player_segment: str = Field(description="Player segment: free / spender / whale")

# Load the model ONCE, when the app starts — not on every request.
MODEL_PATH = Path("models/churn_model.joblib")
model = joblib.load(MODEL_PATH)


@app.post("/predict")
def predict(features: PlayerFeatures):
    """Predict churn for a single player."""
    row = pd.DataFrame([features.model_dump()])
    proba = float(model.predict_proba(row)[:, 1][0])
    CHURN_PROBA.observe(proba)
    will_churn = proba >= 0.5

    try:
        log_prediction(features=features.model_dump(), prediction=will_churn, proba=proba)
    except Exception:
        pass   # never let monitoring take down the prediction path

    return {
        "churn_probability": round(proba, 4),
        "will_churn": will_churn,
    }

@app.get("/")
def health():
    """A simple health check — confirms the server is alive."""
    return {"status": "ok", "service": "player-churn"}

# custom ML metric: the distribution of churn scores we serve
CHURN_PROBA = Histogram(
    "churn_probability",
    "Distribution of predicted churn probabilities",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# expose /metrics — place at the bottom of the file, after routes exist
Instrumentator().instrument(app).expose(app)