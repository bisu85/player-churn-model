from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from pathlib import Path

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
    # Turn the validated request into the one-row DataFrame the pipeline expects.
    row = pd.DataFrame([features.model_dump()])
    proba = float(model.predict_proba(row)[:, 1][0])
    return {
        "churn_probability": round(proba, 4),
        "will_churn": proba >= 0.5,
    }

@app.get("/")
def health():
    """A simple health check — confirms the server is alive."""
    return {"status": "ok", "service": "player-churn"}