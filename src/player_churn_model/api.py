from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from pathlib import Path
from player_churn_model.monitoring.predict_log import log_prediction
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram
from feast import FeatureStore
from functools import lru_cache

app = FastAPI(title="Player Churn API")

# Load the Feast store once at startup, like the model — not per request
FEAST_REPO = Path(__file__).resolve().parents[2] / "feature_repo"
# fs = FeatureStore(repo_path=str(FEAST_REPO))

FEAST_FEATURES = [
    "player_day1_features:events_day1",
    "player_day1_features:purchases_day1",
    "player_day1_features:levels_day1",
    "player_day1_features:max_level_day1",
    "player_day1_features:player_segment",
]


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

@lru_cache(maxsize=1)
def get_feast_store() -> FeatureStore:
    """Build the Feast store on first use, not at import.

    Keeps importing api.py free of a DB/config dependency (so CI can import it),
    and still loads the store only once thanks to the cache.
    """
    return FeatureStore(repo_path=str(FEAST_REPO))


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

@app.post("/predict_by_id")
def predict_by_id(player_id: int):
    """Fetch features from Feast by player_id, then predict — no client-supplied features."""
    fs = get_feast_store()          # ← built on first request, cached after
    feast_row = fs.get_online_features(
        features=FEAST_FEATURES,
        entity_rows=[{"player_id": player_id}],
    ).to_dict()

    # Feast returns lists (one per entity); unwrap to a single-row DataFrame
    row = pd.DataFrame([{
        "events_day1": feast_row["events_day1"][0],
        "purchases_day1": feast_row["purchases_day1"][0],
        "levels_day1": feast_row["levels_day1"][0],
        "max_level_day1": feast_row["max_level_day1"][0],
        "player_segment": feast_row["player_segment"][0],
    }])

    # Guard: unknown player_id → Feast returns None for features
    if row["events_day1"].iloc[0] is None:
        return {"error": f"No features found for player_id {player_id}"}

    proba = float(model.predict_proba(row)[:, 1][0])
    will_churn = proba >= 0.5

    CHURN_PROBA.observe(proba)          # same metric as /predict
    try:
        log_prediction(features=row.iloc[0].to_dict(), prediction=will_churn, proba=proba)
    except Exception:
        pass

    return {
        "player_id": player_id,
        "churn_probability": round(proba, 4),
        "will_churn": will_churn,
        "source": "feast_online_store",
    }

@app.get("/health")
def health():
    """A simple health check — confirms the server is alive."""
    return {"status": "ok", "service": "player-churn"}

@app.get("/ready")
def ready():
    """Readiness — is the model loaded and are we able to serve?"""
    if model is None:
        return JSONResponse(status_code=503, content={"status": "not ready"})
    return {"status": "ready"}

# custom ML metric: the distribution of churn scores we serve
CHURN_PROBA = Histogram(
    "churn_probability",
    "Distribution of predicted churn probabilities",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


# expose /metrics — place at the bottom of the file, after routes exist
Instrumentator().instrument(app).expose(app)