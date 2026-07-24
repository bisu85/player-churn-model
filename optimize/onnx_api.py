# /// script
# requires-python = ">=3.11"
# dependencies = ["onnxruntime", "fastapi", "uvicorn", "numpy", "pydantic"]
# ///
import numpy as np
import onnxruntime as rt
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Churn API (ONNX)")
sess = rt.InferenceSession("models/churn_model.onnx", providers=["CPUExecutionProvider"])

class PlayerFeatures(BaseModel):
    events_day1: int
    purchases_day1: int
    levels_day1: int
    max_level_day1: int
    player_segment: str

@app.post("/predict")
def predict(f: PlayerFeatures):
    inputs = {
        "events_day1":    np.array([[f.events_day1]],    np.float32),
        "purchases_day1": np.array([[f.purchases_day1]], np.float32),
        "levels_day1":    np.array([[f.levels_day1]],    np.float32),
        "max_level_day1": np.array([[f.max_level_day1]], np.float32),
        "player_segment": np.array([[f.player_segment]], object),
    }
    out = sess.run(None, inputs)
    proba = float(out[1][0][1])          # class-1 prob from the list-of-dicts output
    return {"churn_probability": round(proba, 4), "will_churn": proba >= 0.5}

@app.get("/health")
def health():
    return {"status": "ok"}