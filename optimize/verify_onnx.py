# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "onnxruntime",
#     "scikit-learn",
#     "joblib",
#     "numpy",
#     "pandas",
# ]
# ///
import joblib
import numpy as np
import pandas as pd
import onnxruntime as rt
from pathlib import Path

MODEL_PATH = Path("models/churn_model.joblib")
ONNX_PATH = Path("models/churn_model.onnx")

# --- a few sample rows spanning the segments ---
samples = pd.DataFrame([
    {"events_day1": 12, "purchases_day1": 1, "levels_day1": 5,  "max_level_day1": 8,  "player_segment": "free"},
    {"events_day1": 40, "purchases_day1": 5, "levels_day1": 20, "max_level_day1": 35, "player_segment": "spender"},
    {"events_day1": 3,  "purchases_day1": 0, "levels_day1": 2,  "max_level_day1": 4,  "player_segment": "whale"},
])

# --- sklearn prediction (ground truth) ---
pipeline = joblib.load(MODEL_PATH)
sk_proba = pipeline.predict_proba(samples)[:, 1]
print("sklearn probabilities:", sk_proba)

# --- ONNX prediction ---
sess = rt.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
print("\nONNX expected inputs:")
for i in sess.get_inputs():
    print(f"  {i.name}: {i.type} shape={i.shape}")

# Build the input dict: each column its own [N,1] array, numeric as float32, string as object
onnx_inputs = {
    "events_day1":    samples[["events_day1"]].to_numpy(np.float32),
    "purchases_day1": samples[["purchases_day1"]].to_numpy(np.float32),
    "levels_day1":    samples[["levels_day1"]].to_numpy(np.float32),
    "max_level_day1": samples[["max_level_day1"]].to_numpy(np.float32),
    "player_segment": samples[["player_segment"]].to_numpy(object),
}

outputs = sess.run(None, onnx_inputs)
print("\nONNX raw outputs:")
for o, spec in zip(outputs, sess.get_outputs()):
    print(f"  {spec.name}: {o}")

# --- extract ONNX class-1 probs from the list-of-dicts and assert parity ---
onnx_proba = np.array([row[1] for row in outputs[1]])
print("\nsklearn:", sk_proba)
print("onnx:   ", onnx_proba)
np.testing.assert_allclose(sk_proba, onnx_proba, rtol=1e-4)
print("\n✓ ONNX matches sklearn within float32 tolerance")