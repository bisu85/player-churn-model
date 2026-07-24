# /// script
# requires-python = ">=3.11"
# dependencies = ["onnxruntime", "scikit-learn", "joblib", "numpy", "pandas"]
# ///
import time
import joblib
import numpy as np
import pandas as pd
import onnxruntime as rt

pipeline = joblib.load("models/churn_model.joblib")
sess = rt.InferenceSession("models/churn_model.onnx", providers=["CPUExecutionProvider"])

row = pd.DataFrame([{"events_day1": 12, "purchases_day1": 1, "levels_day1": 5,
                     "max_level_day1": 8, "player_segment": "free"}])
onnx_in = {
    "events_day1":    row[["events_day1"]].to_numpy(np.float32),
    "purchases_day1": row[["purchases_day1"]].to_numpy(np.float32),
    "levels_day1":    row[["levels_day1"]].to_numpy(np.float32),
    "max_level_day1": row[["max_level_day1"]].to_numpy(np.float32),
    "player_segment": row[["player_segment"]].to_numpy(object),
}

N = 5000
# warm up both (first call pays one-time init costs)
pipeline.predict_proba(row) 
sess.run(None, onnx_in)

t = time.perf_counter()
for _ in range(N): 
    pipeline.predict_proba(row)
sk_ms = (time.perf_counter() - t) / N * 1000

t = time.perf_counter()
for _ in range(N): 
    sess.run(None, onnx_in)
onnx_ms = (time.perf_counter() - t) / N * 1000

print(f"sklearn: {sk_ms:.3f} ms/prediction")
print(f"onnx:    {onnx_ms:.3f} ms/prediction")
print(f"ratio:   {sk_ms/onnx_ms:.2f}x  ({'ONNX faster' if onnx_ms < sk_ms else 'sklearn faster'})")