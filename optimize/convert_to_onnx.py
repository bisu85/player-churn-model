# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "skl2onnx",
#     "onnx",
#     "onnxruntime",
#     "scikit-learn",
#     "joblib",
# ]
# ///
import joblib
from pathlib import Path
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

MODEL_PATH = Path("models/churn_model.joblib")
ONNX_PATH = Path("models/churn_model.onnx")

pipeline = joblib.load(MODEL_PATH)

# initial_types: one entry per input column, names matching the ColumnTransformer's
# column names, with the RIGHT tensor type. Numeric -> Float, the string col -> String.
initial_types = [
    ("events_day1",     FloatTensorType([None, 1])),
    ("purchases_day1",  FloatTensorType([None, 1])),
    ("levels_day1",     FloatTensorType([None, 1])),
    ("max_level_day1",  FloatTensorType([None, 1])),
    ("player_segment",  StringTensorType([None, 1])),   # the sharp edge
]

onnx_model = convert_sklearn(
    pipeline,
    initial_types=initial_types,
    target_opset=None,          # let skl2onnx pick the latest tested opset
)

ONNX_PATH.write_bytes(onnx_model.SerializeToString())
print(f"Saved ONNX model to {ONNX_PATH} ({ONNX_PATH.stat().st_size} bytes)")