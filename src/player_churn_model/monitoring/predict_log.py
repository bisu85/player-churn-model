"""Log served predictions for later drift monitoring."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
_LOG_DIR = _ROOT / "data" / "predictions"
_LOG_PATH = _LOG_DIR / "requests.jsonl"
_lock = Lock()

# must match the model's raw feature schema (from build_features)
FEATURE_COLUMNS = [
    "events_day1", "purchases_day1", "levels_day1", "max_level_day1", "player_segment",
]


def log_prediction(features: dict, prediction, proba: float | None = None) -> None:
    """Append one prediction record. Cheap enough for the request path."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        **{col: features.get(col) for col in FEATURE_COLUMNS},
        "prediction": int(prediction),
        "proba": float(proba) if proba is not None else None,
    }
    line = json.dumps(record, default=str)
    with _lock:
        with _LOG_PATH.open("a") as f:
            f.write(line + "\n")


def collect_current(out_path: Path) -> Path:
    """Batch logged requests into a parquet the drift check can consume."""
    df = pd.read_json(_LOG_PATH, lines=True)
    df[FEATURE_COLUMNS].to_parquet(out_path)
    return out_path