# src/player_churn_model/monitoring/reference.py
from pathlib import Path
import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
_REF_DIR = _ROOT / "data/reference"

def save_reference(X: pd.DataFrame, version: str = "v1") -> Path:
    """Freeze the exact feature set the model trained on, for drift baselining."""
    _REF_DIR.mkdir(parents=True, exist_ok=True)
    path = _REF_DIR / f"churn_reference_{version}.parquet"
    X.to_parquet(path)
    return path