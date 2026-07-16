# src/player_churn_model/monitoring/drift.py
from pathlib import Path
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# repo root, resolved from this file: monitoring → player_churn_model → src → root
_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_REF = _ROOT / "data/reference/churn_reference_v1.parquet"

def load_reference(path: Path = _DEFAULT_REF) -> pd.DataFrame:
    return pd.read_parquet(path)

def build_report(current: pd.DataFrame, reference: pd.DataFrame):
    report = Report([DataDriftPreset(method="psi")], include_tests=True)
    return report.run(current, reference)          # current first

def drifted_share(my_eval) -> float:
    """Return the fraction of columns flagged as drifted (0.0–1.0)."""
    result = my_eval.dict()
    metrics = result.get("metrics", [])

    for metric in metrics:
        name = str(metric.get("metric_name", ""))
        if name.startswith("DriftedColumnsCount"):
            value = metric["value"]           # {"count": 1.0, "share": 0.2}
            return float(value["share"])

    raise KeyError(
        f"DriftedColumnsCount not found. "
        f"metric_names: {[m.get('metric_name') for m in metrics]}"
    )