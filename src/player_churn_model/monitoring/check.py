"""Run a drift check of a current batch against the frozen reference.

Exit 0 = within threshold. Exit 1 = drift exceeded (the signal 4b acts on).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from player_churn_model.monitoring.drift import (
    load_reference, build_report, drifted_share,
)

_ROOT = Path(__file__).resolve().parents[3]
_REPORTS = _ROOT / "reports"
DRIFT_THRESHOLD = 0.3          # our policy — not Evidently's built-in 0.5 default


def run_check(current_path: Path, threshold: float = DRIFT_THRESHOLD) -> float:
    reference = load_reference()
    current = pd.read_parquet(current_path)

    my_eval = build_report(current, reference)
    share = drifted_share(my_eval)

    _REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = _REPORTS / f"drift_{stamp}.html"
    my_eval.save_html(str(out))

    print(f"Drifted share: {share:.3f} (threshold {threshold})")
    print(f"Report saved: {out}")
    return share


def main() -> int:
    parser = argparse.ArgumentParser(description="Drift check vs reference.")
    parser.add_argument("current", type=Path, help="Parquet of the current batch")
    parser.add_argument("--threshold", type=float, default=DRIFT_THRESHOLD)
    args = parser.parse_args()

    share = run_check(args.current, args.threshold)
    if share > args.threshold:
        print("DRIFT DETECTED — exceeds threshold", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())