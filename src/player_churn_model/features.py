import pandas as pd
from player_churn_model.db import load_features

FEATURE_QUERY = """
WITH player_first AS (
    SELECT player_id, MIN(event_ts) AS first_ts, MAX(event_ts) AS last_ts
    FROM public.fct_events
    GROUP BY player_id
),
day1 AS (
    SELECT
        f.player_id,
        COUNT(*) AS events_day1,
        COUNT(*) FILTER (WHERE f.event_type = 'purchase')       AS purchases_day1,
        COUNT(*) FILTER (WHERE f.event_type = 'level_complete') AS levels_day1,
        MAX(f.level) AS max_level_day1
    FROM public.fct_events f
    JOIN player_first pf ON f.player_id = pf.player_id
    WHERE f.event_ts < pf.first_ts + INTERVAL '1 day'
    GROUP BY f.player_id
)
SELECT
    d.player_id,
    d.events_day1,
    d.purchases_day1,
    d.levels_day1,
    d.max_level_day1,
    dp.player_segment,
    CASE
        WHEN pf.last_ts < DATE '2026-07-05' - INTERVAL '14 days'
        THEN 1 ELSE 0
    END AS churned
FROM day1 d
JOIN player_first pf      ON d.player_id = pf.player_id
JOIN public.dim_player dp ON d.player_id = dp.player_id
"""

# The exact feature columns the model expects, in order — the train/serve contract.
FEATURE_COLUMNS = [
    "events_day1",
    "purchases_day1",
    "levels_day1",
    "max_level_day1",
    "player_segment",
]


def build_features() -> tuple[pd.DataFrame, pd.Series]:
    """Load raw features from the warehouse and return (X, y) ready for training."""
    df = load_features(FEATURE_QUERY)
    y = df["churned"]
    X = df[FEATURE_COLUMNS]
    return X, y