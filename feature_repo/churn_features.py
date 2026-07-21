from datetime import timedelta
from feast import Entity, FeatureView, Field, PushSource
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.types import Int64, String

# 1. The entity — what features attach to
player = Entity(name="player", join_keys=["player_id"])

# 2. The source — day-1 features WITH the point-in-time timestamp.
#    event_timestamp = first_ts + 1 day: the moment day-1 features become known.
player_day1_source = PostgreSQLSource(
    name="player_day1_source",
    query="""
        WITH player_first AS (
            SELECT player_id, MIN(event_ts) AS first_ts
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
            pf.first_ts + INTERVAL '1 day' AS event_timestamp
        FROM day1 d
        JOIN player_first pf      ON d.player_id = pf.player_id
        JOIN public.dim_player dp ON d.player_id = dp.player_id
    """,
    timestamp_field="event_timestamp",
)

# 3. The feature view — the named group both training and serving read from
player_day1_fv = FeatureView(
    name="player_day1_features",
    entities=[player],
    ttl=timedelta(days=365),
    schema=[
        Field(name="events_day1", dtype=Int64),
        Field(name="purchases_day1", dtype=Int64),
        Field(name="levels_day1", dtype=Int64),
        Field(name="max_level_day1", dtype=Int64),
        Field(name="player_segment", dtype=String),
    ],
    source=player_day1_source,
    online=True,
)