import pandas as pd
from feast import FeatureStore
from player_churn_model.db import load_features

# 1. Entity dataframe: which players, and AT WHAT TIME we want their features.
#    The timestamp is what makes the join point-in-time correct.
entity_df = load_features("""
    SELECT
        player_id,
        MIN(event_ts) + INTERVAL '1 day' AS event_timestamp
    FROM public.fct_events
    GROUP BY player_id
""")

# 2. Ask Feast for the features as of each player's timestamp
store = FeatureStore(repo_path=".")
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "player_day1_features:events_day1",
        "player_day1_features:purchases_day1",
        "player_day1_features:levels_day1",
        "player_day1_features:max_level_day1",
        "player_day1_features:player_segment",
    ],
).to_df()

print("Feast training set shape:", training_df.shape)
print(training_df.head())

# 3. Parity check against the direct SQL path (features.py)
from player_churn_model.features import build_features
X_direct, _ = build_features()
print("\nDirect features.py shape:", X_direct.shape)
print("Row counts match:", len(training_df) == len(X_direct))