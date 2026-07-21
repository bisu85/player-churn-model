from feast import FeatureStore

store = FeatureStore(repo_path=".")

features = store.get_online_features(
    features=[
        "player_day1_features:events_day1",
        "player_day1_features:purchases_day1",
        "player_day1_features:levels_day1",
        "player_day1_features:max_level_day1",
        "player_day1_features:player_segment",
    ],
    entity_rows=[{"player_id": 184}],   # a player you saw in the training set
).to_dict()

for k, v in features.items():
    print(f"{k}: {v}")