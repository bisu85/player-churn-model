# src/player_churn_model/monitoring/promote.py
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
MODEL_NAME = "player-churn"
CHAMPION_ALIAS = "champion"
TOLERANCE = 0.005   # challenger must beat champion by at least this to promote


def _auc_of_version(client: MlflowClient, version: str) -> float | None:
    """roc_auc logged on the run behind a version, or None if absent."""
    mv = client.get_model_version(MODEL_NAME, version)
    run = client.get_run(mv.run_id)
    return run.data.metrics.get("roc_auc")   # .get, not [] — no KeyError


def get_champion_auc(client: MlflowClient) -> float | None:
    """AUC of the current champion, or None if no champion / no score."""
    try:
        champ = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    except Exception:
        return None
    return _auc_of_version(client, champ.version)   # may be None for legacy versions


def consider_promotion(challenger_version: str, challenger_auc: float) -> bool:
    """Promote the challenger to champion only if it beats the incumbent.

    Returns True if promoted. First-ever model auto-promotes (no incumbent).
    """
    client = MlflowClient()

    if challenger_auc <= 0.50:                          # ← add these two lines
        print(f"REJECTED v{challenger_version}: AUC {challenger_auc:.4f} below floor 0.55")
        return False

    champion_auc = get_champion_auc(client)

    if champion_auc is None:
        client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, challenger_version)
        print(f"No champion yet — promoting v{challenger_version} (AUC={challenger_auc:.4f})")
        return True

    margin = challenger_auc - champion_auc
    if margin >= TOLERANCE:
        client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, challenger_version)
        print(f"PROMOTED v{challenger_version}: {challenger_auc:.4f} beats "
              f"champion {champion_auc:.4f} by {margin:+.4f}")
        return True

    print(f"KEPT champion: challenger {challenger_auc:.4f} vs "
          f"champion {champion_auc:.4f} (margin {margin:+.4f} < {TOLERANCE})")
    return False