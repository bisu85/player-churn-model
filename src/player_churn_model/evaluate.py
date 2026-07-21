# src/player_churn_model/evaluate.py
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from player_churn_model.features import build_features
from player_churn_model.train import build_pipeline

RANDOM_STATE = 42
TEST_SIZE = 0.25


def evaluate() -> float:
    """Honest held-out AUC: split, fit on train, score on test.

    Uses the same split and pipeline as registration so champion and
    challenger scores are directly comparable.
    """
    X, y = build_features()

    # Sanity floor: a wildly imbalanced target usually means the data
    # regenerated or the label definition drifted — fail loud, don't train silently.
    minority_frac = y.value_counts(normalize=True).min()
    if minority_frac < 0.10:
        raise ValueError(
            f"Severe class imbalance (minority={minority_frac:.1%}) — "
            f"check the churn label / data source before trusting this score."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    return roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])