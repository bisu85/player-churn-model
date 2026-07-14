from fastapi.testclient import TestClient
from player_churn_model.api import app

client = TestClient(app)


def test_health_endpoint():
    """The root health check returns ok."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_valid_probability():
    """A well-formed request returns a probability in [0, 1] and a boolean."""
    payload = {
        "events_day1": 30,
        "purchases_day1": 2,
        "levels_day1": 15,
        "max_level_day1": 20,
        "player_segment": "spender",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["will_churn"], bool)


def test_predict_rejects_missing_field():
    """Omitting a required field is rejected by the Pydantic contract (422)."""
    payload = {
        "events_day1": 30,
        "purchases_day1": 2,
        "levels_day1": 15,
        "max_level_day1": 20,
        # player_segment deliberately missing
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_wrong_type():
    """A non-integer where an int is required is rejected (422)."""
    payload = {
        "events_day1": "lots",   # should be int
        "purchases_day1": 2,
        "levels_day1": 15,
        "max_level_day1": 20,
        "player_segment": "spender",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422