import random
from locust import HttpUser, task, between


class ChurnUser(HttpUser):
    wait_time = between(0.1, 0.5)   # each simulated user pauses briefly between requests
    

    @task
    def predict(self):
        self.client.post("/predict", json={
            "events_day1": random.randint(0, 60),
            "purchases_day1": random.randint(0, 6),
            "levels_day1": random.randint(0, 25),
            "max_level_day1": random.randint(0, 40),
            "player_segment": random.choice(["free", "spender", "whale"]),
        })