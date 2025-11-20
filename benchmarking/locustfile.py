from locust import HttpUser, between, task

PATH = "/v1/dish"

BODY = {"text": "Need a french dish"}
HEADERS = {"Content-Type": "application/json"}


class User(HttpUser):
    wait_time = between(0.5, 1.5)

    @task
    def post_embedding(self):
        self.client.post(PATH, json=BODY, headers=HEADERS)
