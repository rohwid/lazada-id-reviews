import json

from random import randint
from locust import TaskSet, constant, task, HttpUser

f = open('data.json') 
data = json.load(f)

class HitServer(TaskSet):
    @task
    def get_url(self):
        self.client.get("/")

class HitEndpoint(TaskSet):
    @task
    def post_predict(self):
        limit = randint(1, len(data['reviewContents']))
        
        # Update data with selected data
        data['reviewContents'] = data['reviewContents'][0:limit]
        
        # Hit API endpoint
        self.client.post(
            "/predict",
            json=data
        )
    
class UserLoadTest(HttpUser):
    host="https://mlops-class-prd.valiance.ai"
    tasks=[HitEndpoint]
    wait_time=constant(1)