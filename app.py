import os
import logging

from LazadaIDReviews.config.configuration import ConfigurationManager
from LazadaIDReviews.components.predict import Predict
from fastapi import FastAPI
from pydantic import BaseModel

os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = FastAPI()
config = ConfigurationManager()

class Item(BaseModel):
    reviewContents: list

@app.get("/")
def read_root():
    return {"message": "Up!"}

@app.post("/predict")
def predict(item: Item):
    logging.info("Load configuration.")
    predict_config = config.get_prediction_config()
    predict = Predict(config=predict_config)
    
    logging.info("Make prediction.")
    result = predict.run(item.reviewContents)
    
    return result