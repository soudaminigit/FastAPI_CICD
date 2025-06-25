from fastapi import FastAPI
from pydantic import BaseModel
from model import load_model, predict

app = FastAPI()
model = load_model("model/house_model.pkl")

class InputFeatures(BaseModel):
    sqft: float
    bedrooms: int
    age: int

@app.post("/predict")
def predict_price(features: InputFeatures):
    return {"price": predict(model, features.dict())}
