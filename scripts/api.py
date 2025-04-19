from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("scripts/apt_group_rf_model.joblib")

class Sample(BaseModel):
    features: list

@app.get("/")
def root():
    return {"message": "APT Group Prediction API is running"}

@app.post("/predict")
def predict(sample: Sample):
    if len(sample.features) != 148:
        return {"error": "Expected 148 features"}
    
    data = [sample.features]
    prediction = model.predict(data)
    return {"APTGroup": int(prediction[0])}
