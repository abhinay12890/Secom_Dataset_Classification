from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import json
from typing import Dict
import joblib

with open("config.json","r") as f:
    config=json.load(f)


threshold=config["threshold"]


model = joblib.load("final_model.pkl")

app=FastAPI()

class InputData(BaseModel):
    inp:Dict[str,float]

@app.get("/")
def welcome():
    return {"message":"API is running"}

@app.post("/predict")
def predict(data:InputData):
    required_cols=model.feature_names_in_
    missing=[col for col in required_cols if col not in data.inp]
    if missing:
        return {"missing_colunms":missing}
    model_input=pd.DataFrame([data.inp])
    pred=model.predict_proba(model_input)[0][1]
    prediction="Fail" if pred>=threshold else "Pass"
    pass_prob=1-pred
    fail_prob=pred
    return {
    "prediction": prediction,
    "prob_pass": float(pass_prob),
    "prob_fail": float(fail_prob),
    "threshold":threshold}