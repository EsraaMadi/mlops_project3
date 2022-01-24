# Put the code for your API here.

import os
import pickle
import pandas as pd
import numpy as np
import json 

from typing import Optional

from fastapi import FastAPI, Response
from pydantic import BaseModel, Field


app = FastAPI()

class DataRow(BaseModel):
    age           : int
    fnlgt         : int
    education_num : int = Field(alias='education-num')
    capital_gain  : int = Field(alias='capital-gain')
    capital_loss  : int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    workclass     : str
    education     : str
    marital_status: str = Field(alias='marital-status')
    occupation    : str
    relationship  : str
    race          : str
    sex           : str
    native_country: str = Field(alias='native-country')
        

# load model and encoder
root_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(root_path, 'model/model.pkl')
encoder_path = os.path.join(root_path, 'model/encoder.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(encoder_path, 'rb') as f:
    encoder = pickle.load(f)
    
    
cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]





if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")






@app.get("/")
def welcome():
    return {"Home": "Welcome I am prediction app"}

@app.post("/predict")
def predict_salary(row: DataRow):
    
    X = pd.DataFrame([row.dict()])

    X_categorical = X[cat_features].values
    X_categorical = encoder.transform(X_categorical)
    
    X_continuous = X.drop(cat_features, axis=1)
    X = np.concatenate([X_continuous, X_categorical], axis=1)

    pred = int(model.predict(X)[0])
    return pred