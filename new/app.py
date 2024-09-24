from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle 

app = FastAPI()

class CancerPredictionInput(BaseModel):
    radius_mean: float
    texture_mean: float

with open('new\model_pickle_new', 'rb') as file:
    model = pickle.load(file)

@app.post("/predict")
def predict_diagnosis(input_data: CancerPredictionInput):
    
    data = np.array([[input_data.radius_mean, input_data.texture_mean]])

    prediction = model.predict(data)[0]

    diagnosis = "Malignant" if prediction == 1 else "Benign"

    return {"diagnosis": diagnosis}