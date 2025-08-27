import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from agent.ml_agent import MLAgent
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from agent.ml_agent import MLAgent

# load the model from artifacts folder
model_path = "artifacts/best_model_pipeline.pkl"

# load the model using pickle
with open(file=model_path, mode='rb') as file:
    model = pickle.load(file)

app = FastAPI()

class HouseFeatures(BaseModel):
    area_type: str
    availability: str
    location: str
    total_sqft: float
    bath: int
    balcony: int
    bhk: int


@app.post("/predict")
def predict_prices(features: HouseFeatures):
    input_data = pd.DataFrame([features.model_dump()])
    prediction = model.predict(input_data)
    return {"Predicted Price in Lakhs": prediction[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent_tools.main:app", host="127.0.0.1", port=8000)