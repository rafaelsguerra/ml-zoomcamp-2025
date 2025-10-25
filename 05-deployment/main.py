from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn

class Customer(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

class PredictResponse(BaseModel):
    subscription_probability: float
    subscription: bool

app = FastAPI()

def predict_single(client):
    result = pipeline.predict_proba(client)[:, 1]
    return float(result)


PIPELINE_VERSION = "v1"
# PIPELINE_VERSION = "v2"

with open(f'pipeline_{PIPELINE_VERSION}.bin', 'rb') as f:
    pipeline = pickle.load(f)

@app.get("/ping")
def ping():
    return "pong"

@app.post("/predict")
def predict(client: Customer) -> PredictResponse:

    result = predict_single(client.model_dump())

    return PredictResponse(
        subscription_probability = result,
        subscription = result >= .5
    )
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)