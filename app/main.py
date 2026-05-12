from fastapi import FastAPI
from app.schema import CustomerData
from app.predictor import predict_churn

app = FastAPI(
    title="Telco Churn Prediction API",
    version="1.0"
)

@app.get("/")
def home():
    return {"message": "API Running"}

@app.post("/predict")
def predict(data: CustomerData):

    result = predict_churn(data.dict())

    return result