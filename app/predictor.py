import joblib
import json
import pandas as pd


MODEL_PATH = "./models/churn_model.pkl"

METADATA_PATH = "./models/model_metadata.json"


model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r") as f:

    metadata = json.load(f)

THRESHOLD = metadata["optimal_threshold"]


def get_risk_level(probability):

    if probability >= 0.75:
        return "High"

    elif probability >= 0.45:
        return "Medium"

    return "Low"


def predict_churn(customer_data):

    df = pd.DataFrame([customer_data])

    probability = model.predict_proba(df)[0][1]

    prediction = int(probability >= THRESHOLD)

    return {

        "prediction": prediction,

        "probability": round(float(probability), 4),

        "threshold_used": THRESHOLD,

        "risk_level": get_risk_level(probability)
    }