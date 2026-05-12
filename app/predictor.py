import joblib
import pandas as pd

MODEL_PATH = "./models/churn_model.pkl"

model = joblib.load(MODEL_PATH)

def get_risk_level(probability):

    if probability >= 0.75:
        return "High"

    elif probability >= 0.40:
        return "Medium"

    return "Low"

def predict_churn(customer_data):

    df = pd.DataFrame([customer_data])

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0][1]

    risk_level = get_risk_level(probability)

    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(probability),
        "risk_level": risk_level
    }