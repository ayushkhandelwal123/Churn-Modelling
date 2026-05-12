import joblib
import pandas as pd

MODEL_PATH = "./models/churn_model.pkl"

model = joblib.load(MODEL_PATH)

def predict_churn(customer_data):

    df = pd.DataFrame([customer_data])

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(probability)
    }