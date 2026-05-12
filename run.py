from src.train import train_model
from src.evaluate import evaluate_model
from src.save_model import save_model

MODEL_PATH = "./models/churn_model.pkl"

def main():

    model, X_test, y_test = train_model()

    evaluate_model(model, X_test, y_test)

    save_model(model, MODEL_PATH)

    print("Model saved successfully.")

if __name__ == "__main__":
    main()