from src.compare_models import compare_models
from src.save_model import save_model

MODEL_PATH = "./models/churn_model.pkl"


def main():

    best_model, results_df = compare_models()

    print("\nSaving Best Model...")

    save_model(best_model, MODEL_PATH)

    print("Best model saved successfully.")


if __name__ == "__main__":
    main()