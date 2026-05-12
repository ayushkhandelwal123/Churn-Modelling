import pandas as pd

from src.train import train_all_models
from src.evaluate import evaluate_model


def compare_models():

    trained_models, X_test, y_test = train_all_models()

    results = []

    best_model = None
    best_score = 0

    for name, model in trained_models.items():

        metrics = evaluate_model(model, X_test, y_test)

        metrics["Model"] = name

        results.append(metrics)

        if metrics["ROC-AUC"] > best_score:

            best_score = metrics["ROC-AUC"]

            best_model = model

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="ROC-AUC",
        ascending=False
    )

    print("\nMODEL COMPARISON")
    print(results_df)

    return best_model, results_df