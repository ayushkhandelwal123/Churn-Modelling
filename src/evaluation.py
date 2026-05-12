import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def evaluate_models(models, X_test, y_test):

    results = []

    for name, model in models.items():

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {

            "Model": name,

            "Accuracy": accuracy_score(y_test, predictions),

            "Precision": precision_score(y_test, predictions),

            "Recall": recall_score(y_test, predictions),

            "F1 Score": f1_score(y_test, predictions),

            "ROC-AUC": roc_auc_score(y_test, probabilities)
        }

        results.append(metrics)

    return pd.DataFrame(results)