import numpy as np

import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)


def optimize_threshold(
    model,
    X_test,
    y_test
):

    probabilities = model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.1, 0.95, 0.05)

    results = []

    best_threshold = 0.5

    best_f1 = 0

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
                        y_test,
                        predictions,
                        zero_division=0
                    )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        results.append({
            "Threshold": threshold,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        if f1 > best_f1:

            best_f1 = f1

            best_threshold = threshold

    results_df = pd.DataFrame(results)

    print("\nThreshold Optimization Results:")
    print(results_df)

    print(f"\nBest Threshold: {best_threshold}")

    print(f"Best F1 Score: {best_f1:.4f}")

    return best_threshold