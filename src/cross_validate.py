from sklearn.model_selection import cross_val_score


def perform_cross_validation(
    model,
    X_train,
    y_train
):

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="roc_auc"
    )

    print("\nCross Validation ROC-AUC Scores:")
    print(scores)

    print(f"\nMean ROC-AUC: {scores.mean():.4f}")

    return scores