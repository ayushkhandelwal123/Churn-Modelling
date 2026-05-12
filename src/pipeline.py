from src.data_loader import load_data

from src.preprocessing import (
    clean_data,
    create_preprocessor
)

from src.modelling import build_models

from src.tune_model import tune_models

from src.evaluation import evaluate_models

from src.cross_validate import (
    perform_cross_validation
)

from src.threshold import (
    optimize_threshold
)

from src.visualization import (
    plot_roc_curves
)

from src.shap_explanability import (
    generate_shap_summary
)

from src.save_model import save_model

from sklearn.model_selection import (
    train_test_split
)


DATA_PATH = "./data/Telco_Customer_Churn.xlsx"

MODEL_PATH = "./models/churn_model.pkl"

METADATA_PATH = "./models/model_metadata.json"


def run_pipeline():

    print("\nSTEP 1: Loading data")

    df = load_data(DATA_PATH)

    print("\nSTEP 2: Cleaning data")

    df = clean_data(df)

    X = df.drop("Churn_Value", axis=1)

    y = df["Churn_Value"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
    )

    print("\nSTEP 3: Creating preprocessor")

    preprocessor = create_preprocessor(
        X_train
    )

    print("\nSTEP 4: Building models")

    models = build_models(preprocessor)

    print("\nSTEP 5: Hyperparameter tuning")

    tuned_models = tune_models(
        models,
        X_train,
        y_train
    )

    print("\nSTEP 6: Evaluating models")

    results_df = evaluate_models(
        tuned_models,
        X_test,
        y_test
    )

    print(results_df)

    print("\nSTEP 7: ROC curve generation")

    plot_roc_curves(
        tuned_models,
        X_test,
        y_test
    )

    best_model_name = (
        results_df.sort_values(
            by="ROC-AUC",
            ascending=False
        )
        .iloc[0]["Model"]
    )

    best_model = tuned_models[
        best_model_name
    ]

    print(f"\nBest Model: {best_model_name}")

    print("\nSTEP 9: Threshold Optimization")

    best_threshold = optimize_threshold(
        best_model,
        X_test,
        y_test
    )

    print(
        f"\nOptimal Threshold: {best_threshold}"
    )

    metadata = {

    "best_model": best_model_name,

    "optimal_threshold": float(best_threshold),

    "roc_auc": float(
        results_df.sort_values(
            by="ROC-AUC",
            ascending=False
        ).iloc[0]["ROC-AUC"]
    )
}

    print("\nSTEP 8: Cross Validation")

    perform_cross_validation(
        best_model,
        X_train,
        y_train
    )


    print("\nSTEP 10: SHAP Explainability")

    generate_shap_summary(
        best_model,
        X_test
    )

    print("\nSTEP 11: Saving model")

    save_model(
        best_model,
        MODEL_PATH,
        METADATA_PATH,
        metadata
    )

    print("\nPipeline execution completed.")