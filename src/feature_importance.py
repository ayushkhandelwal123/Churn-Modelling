import pandas as pd
import matplotlib.pyplot as plt

from src.train import train_all_models


def plot_feature_importance():

    trained_models, X_test, y_test = train_all_models()

    xgb_pipeline = trained_models["XGBoost"]

    preprocessor = xgb_pipeline.named_steps["preprocessor"]

    model = xgb_pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    ).head(20)

    plt.figure(figsize=(12, 8))

    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    plt.gca().invert_yaxis()

    plt.title("Top 20 Important Features")

    plt.xlabel("Importance")

    plt.show()