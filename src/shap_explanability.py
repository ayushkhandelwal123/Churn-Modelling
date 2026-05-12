import shap


def generate_shap_summary(
    model_pipeline,
    X_test
):

    preprocessor = model_pipeline.named_steps[
        "preprocessor"
    ]

    model = model_pipeline.named_steps[
        "model"
    ]

    X_processed = preprocessor.transform(X_test)

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        X_processed
    )

    shap.summary_plot(
        shap_values,
        X_processed,
        feature_names=feature_names
    )