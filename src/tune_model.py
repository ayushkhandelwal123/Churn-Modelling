from sklearn.model_selection import RandomizedSearchCV


def tune_models(models, X_train, y_train):

    tuned_models = {}

    param_grids = {

        "Logistic Regression": {

            "model__C": [0.01, 0.1, 1, 10],
        },

        "Random Forest": {

            "model__n_estimators": [100, 200, 300],

            "model__max_depth": [5, 10, 20],

            "model__min_samples_split": [2, 5, 10]
        },

        "XGBoost": {

            "model__n_estimators": [100, 200],

            "model__max_depth": [3, 5, 7],

            "model__learning_rate": [0.01, 0.05, 0.1],

            "model__subsample": [0.7, 0.8, 1.0]
        }
    }

    for name, model in models.items():

        print(f"\nTuning {name}...")

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grids[name],
            n_iter=5,
            scoring="roc_auc",
            cv=5,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )

        search.fit(X_train, y_train)

        print(f"\nBest Parameters for {name}:")
        print(search.best_params_)

        print(f"Best ROC-AUC: {search.best_score_:.4f}")

        tuned_models[name] = search.best_estimator_

    return tuned_models