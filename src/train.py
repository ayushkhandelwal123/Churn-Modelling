from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from src.data_loader import load_data
from src.preprocessing import clean_data, create_preprocessor

DATA_PATH = "./data/Telco_Customer_Churn.xlsx"


def get_data():

    df = load_data(DATA_PATH)

    df = clean_data(df)

    X = df.drop("Churn_Value", axis=1)
    y = df["Churn_Value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def build_models(preprocessor):

    models = {

        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=1000
            ))
        ]),

        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42
            ))
        ]),

        "XGBoost": Pipeline([
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42
            ))
        ])
    }

    return models


def train_all_models():

    X_train, X_test, y_train, y_test = get_data()

    preprocessor = create_preprocessor(X_train)

    models = build_models(preprocessor)

    trained_models = {}

    for name, pipeline in models.items():

        print(f"\nTraining {name}...")

        pipeline.fit(X_train, y_train)

        trained_models[name] = pipeline

    return trained_models, X_test, y_test