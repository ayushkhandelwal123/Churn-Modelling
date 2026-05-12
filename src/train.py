from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocessing import clean_data, create_preprocessor

DATA_PATH = "./data/Telco_Customer_Churn.xlsx"

def train_model():

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

    preprocessor = create_preprocessor(X)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(class_weight="balanced"))
    ])

    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test