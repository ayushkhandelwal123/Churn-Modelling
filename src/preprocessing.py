import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DROP_COLS = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Label",
    "Churn Score",
    "CLTV",
    "Churn Reason"
]

def clean_data(df):

    df["Total Charges"] = pd.to_numeric(
        df["Total Charges"],
        errors='coerce'
    )

    df = df.dropna(subset=["Total Charges"])

    df = df.drop(columns=DROP_COLS)

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
    )

    return df


def create_preprocessor(X):

    categorical_cols = X.select_dtypes(include="object").columns
    numerical_cols = X.select_dtypes(exclude="object").columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    return preprocessor