import joblib
import json
from datetime import datetime


def save_model(
    model,
    model_path,
    metadata_path,
    metadata
):

    joblib.dump(model, model_path)

    metadata["training_date"] = str(
        datetime.now()
    )

    with open(metadata_path, "w") as f:

        json.dump(metadata, f, indent=4)

    print("\nModel + metadata saved.")