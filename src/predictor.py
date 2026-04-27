import os
import joblib
from src.features import make_features

MODEL_PATH = os.path.join("models", "final_bbb_model.pkl")
FEATURE_COLUMNS_PATH = os.path.join("models", "feature_columns.pkl")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model file not found. Add models/final_bbb_model.pkl locally. "
            "Do not commit private model files if restricted."
        )

    model = joblib.load(MODEL_PATH)

    if os.path.exists(FEATURE_COLUMNS_PATH):
        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    else:
        feature_columns = None

    return model, feature_columns

def predict_bbb(sequence: str):
    model, feature_columns = load_model()
    X = make_features(sequence)

    if feature_columns is not None:
        X = X[feature_columns]

    probability = float(model.predict_proba(X)[0, 1])

    if probability >= 0.70:
        label = "Likely BBB+"
    elif probability <= 0.30:
        label = "Likely BBB-"
    else:
        label = "Uncertain"

    return probability, label
