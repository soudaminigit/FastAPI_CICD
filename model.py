import joblib
import numpy as np

# ✅ This is the custom load_model function
def load_model(path: str):
    return joblib.load(path)

# Optional: Prediction helper
def predict(model, features: dict):
    input_array = np.array([[features["sqft"], features["bedrooms"], features["age"]]])
    return float(model.predict(input_array)[0])
