import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os

# Load updated training data
data = pd.read_csv("current_data.tsv",delimiter='\t')
X = data[["sqft", "bedrooms", "age"]]
y = data["price"]

# Split and retrain model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Save updated model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/house_model.pkl")