import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# 1. Create synthetic dataset
data = {
    "sqft": [1000, 1500, 2000, 1200, 1700],
    "bedrooms": [2, 3, 4, 2, 3],
    "age": [10, 5, 2, 12, 7],
    "price": [200000, 300000, 400000, 220000, 330000]
}
df = pd.DataFrame(data)

# 2. Features and target
X = df[["sqft", "bedrooms", "age"]]
y = df["price"]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Save model to a .pkl file
joblib.dump(model, "model/house_model.pkl")

print(" Model saved as model/house_model.pkl")
