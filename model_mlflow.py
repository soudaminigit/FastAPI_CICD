import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib


mlflow.set_experiment("RealEstate")
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

input_example = pd.DataFrame({
    "sqft": [1050],
    "bedrooms": [2.0],
    "age": ["2"]
})

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
	
    mlflow.log_metric("accuracy", 0.87)
    mlflow.log_metric("AUC", 0.89)
    
    
    mlflow.sklearn.log_model(
    sk_model=model,
    name="house_pricing_model",
    input_example=input_example )
    print("Artifact URI:", mlflow.get_artifact_uri())