from fastapi.testclient import TestClient
from model_serve import app

client = TestClient(app)

def test_prediction():
    response = client.post("/predict", json={"sqft": 1000, "bedrooms": 3, "age": 5})
    print(response.json())
    assert response.status_code == 200
    assert "price" in response.json()
    	
