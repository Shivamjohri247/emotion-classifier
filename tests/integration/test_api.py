from fastapi.testclient import TestClient
from api import app
import pytest

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "model_name" in response.json()
    assert "supported_emotions" in response.json()

def test_prediction():
    response = client.post(
        "/predict",
        json={"text": "I am feeling happy today!"}
    )
    assert response.status_code == 200
    assert "emotion" in response.json()
    assert "confidence" in response.json()

def test_batch_prediction():
    response = client.post(
        "/predict/batch",
        json={"texts": ["I am happy!", "I am sad."]}
    )
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert "total_processed" in response.json() 