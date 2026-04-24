# tests/test_api.py
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_kwh_forecast():
    response = client.post("/kwh/forecast", json={"days": 3})
    
    assert response.status_code in [200, 503]  # 503 if model not loaded
    if response.status_code == 200:
        data = response.json()
        assert "forecast" in data


def test_error_forecast():
    response = client.post("/error/forecast", json={"days": 3})
    
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "forecast" in data