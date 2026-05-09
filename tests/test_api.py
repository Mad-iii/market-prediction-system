from fastapi.testclient import TestClient
from src.api.main import app
import os

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200

def test_get_latest_no_data():
    # If file doesn't exist, it should return error or empty list
    response = client.get("/latest")
    assert response.status_code == 200

def test_metrics_exists():
    response = client.get("/metrics")
    assert response.status_code == 200
