"""
VisionOps Guard - Automated FastAPI REST Service Integration Tests
Verifies HTTP endpoints, status codes, health checks, and multipart image prediction payloads.
"""

from pathlib import Path
import io
import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient
from src.serving.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    json_resp = response.json()
    assert json_resp["service"] == "VisionOps Guard Microservice"
    assert json_resp["status"] == "healthy"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    json_resp = response.json()
    assert json_resp["status"] == "up"


def test_predict_invalid_content_type():
    # Send text file instead of image
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"hello world", "text/plain")}
    )
    assert response.status_code == 400
    assert "Uploaded file must be a valid image" in response.json()["detail"]


def test_predict_valid_image():
    # Generate dummy image bytes
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", img)
    img_bytes = buffer.tobytes()

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    )
    assert response.status_code == 200
    json_resp = response.json()
    assert json_resp["status"] == "success"
    assert "detections" in json_resp
    assert "annotated_image_base64" in json_resp
