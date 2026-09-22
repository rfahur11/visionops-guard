"""
VisionOps Guard - Automated Model & Inference Unit Tests
Verifies ONNX model loading, bounding box formatting, and prediction data structure.
"""

from pathlib import Path
import numpy as np
import cv2
import pytest
from src.serving.predictor import SafetyPPEPredictor


@pytest.fixture
def sample_image_bytes():
    """Generates a dummy 640x640 JPEG image in memory."""
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()


def test_predictor_initialization():
    """Verify Predictor initializes configuration and session properly."""
    predictor = SafetyPPEPredictor()
    assert predictor.img_size == 640
    assert "person" in predictor.classes.values()


def test_predictor_prediction_structure(sample_image_bytes):
    """Verify prediction return dictionary schema and base64 preview."""
    predictor = SafetyPPEPredictor()
    result = predictor.predict(sample_image_bytes)

    assert result["status"] == "success"
    assert "is_compliant" in result
    assert "violations_count" in result
    assert "latency_ms" in result
    assert isinstance(result["latency_ms"], float)
    assert len(result["annotated_image_base64"]) > 0
