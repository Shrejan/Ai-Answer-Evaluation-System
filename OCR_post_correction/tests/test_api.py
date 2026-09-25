"""
tests/test_api.py — Unit and integration tests for FastAPI ocr_post_pro_pipeline server.
"""

import pytest
from fastapi.testclient import TestClient
from ocr_post_pro_pipeline.main import app

client = TestClient(app)


def test_api_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "ocr_post_pro_pipeline"


def test_api_correct_endpoint():
    payload = {
        "request_id": "api_test_01",
        "text": "General health medical camps provide basic denital check-ups.",
        "words": [
            {"id": 0, "text": "General", "confidence": 0.98},
            {"id": 1, "text": "health", "confidence": 0.99},
            {"id": 2, "text": "medical", "confidence": 0.99},
            {"id": 3, "text": "camps", "confidence": 0.97},
            {"id": 4, "text": "provide", "confidence": 0.99},
            {"id": 5, "text": "basic", "confidence": 0.99},
            {"id": 6, "text": "denital", "confidence": 0.41},
            {"id": 7, "text": "check-ups.", "confidence": 0.95},
        ],
    }

    response = client.post("/correct", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["request_id"] == "api_test_01"
    assert "final_text" in data
    assert "quality_flags" in data
    assert "stage3_metadata" in data
