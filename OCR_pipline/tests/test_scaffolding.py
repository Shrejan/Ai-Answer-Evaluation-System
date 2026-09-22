"""Phase 0 smoke tests — config load and schema validation."""

from ocr_pipeline.api.schemas import (
    HealthResponse,
    OCRSuccessResponse,
    TimingsMs,
    WordRecord,
)
from ocr_pipeline.config import load_config


def test_load_config_defaults():
    cfg = load_config()
    assert cfg.detection.max_dimension == 2048
    assert cfg.recognition.batch_size == 4
    assert cfg.crop.process_pool_workers == 2
    assert cfg.spell.enabled is True
    assert cfg.queue.max_depth == 8


def test_success_response_schema_roundtrip():
    payload = OCRSuccessResponse(
        request_id="req_123",
        page_id="page_001",
        text="hello world",
        words=[
            WordRecord(id=0, text="hello", confidence=0.9, line_id=0),
            WordRecord(id=1, text="world", confidence=0.85, line_id=0),
        ],
        line_count=1,
        recognized_line_count=1,
        timings_ms=TimingsMs(total=100),
    )
    data = payload.model_dump()
    assert data["status"] == "ok"
    assert len(data["words"]) == 2


def test_health_response_schema():
    health = HealthResponse(
        gpu={"available": False},
        models={"warmed_up": False},
    )
    assert health.status == "ok"
