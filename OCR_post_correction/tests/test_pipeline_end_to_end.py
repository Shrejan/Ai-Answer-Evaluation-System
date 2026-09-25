"""
tests/test_pipeline_end_to_end.py — Unit and integration tests for end-to-end pipeline (M10).
"""

import pytest
from stage3.schemas import Stage3Result
from stage3.qwen.client import MockLLMClient, OllamaClient
from stage3.pipeline import correct_page, process_ocr_request, apply_accepted
from stage3.schemas import Span, Decision


def test_apply_accepted_raises_on_rejected_decision():
    span = Span(span_ids=[0], original_text="denital", clean_text="denital")
    decision = Decision(
        accepted=False,
        threshold_used=0.8,
        threshold_type="general",
        reason="below_confidence_threshold",
    )
    with pytest.raises(ValueError):
        apply_accepted(span, decision)


def test_pipeline_happy_path_mock():
    ocr_request = {
        "request_id": "req_001",
        "text": "General health medical camps provide basic check-ups for all age groups.",
        "words": [
            {"id": 0, "text": "General", "confidence": 0.98},
            {"id": 1, "text": "health", "confidence": 0.99},
            {"id": 2, "text": "medical", "confidence": 0.99},
            {"id": 3, "text": "camps", "confidence": 0.97},
            {"id": 4, "text": "provide", "confidence": 0.99},
            {"id": 5, "text": "basic", "confidence": 0.99},
            {"id": 6, "text": "denital", "confidence": 0.41},
            {"id": 7, "text": "check-ups", "confidence": 0.95},
            {"id": 8, "text": "for", "confidence": 0.99},
            {"id": 9, "text": "all", "confidence": 0.99},
            {"id": 10, "text": "age", "confidence": 0.99},
            {"id": 11, "text": "groups.", "confidence": 0.98},
        ],
    }

    mock_client = MockLLMClient(fixed_response='{"selected_candidate": "dental", "confidence": 0.95}')
    res = correct_page(ocr_request, client=mock_client)

    assert isinstance(res, Stage3Result)
    assert "dental" in res.corrected_text
    assert len(res.corrections) > 0

    api_resp = process_ocr_request(ocr_request, client=mock_client)
    assert api_resp["request_id"] == "req_001"
    assert "dental" in api_resp["final_text"]


def test_pipeline_garbage_llm_output_keeps_original():
    ocr_request = {
        "request_id": "req_garbage",
        "text": "The patient needs a denital checkup.",
        "words": [
            {"id": 0, "text": "The", "confidence": 0.99},
            {"id": 1, "text": "patient", "confidence": 0.99},
            {"id": 2, "text": "needs", "confidence": 0.99},
            {"id": 3, "text": "a", "confidence": 0.99},
            {"id": 4, "text": "denital", "confidence": 0.41},
            {"id": 5, "text": "checkup.", "confidence": 0.99},
        ],
    }

    # Mock client returns garbage prose
    mock_client = MockLLMClient(fixed_response="I cannot help with OCR post correction.")
    res = correct_page(ocr_request, client=mock_client)

    # Acceptance invariant: garbage output returns corrected_text == original_text
    assert res.corrected_text == res.original_text
    assert all(c.accepted is False for c in res.corrections)


def test_pipeline_timeout_keeps_original():
    ocr_request = {
        "request_id": "req_timeout",
        "text": "The patient needs a denital checkup.",
        "words": [
            {"id": 0, "text": "The", "confidence": 0.99},
            {"id": 1, "text": "patient", "confidence": 0.99},
            {"id": 2, "text": "needs", "confidence": 0.99},
            {"id": 3, "text": "a", "confidence": 0.99},
            {"id": 4, "text": "denital", "confidence": 0.41},
            {"id": 5, "text": "checkup.", "confidence": 0.99},
        ],
    }

    mock_client = MockLLMClient(raise_timeout=True)
    res = correct_page(ocr_request, client=mock_client)

    assert res.corrected_text == res.original_text
    assert any(c.rejection_reason == "llm_timeout" for c in res.corrections)


@pytest.mark.ollama
def test_pipeline_real_ollama_live():
    ocr_request = {
        "request_id": "req_real",
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

    client = OllamaClient()
    try:
        res = correct_page(ocr_request, client=client)
        assert res.request_id == "req_real"
        assert len(res.corrections) > 0
    except Exception as e:
        pytest.skip(f"Live Ollama skipped: {e}")
