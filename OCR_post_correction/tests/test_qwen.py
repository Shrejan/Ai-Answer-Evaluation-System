"""
tests/test_qwen.py — Unit and integration tests for Qwen verifier components (M07).
"""

import json
import pytest
from stage3.schemas import Span, CandidatePool, PromptBundle
from stage3.qwen.client import OllamaClient, MockLLMClient
from stage3.qwen.prompt_templates import build_prompt_bundle, build_schema, SYSTEM_PROMPT
from stage3.qwen.verifier import verify_span, VerifierError
from stage3.qwen.model_loader import warmup_model


def test_system_prompt_rules():
    assert "You are an OCR correction verifier." in SYSTEM_PROMPT
    assert "1. You may select ONLY from the candidate list." in SYSTEM_PROMPT
    assert "9. Return only valid JSON. No explanation, no markdown, no extra text." in SYSTEM_PROMPT


def test_build_schema():
    candidates = ["beat", "bear", "boat"]
    schema = build_schema(candidates)

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["selected_candidate"]["enum"] == candidates
    assert schema["required"] == ["selected_candidate", "confidence"]


def test_build_prompt_bundle():
    span = Span(
        span_ids=[0],
        original_text="pregnue",
        clean_text="pregnue",
        prev_context="severe",
        next_context="women",
    )
    pool = CandidatePool(
        candidates=["pregnue", "pregnant"],
        sources={"pregnue": ["original"], "pregnant": ["symspell"]},
    )

    bundle = build_prompt_bundle(span, pool)
    assert isinstance(bundle, PromptBundle)
    assert bundle.system == SYSTEM_PROMPT
    assert "Context (before): severe" in bundle.user
    assert "OCR text: pregnue" in bundle.user
    assert "Context (after): women" in bundle.user
    assert '["pregnue", "pregnant"]' in bundle.user
    assert bundle.schema_dict["properties"]["selected_candidate"]["enum"] == ["pregnue", "pregnant"]


def test_mock_llm_client_default():
    client = MockLLMClient()
    span = Span(span_ids=[0], original_text="denital", clean_text="denital")
    pool = CandidatePool(
        candidates=["denital", "dental"],
        sources={"denital": ["original"], "dental": ["symspell"]},
    )
    bundle = build_prompt_bundle(span, pool)
    res = client.select(bundle)

    data = json.loads(res)
    assert data["selected_candidate"] == "dental"
    assert data["confidence"] == 0.95
    assert client.call_count == 1


def test_mock_llm_client_fixed_response():
    client = MockLLMClient(fixed_response='{"selected_candidate": "denital", "confidence": 0.88}')
    span = Span(span_ids=[0], original_text="denital", clean_text="denital")
    pool = CandidatePool(candidates=["denital"], sources={"denital": ["original"]})
    bundle = build_prompt_bundle(span, pool)

    res = client.select(bundle)
    assert res == '{"selected_candidate": "denital", "confidence": 0.88}'


def test_verify_span_returns_raw_string():
    client = MockLLMClient(fixed_response='{"selected_candidate": "dental", "confidence": 0.99}')
    span = Span(span_ids=[0], original_text="denital", clean_text="denital")
    pool = CandidatePool(candidates=["denital", "dental"], sources={"denital": ["original"], "dental": ["symspell"]})

    raw = verify_span(client, span, pool)
    assert isinstance(raw, str)
    assert raw == '{"selected_candidate": "dental", "confidence": 0.99}'


def test_verify_span_timeout_error():
    client = MockLLMClient(raise_timeout=True)
    span = Span(span_ids=[0], original_text="test", clean_text="test")
    pool = CandidatePool(candidates=["test"], sources={"test": ["original"]})

    with pytest.raises(VerifierError) as exc_info:
        verify_span(client, span, pool)
    assert exc_info.value.error_type == "llm_timeout"


def test_verify_span_exception_error():
    client = MockLLMClient(raise_exception=RuntimeError("Connection failed"))
    span = Span(span_ids=[0], original_text="test", clean_text="test")
    pool = CandidatePool(candidates=["test"], sources={"test": ["original"]})

    with pytest.raises(VerifierError) as exc_info:
        verify_span(client, span, pool)
    assert exc_info.value.error_type == "llm_exception"


def test_warmup_model():
    client = MockLLMClient(fixed_response='{"selected_candidate": "warmup", "confidence": 1.0}')
    assert warmup_model(client) is True

    failing_client = MockLLMClient(raise_exception=RuntimeError("Ollama offline"))
    assert warmup_model(failing_client) is False


@pytest.mark.ollama
def test_ollama_client_live():
    client = OllamaClient()
    span = Span(span_ids=[0], original_text="denital", clean_text="denital")
    pool = CandidatePool(candidates=["denital", "dental"], sources={"denital": ["original"], "dental": ["symspell"]})
    try:
        raw = verify_span(client, span, pool, timeout_s=10.0)
        assert isinstance(raw, str)
        assert len(raw) > 0
    except VerifierError as e:
        pytest.skip(f"Ollama server not reachable: {e}")
