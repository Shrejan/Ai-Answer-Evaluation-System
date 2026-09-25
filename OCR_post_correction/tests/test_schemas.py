"""
tests/test_schemas.py — Tests for Pydantic v2 schemas: boundary values, invalid confidences,
empty word lists, and full serialization of the target Stage3Result schema.
"""

import pytest
from pydantic import ValidationError
from stage3.schemas import (
    Word,
    OCRInput,
    Span,
    CandidatePool,
    ValidationResult,
    Decision,
    Correction,
    LowConfidenceSpan,
    Segment,
    DocumentStructure,
    QualityFlags,
    Stage3Metadata,
    Stage3Result,
)


def test_word_valid():
    w = Word(id=0, text="General", confidence=0.98)
    assert w.id == 0
    assert w.text == "General"
    assert w.confidence == 0.98


def test_word_invalid_confidence():
    with pytest.raises(ValidationError):
        Word(id=0, text="General", confidence=1.5)

    with pytest.raises(ValidationError):
        Word(id=0, text="General", confidence=-0.1)


def test_ocr_input_valid():
    payload = {
        "request_id": "page_001",
        "student_id": "S2024_112",
        "question_id": "Q3",
        "page_id": 1,
        "text": "General health medical camps",
        "words": [
            {"id": 0, "text": "General", "confidence": 0.98},
            {"id": 1, "text": "health", "confidence": 0.97},
        ],
    }
    inp = OCRInput(**payload)
    assert inp.request_id == "page_001"
    assert len(inp.words) == 2


def test_ocr_input_empty_words():
    payload = {
        "request_id": "page_001",
        "text": "",
        "words": [],
    }
    with pytest.raises(ValidationError):
        OCRInput(**payload)


def test_stage3_result_serialization():
    result = Stage3Result(
        request_id="page_001",
        student_id="S2024_112",
        question_id="Q3",
        final_text="General health medical camps provide basic check-ups for all age groups.",
        segments=[
            Segment(
                segment_id=0,
                text="General health medical camps provide basic check-ups for all age groups.",
                confidence=0.96,
                word_count=10,
            )
        ],
        low_confidence_spans=[
            LowConfidenceSpan(
                text="nubation check",
                span=[140, 155],
                confidence=0.41,
                corrected=False,
                rejection_reason="below_confidence_threshold",
            )
        ],
        structure=DocumentStructure(paragraphs=[0, 1], line_breaks_preserved=True),
        quality_flags=QualityFlags(
            avg_confidence=0.89,
            min_confidence=0.41,
            pct_corrected=0.08,
            has_unresolved_uncertainty=True,
        ),
        stage3_metadata=Stage3Metadata(
            corrections_applied=3,
            corrections_rejected=1,
            model_version="qwen3:1.7b",
            threshold_snapshot="thresholds_v2.yaml",
        ),
    )

    data = result.model_dump()
    assert data["request_id"] == "page_001"
    assert data["student_id"] == "S2024_112"
    assert data["question_id"] == "Q3"
    assert "General health" in data["final_text"]
    assert data["corrected_text"] == data["final_text"]
    assert len(data["low_confidence_spans"]) == 1
    assert data["quality_flags"]["has_unresolved_uncertainty"] is True
