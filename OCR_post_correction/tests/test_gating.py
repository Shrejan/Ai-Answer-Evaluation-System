"""
tests/test_gating.py — Unit tests for gating module (M09).
Achieves 100% line and branch coverage.
"""

import pytest
from stage3.schemas import ValidationResult, Span, CandidatePool
from stage3.gating import decide, make_accepted_decision


def test_decide_validation_failed():
    val = ValidationResult(accepted=False, reason="validation_failed:check_json_parse")
    span = Span(span_ids=[0], original_text="denital", clean_text="denital")
    pool = CandidatePool(candidates=["denital"], sources={"denital": ["original"]})

    res = decide(val, span, pool)
    assert res.accepted is False
    assert res.reason == "validation_failed:check_json_parse"


def test_decide_validation_failed_none_reason():
    val = ValidationResult(accepted=False, reason=None)
    span = Span(span_ids=[0], original_text="denital", clean_text="denital")
    pool = CandidatePool(candidates=["denital"], sources={"denital": ["original"]})

    res = decide(val, span, pool)
    assert res.accepted is False
    assert res.reason == "validation_failed:unknown"


def test_decide_selected_original():
    val = ValidationResult(accepted=True, selected="denital", confidence=0.99)
    span = Span(span_ids=[0], original_text="denital", clean_text="denital")
    pool = CandidatePool(candidates=["denital", "dental"], sources={"denital": ["original"], "dental": ["symspell"]})

    res = decide(val, span, pool)
    assert res.accepted is False
    assert res.reason == "selected_original"


def test_decide_below_confidence_threshold_general():
    val = ValidationResult(accepted=True, selected="dental", confidence=0.50)
    span = Span(span_ids=[0], original_text="denital", clean_text="denital")
    pool = CandidatePool(candidates=["denital", "dental"], sources={"denital": ["original"], "dental": ["symspell"]})

    res = decide(val, span, pool, thresholds={"tau_general": 0.80, "tau_domain": 0.70})
    assert res.accepted is False
    assert res.threshold_type == "general"
    assert res.threshold_used == 0.80
    assert res.reason == "below_confidence_threshold"


def test_decide_below_confidence_threshold_domain():
    val = ValidationResult(accepted=True, selected="dental", confidence=0.60)
    span = Span(span_ids=[0], original_text="denital", clean_text="denital")
    pool = CandidatePool(
        candidates=["denital", "dental"],
        sources={"denital": ["original"], "dental": ["domain_dictionary"]},
    )

    res = decide(val, span, pool, thresholds={"tau_general": 0.85, "tau_domain": 0.75})
    assert res.accepted is False
    assert res.threshold_type == "domain"
    assert res.threshold_used == 0.75
    assert res.reason == "below_confidence_threshold"


def test_decide_accepted_decision():
    val = ValidationResult(accepted=True, selected="dental", confidence=0.95)
    span = Span(span_ids=[0], original_text="denital", clean_text="denital")
    pool = CandidatePool(
        candidates=["denital", "dental"],
        sources={"denital": ["original"], "dental": ["domain_dictionary"]},
    )

    res = decide(val, span, pool, thresholds={"tau_general": 0.85, "tau_domain": 0.75})
    assert res.accepted is True
    assert res.replacement == "dental"
    assert res.threshold_type == "domain"
    assert res.threshold_used == 0.75
    assert res.reason is None


def test_decide_accepted_default_config_thresholds():
    val = ValidationResult(accepted=True, selected="dental", confidence=0.99)
    span = Span(span_ids=[0], original_text="denital", clean_text="denital")
    pool = CandidatePool(candidates=["denital", "dental"], sources={"denital": ["original"], "dental": ["symspell"]})

    res = decide(val, span, pool)
    assert res.accepted is True
    assert res.replacement == "dental"


def test_make_accepted_decision_factory():
    dec = make_accepted_decision(0.85, "general", "dental")
    assert dec.accepted is True
    assert dec.threshold_used == 0.85
    assert dec.threshold_type == "general"
    assert dec.replacement == "dental"
    assert dec.reason is None
