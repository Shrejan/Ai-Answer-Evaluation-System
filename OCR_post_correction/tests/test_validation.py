"""
tests/test_validation.py — Unit tests for hard output validation module (M08).
"""

import math
import pytest
from stage3.validation import validate_qwen_output, _normalize


def test_validate_happy_path():
    candidate_set = {"denital", "dental"}
    raw = '{"selected_candidate": "dental", "confidence": 0.95}'
    res = validate_qwen_output(raw, candidate_set)

    assert res.accepted is True
    assert res.selected == "dental"
    assert res.confidence == 0.95
    assert res.reason is None


def test_validate_fenced_json_happy_path():
    candidate_set = {"denital", "dental"}
    raw = '```json\n{"selected_candidate": "dental", "confidence": 0.90}\n```'
    res = validate_qwen_output(raw, candidate_set)

    assert res.accepted is True
    assert res.selected == "dental"
    assert res.confidence == 0.90


def test_validate_think_tag_rejection():
    candidate_set = {"dental"}
    raw = '<think>I should pick dental</think>{"selected_candidate": "dental", "confidence": 0.9}'
    res = validate_qwen_output(raw, candidate_set)

    assert res.accepted is False
    assert res.reason == "validation_failed:check_think"


def test_validate_malformed_json():
    candidate_set = {"dental"}
    raw = '{"selected_candidate": "dental", "confidence": 0.95'
    res = validate_qwen_output(raw, candidate_set)

    assert res.accepted is False
    assert res.reason == "validation_failed:check_json_parse"


def test_validate_trailing_prose_rejection():
    candidate_set = {"dental"}
    raw = '{"selected_candidate": "dental", "confidence": 0.95}\nHope this helps!'
    res = validate_qwen_output(raw, candidate_set)

    assert res.accepted is False
    assert res.reason == "validation_failed:check_json_parse"


def test_validate_extra_keys():
    candidate_set = {"dental"}
    raw = '{"selected_candidate": "dental", "confidence": 0.95, "extra": "field"}'
    res = validate_qwen_output(raw, candidate_set)

    assert res.accepted is False
    assert res.reason == "validation_failed:check_schema_keys"


def test_validate_missing_keys():
    candidate_set = {"dental"}
    raw = '{"selected_candidate": "dental"}'
    res = validate_qwen_output(raw, candidate_set)

    assert res.accepted is False
    assert res.reason == "validation_failed:check_schema_keys"


def test_validate_selected_candidate_not_string():
    candidate_set = {"dental"}
    raw = '{"selected_candidate": 123, "confidence": 0.95}'
    res = validate_qwen_output(raw, candidate_set)
    assert res.accepted is False
    assert res.reason == "validation_failed:check_schema_keys"


def test_validate_invalid_types():
    candidate_set = {"dental"}
    raw = '{"selected_candidate": "dental", "confidence": "high"}'
    res = validate_qwen_output(raw, candidate_set)

    assert res.accepted is False
    assert res.reason == "validation_failed:check_schema_keys"

    raw_bool = '{"selected_candidate": "dental", "confidence": true}'
    res_bool = validate_qwen_output(raw_bool, candidate_set)
    assert res_bool.accepted is False
    assert res_bool.reason == "validation_failed:check_schema_keys"


def test_validate_not_in_candidate_set():
    candidate_set = {"denital", "dental"}
    raw = '{"selected_candidate": "dentist", "confidence": 0.99}'
    res = validate_qwen_output(raw, candidate_set)

    assert res.accepted is False
    assert res.reason == "validation_failed:check_candidate_set"


def test_validate_confidence_out_of_range():
    candidate_set = {"dental"}
    raw_high = '{"selected_candidate": "dental", "confidence": 1.5}'
    res_high = validate_qwen_output(raw_high, candidate_set)
    assert res_high.accepted is False
    assert res_high.reason == "validation_failed:check_confidence_range"

    raw_low = '{"selected_candidate": "dental", "confidence": -0.1}'
    res_low = validate_qwen_output(raw_low, candidate_set)
    assert res_low.accepted is False
    assert res_low.reason == "validation_failed:check_confidence_range"


def test_validate_nan_confidence():
    candidate_set = {"dental"}
    raw = '{"selected_candidate": "dental", "confidence": NaN}'
    res = validate_qwen_output(raw, candidate_set)
    assert res.accepted is False
    assert res.reason in ("validation_failed:check_json_parse", "validation_failed:check_confidence_range")


def test_normalize_helper():
    assert _normalize("  dental  ") == "dental"
    assert _normalize(None) == ""
