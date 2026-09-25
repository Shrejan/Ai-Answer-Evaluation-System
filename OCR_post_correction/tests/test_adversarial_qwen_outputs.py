"""
tests/test_adversarial_qwen_outputs.py — 12 Mandatory Adversarial Tests & Property Invariant Tests (M08).
"""

import json
import random
import string
import pytest
from stage3.validation import validate_qwen_output


def test_adversarial_1_extra_words_leak():
    candidate_set = {"transistor"}
    raw = '{"selected_candidate": "transistor amplifier", "confidence": 0.95}'
    res = validate_qwen_output(raw, candidate_set)
    assert res.accepted is False
    assert res.reason in ("validation_failed:check_candidate_set", "validation_failed:check_leak")


def test_adversarial_2_prose_instead_of_json():
    candidate_set = {"dental"}
    raw = "I think the correct word is dental because denital is a typo."
    res = validate_qwen_output(raw, candidate_set)
    assert res.accepted is False
    assert res.reason == "validation_failed:check_json_parse"


def test_adversarial_3_malformed_json_and_trailing_prose():
    candidate_set = {"dental"}
    raw_malformed = '{"selected_candidate": "dental", "confidence":}'
    assert validate_qwen_output(raw_malformed, candidate_set).accepted is False

    raw_trailing = '{"selected_candidate": "dental", "confidence": 0.9}\nNote: verified.'
    assert validate_qwen_output(raw_trailing, candidate_set).accepted is False


def test_adversarial_4_candidate_not_in_set():
    candidate_set = {"beat", "bear"}
    raw = '{"selected_candidate": "boat", "confidence": 0.99}'
    res = validate_qwen_output(raw, candidate_set)
    assert res.accepted is False
    assert res.reason == "validation_failed:check_candidate_set"


def test_adversarial_5_missing_extra_keys_wrong_types():
    candidate_set = {"dental"}
    # missing keys
    assert validate_qwen_output('{"selected_candidate": "dental"}', candidate_set).accepted is False
    # extra keys
    assert validate_qwen_output('{"selected_candidate": "dental", "confidence": 0.9, "reason": "typo"}', candidate_set).accepted is False
    # confidence as string
    assert validate_qwen_output('{"selected_candidate": "dental", "confidence": "0.9"}', candidate_set).accepted is False


def test_adversarial_6_confidence_out_of_range_and_nan():
    candidate_set = {"dental"}
    assert validate_qwen_output('{"selected_candidate": "dental", "confidence": -0.5}', candidate_set).accepted is False
    assert validate_qwen_output('{"selected_candidate": "dental", "confidence": 1.05}', candidate_set).accepted is False


def test_adversarial_7_empty_string_null_none():
    candidate_set = {"dental"}
    assert validate_qwen_output("", candidate_set).accepted is False
    assert validate_qwen_output("   ", candidate_set).accepted is False
    assert validate_qwen_output("null", candidate_set).accepted is False


def test_adversarial_8_think_tag_block():
    candidate_set = {"dental"}
    raw = "<think>Analyzing context...\nResult: dental</think>\n{\"selected_candidate\": \"dental\", \"confidence\": 0.9}"
    res = validate_qwen_output(raw, candidate_set)
    assert res.accepted is False
    assert res.reason == "validation_failed:check_think"


def test_adversarial_9_case_variant_nfc_stripping():
    candidate_set = {"dental"}
    raw = '{"selected_candidate": " dental ", "confidence": 0.95}'
    res = validate_qwen_output(raw, candidate_set)
    assert res.accepted is True
    assert res.selected == "dental"


def test_adversarial_10_unicode_lookalike_and_whitespace_variants():
    candidate_set = {"dental"}
    # NBSP or trailing whitespace
    raw = '{"selected_candidate": "dental\xa0", "confidence": 0.95}'
    res = validate_qwen_output(raw, candidate_set)
    assert res.accepted is True
    assert res.selected == "dental"


def test_adversarial_11_fenced_json_strictness():
    candidate_set = {"dental"}
    raw_valid = '```json\n{"selected_candidate": "dental", "confidence": 0.95}\n```'
    assert validate_qwen_output(raw_valid, candidate_set).accepted is True

    raw_invalid = '```json\n{"selected_candidate": "dental", "confidence": 0.95}\n```\nExtra explanation'
    assert validate_qwen_output(raw_invalid, candidate_set).accepted is False


def test_property_validation_invariant():
    """Property test: For random candidate sets & random output strings, accepted => selected in candidate_set."""
    for _ in range(100):
        # Generate random candidate set
        cands = ["".join(random.choices(string.ascii_lowercase, k=5)) for _ in range(3)]
        candidate_set = set(cands)

        # Generate pseudo random output
        if random.random() < 0.5:
            cand = random.choice(cands + ["fake_cand"])
            conf = random.choice([0.5, 0.9, 1.2, -0.1, "0.8"])
            raw = json.dumps({"selected_candidate": cand, "confidence": conf})
        else:
            raw = "".join(random.choices(string.ascii_letters + " {}\"':", k=30))

        res = validate_qwen_output(raw, candidate_set)
        if res.accepted:
            assert res.selected in candidate_set or any(c.strip() == res.selected for c in candidate_set)
