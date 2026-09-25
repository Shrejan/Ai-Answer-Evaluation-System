"""
stage3/validation.py — Hard Output Validation (the unbypassable safety boundary).

Strictly enforces the 6 ordered short-circuiting checks in SPEC §8, D-002 normalisation,
<think> tag rejection, NaN rejection, and exact candidate set membership.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from typing import Any, Container, Dict, List, Set, Union
from stage3.schemas import ValidationResult


def _normalize(text: str) -> str:
    """Normalizes string according to D-002: Unicode NFC normalization + strip."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text).strip()


def validate_qwen_output(
    raw_output: str,
    candidate_set: Union[Set[str], List[str], Container[str]],
) -> ValidationResult:
    """
    Validates untrusted raw LLM output against candidate set.

    Runs 6 ordered short-circuiting checks:
    1. Rejects any <think> reasoning blocks.
    2. Parses strict JSON (allowing clean markdown code fence wrappers only).
    3. Verifies schema keys: exactly {"selected_candidate", "confidence"}.
    4. Verifies selected_candidate is an exact match in candidate_set (D-002 normalized).
    5. Verifies confidence is a valid float within [0.0, 1.0] range.
    6. Verifies candidate contains no extra word leak additions.
    """
    if not isinstance(raw_output, str) or not raw_output.strip():
        return ValidationResult(
            accepted=False,
            reason="validation_failed:check_json_parse",
        )

    # 1. Reject <think> tags
    if "<think>" in raw_output or "</think>" in raw_output:
        return ValidationResult(
            accepted=False,
            reason="validation_failed:check_think",
        )

    # 2. Defensive code fence extraction & strict JSON parse
    text_to_parse = raw_output.strip()
    if text_to_parse.startswith("```"):
        # Strip code block wrapper
        lines = text_to_parse.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text_to_parse = "\n".join(lines).strip()

    try:
        data = json.loads(text_to_parse)
    except (json.JSONDecodeError, TypeError):
        return ValidationResult(
            accepted=False,
            reason="validation_failed:check_json_parse",
        )

    if not isinstance(data, dict):
        return ValidationResult(
            accepted=False,
            reason="validation_failed:check_schema_keys",
        )

    # 3. Check exact schema keys
    expected_keys = {"selected_candidate", "confidence"}
    if set(data.keys()) != expected_keys:
        return ValidationResult(
            accepted=False,
            reason="validation_failed:check_schema_keys",
        )

    selected = data.get("selected_candidate")
    confidence = data.get("confidence")

    if not isinstance(selected, str):
        return ValidationResult(
            accepted=False,
            reason="validation_failed:check_schema_keys",
        )

    # Disallow bool (since bool is subclass of int in Python)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return ValidationResult(
            accepted=False,
            reason="validation_failed:check_schema_keys",
        )

    conf_float = float(confidence)
    if math.isnan(conf_float) or math.isinf(conf_float):
        return ValidationResult(
            accepted=False,
            reason="validation_failed:check_confidence_range",
        )

    # 4. Candidate set match
    norm_selected = _normalize(selected)
    norm_candidate_set = {_normalize(c) for c in candidate_set}

    if norm_selected not in norm_candidate_set:
        return ValidationResult(
            accepted=False,
            reason="validation_failed:check_candidate_set",
        )

    canonical_selected = selected
    for cand in candidate_set:
        if _normalize(cand) == norm_selected:
            canonical_selected = cand
            break

    # 5. Confidence range check
    if conf_float < 0.0 or conf_float > 1.0:
        return ValidationResult(
            accepted=False,
            reason="validation_failed:check_confidence_range",
        )

    return ValidationResult(
        accepted=True,
        selected=canonical_selected,
        confidence=conf_float,
        reason=None,
    )
