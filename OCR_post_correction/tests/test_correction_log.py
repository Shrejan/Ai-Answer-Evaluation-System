"""
tests/test_correction_log.py — Tests for append-only correction audit logger (M11).
"""

import os
import pytest
from stage3.schemas import Correction, Stage3Result, QualityFlags, Stage3Metadata
from stage3.correction_log import log_correction_event, read_correction_logs, get_prompt_hash


def test_prompt_hash():
    phash = get_prompt_hash()
    assert isinstance(phash, str)
    assert len(phash) == 16


def test_roundtrip_logging(tmp_path):
    log_file = tmp_path / "test_log.jsonl"

    res1 = Stage3Result(
        request_id="p1",
        original_text="medical denital camp",
        final_text="medical dental camp",
        corrected_text="medical dental camp",
        quality_flags=QualityFlags(
            avg_confidence=0.9, min_confidence=0.4, pct_corrected=0.33, has_unresolved_uncertainty=False
        ),
        stage3_metadata=Stage3Metadata(
            corrections_applied=1,
            corrections_rejected=0,
            model_version="qwen3:1.7b",
            threshold_snapshot="thresholds.yaml",
        ),
        corrections=[
            Correction(
                original="denital",
                replacement="dental",
                position=[1, 1],
                candidate_set=["denital", "dental"],
                candidate_source=["domain_dictionary"],
                qwen_confidence=0.95,
                threshold_used=0.75,
                threshold_type="domain",
                accepted=True,
                rejection_reason=None,
            )
        ],
        stage3_latency_ms={"total_ms": 12.5},
    )

    log_correction_event(res1, log_path=log_file)

    logs = read_correction_logs(log_file)
    assert len(logs) == 1
    assert logs[0]["request_id"] == "p1"
    assert logs[0]["corrected_text"] == "medical dental camp"
    assert logs[0]["config_snapshot"]["model"] == "qwen3:1.7b"
    assert len(logs[0]["config_snapshot"]["prompt_hash"]) == 16


def test_append_only_preserves_existing_logs(tmp_path):
    log_file = tmp_path / "append_test.jsonl"

    res1 = Stage3Result(
        request_id="page_1",
        original_text="page 1 orig",
        final_text="page 1 corr",
        quality_flags=QualityFlags(avg_confidence=0.9, min_confidence=0.8, pct_corrected=0.0, has_unresolved_uncertainty=False),
        stage3_metadata=Stage3Metadata(corrections_applied=0, corrections_rejected=0, model_version="qwen3:1.7b", threshold_snapshot="t.yaml"),
    )
    res2 = Stage3Result(
        request_id="page_2",
        original_text="page 2 orig",
        final_text="page 2 corr",
        quality_flags=QualityFlags(avg_confidence=0.9, min_confidence=0.8, pct_corrected=0.0, has_unresolved_uncertainty=False),
        stage3_metadata=Stage3Metadata(corrections_applied=0, corrections_rejected=0, model_version="qwen3:1.7b", threshold_snapshot="t.yaml"),
    )

    log_correction_event(res1, log_path=log_file)
    log_correction_event(res2, log_path=log_file)

    logs = read_correction_logs(log_file)
    assert len(logs) == 2
    assert logs[0]["request_id"] == "page_1"
    assert logs[1]["request_id"] == "page_2"
