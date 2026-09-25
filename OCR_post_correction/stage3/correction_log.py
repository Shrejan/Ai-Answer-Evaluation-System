"""
stage3/correction_log.py — Append-Only Correction Audit Logger (M11).

Persists one JSONL record per processed page containing corrections, latencies,
and complete config provenance snapshots. File is strictly append-only.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from stage3.schemas import Stage3Result
from stage3.config import get_config
from stage3.qwen.prompt_templates import SYSTEM_PROMPT


def get_prompt_hash() -> str:
    """Returns SHA256 hex digest of the system prompt for audit provenance."""
    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]


def get_config_snapshot() -> Dict[str, Any]:
    """Generates snapshot dictionary recording runtime configuration state."""
    cfg = get_config()
    return {
        "thresholds_version": "thresholds.yaml",
        "tau_general": cfg.thresholds.tau_general,
        "tau_domain": cfg.thresholds.tau_domain,
        "model": cfg.model.model,
        "prompt_hash": get_prompt_hash(),
        "dictionary_version": "v1.0",
        "confusion_matrix_hash": "seed_v1",
    }


def log_correction_event(
    result: Stage3Result,
    log_path: Optional[Union[Path, str]] = None,
) -> None:
    """
    Appends one JSON line to the correction log file. Never overwrites existing logs.
    """
    if log_path is None:
        log_dir = Path("data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "correction_log.jsonl"
    else:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "request_id": result.request_id,
        "student_id": result.student_id,
        "question_id": result.question_id,
        "original_text": result.original_text,
        "corrected_text": result.corrected_text or result.final_text,
        "corrections": [c.model_dump() for c in result.corrections],
        "stage3_latency_ms": result.stage3_latency_ms,
        "quality_flags": result.quality_flags.model_dump() if result.quality_flags else None,
        "config_snapshot": get_config_snapshot(),
    }

    line = json.dumps(record, ensure_ascii=False) + "\n"

    # Append mode only
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)


def read_correction_logs(
    log_path: Optional[Union[Path, str]] = None,
) -> List[Dict[str, Any]]:
    """Reads all lines from the append-only JSONL correction log file."""
    if log_path is None:
        log_path = Path("data/logs/correction_log.jsonl")
    else:
        log_path = Path(log_path)

    if not log_path.exists():
        return []

    logs = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                logs.append(json.loads(line_str))
    return logs
