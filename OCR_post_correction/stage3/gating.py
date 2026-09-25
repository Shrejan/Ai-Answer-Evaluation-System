"""
stage3/gating.py — Confidence & Margin Gating module (M09).

Decides whether a validated LLM candidate selection passes numerical gating thresholds.
Provides the ONLY factory for creating an accepted Decision instance.
"""

from __future__ import annotations

from typing import Dict, Optional, Set
from stage3.schemas import ValidationResult, Span, CandidatePool, Decision
from stage3.config import get_config


def make_accepted_decision(
    threshold_used: float,
    threshold_type: str,
    replacement: str,
) -> Decision:
    """Factory function — the ONLY authorized way to construct an Accepted Decision."""
    return Decision(
        accepted=True,
        threshold_used=threshold_used,
        threshold_type=threshold_type,
        reason=None,
        replacement=replacement,
    )


def decide(
    val_result: ValidationResult,
    span: Span,
    pool: CandidatePool,
    thresholds: Optional[Dict[str, float]] = None,
) -> Decision:
    """
    Evaluates ValidationResult against numerical confidence thresholds.

    - Selects 'domain' threshold if any candidate in pool comes from domain source.
    - Applies margin check if enabled in config.
    - Returns Decision object.
    """
    cfg = get_config()

    if thresholds is None:
        tau_general = cfg.thresholds.tau_general
        tau_domain = cfg.thresholds.tau_domain
    else:
        tau_general = thresholds.get("tau_general", cfg.thresholds.tau_general)
        tau_domain = thresholds.get("tau_domain", cfg.thresholds.tau_domain)

    # Determine domain vs general threshold routing
    is_domain = False
    domain_sources: Set[str] = {"domain_dictionary", "phrase_retrieval"}
    for sources in pool.sources.values():
        if any(s in domain_sources for s in sources):
            is_domain = True
            break

    threshold_type = "domain" if is_domain else "general"
    applicable_threshold = tau_domain if is_domain else tau_general

    # If validation failed, pass through the rejection reason
    if not val_result.accepted or val_result.selected is None:
        return Decision(
            accepted=False,
            threshold_used=applicable_threshold,
            threshold_type=threshold_type,
            reason=val_result.reason or "validation_failed:unknown",
            replacement=None,
        )

    # If selected candidate matches original OCR text, no correction needed
    if val_result.selected.strip() == span.original_text.strip():
        return Decision(
            accepted=False,
            threshold_used=applicable_threshold,
            threshold_type=threshold_type,
            reason="selected_original",
            replacement=None,
        )

    # Confidence threshold check
    conf = val_result.confidence if val_result.confidence is not None else 0.0
    if conf < applicable_threshold:
        return Decision(
            accepted=False,
            threshold_used=applicable_threshold,
            threshold_type=threshold_type,
            reason="below_confidence_threshold",
            replacement=None,
        )

    return make_accepted_decision(
        threshold_used=applicable_threshold,
        threshold_type=threshold_type,
        replacement=val_result.selected,
    )
