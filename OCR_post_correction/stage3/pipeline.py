"""
stage3/pipeline.py — End-to-end Stage 3 Post-OCR Correction Pipeline.

Orchestrates detection, candidate aggregation, Qwen verification, validation, gating,
and sentence reconstitution according to SPEC §1 & §3.
"""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Set

from stage3.schemas import (
    Word,
    Span,
    CandidatePool,
    ValidationResult,
    Decision,
    Correction,
    Stage3Result,
)
from stage3.detector import SuspicionDetector
from stage3.candidates.aggregator import build_candidate_pool, collect_and_build_candidate_pool
from stage3.qwen.client import LLMClient, OllamaClient
from stage3.qwen.verifier import verify_span, VerifierError
from stage3.validation import validate_qwen_output
from stage3.gating import decide
from stage3.config import get_config


logger = logging.getLogger(__name__)


def _preserve_case_and_punct(original: str, replacement: str) -> str:
    """
    Preserves case and re-attaches punctuation per D-002 and D-004.
    """
    if not original or not replacement:
        return replacement or original

    # Strip leading/trailing punctuation from original to find core
    leading_punct = ""
    trailing_punct = ""

    start_idx = 0
    while start_idx < len(original) and not original[start_idx].isalnum():
        leading_punct += original[start_idx]
        start_idx += 1

    end_idx = len(original) - 1
    while end_idx >= start_idx and not original[end_idx].isalnum():
        trailing_punct = original[end_idx] + trailing_punct
        end_idx -= 1

    core_orig = original[start_idx : end_idx + 1]
    core_repl = replacement.strip()

    # Case match
    if core_orig.isupper():
        core_repl = core_repl.upper()
    elif core_orig and core_orig[0].isupper():
        core_repl = core_repl.capitalize()

    return f"{leading_punct}{core_repl}{trailing_punct}"


def apply_accepted(span: Span, decision: Decision) -> str:
    """
    The ONLY authorized path to construct corrected text from an AcceptedDecision.
    Raises ValueError if decision.accepted is False.
    """
    if not decision.accepted or decision.replacement is None:
        raise ValueError("Cannot apply a rejected decision")

    return _preserve_case_and_punct(span.original_text, decision.replacement)


def correct_page(
    ocr_json: Dict[str, Any],
    client: Optional[LLMClient] = None,
) -> Stage3Result:
    """
    Main entrypoint: runs complete Stage 3 post-OCR correction on a Stage 2 page payload.
    """
    start_total = time.perf_counter()

    if client is None:
        client = OllamaClient()

    request_id = ocr_json.get("request_id", "page_unknown")

    # 1. Parse input words and text
    raw_words = ocr_json.get("words", [])
    words: List[Word] = []

    if raw_words:
        for idx, w in enumerate(raw_words):
            if isinstance(w, dict):
                words.append(
                    Word(
                        id=w.get("id", idx),
                        text=str(w.get("text", "")),
                        confidence=float(w.get("confidence", 1.0)),
                    )
                )
            elif isinstance(w, Word):
                words.append(w)
    else:
        text_str = str(ocr_json.get("text", ""))
        for idx, tok in enumerate(text_str.split()):
            words.append(Word(id=idx, text=tok, confidence=1.0))

    original_text = ocr_json.get("text")
    if not original_text:
        original_text = " ".join(w.text for w in words)

    detector_start = time.perf_counter()
    detector = SuspicionDetector()
    spans = detector.detect_spans(words)
    detector_ms = (time.perf_counter() - detector_start) * 1000.0

    corrections: List[Correction] = []
    cand_gen_ms = 0.0
    qwen_infer_ms = 0.0
    validation_ms = 0.0

    word_replacements: Dict[int, str] = {}

    # 3. Candidate pool generation & single candidate filtering
    spans_to_verify: List[Tuple[Span, CandidatePool]] = []

    for span in spans:
        cg_start = time.perf_counter()
        pool = collect_and_build_candidate_pool(span)
        cand_gen_ms += (time.perf_counter() - cg_start) * 1000.0

        if len(pool.candidates) <= 1:
            corrections.append(
                Correction(
                    original=span.original_text,
                    replacement=span.original_text,
                    position=[span.span_ids[0], span.span_ids[-1]],
                    candidate_set=pool.candidates,
                    candidate_source=pool.sources.get(span.original_text, ["original"]),
                    qwen_confidence=None,
                    threshold_used=0.0,
                    threshold_type="general",
                    accepted=False,
                    rejection_reason="single_candidate_skip",
                )
            )
        else:
            spans_to_verify.append((span, pool))

    # 4. Parallel LLM Verification for multi-candidate spans
    llm_start = time.perf_counter()

    def _verify_item(item: Tuple[Span, CandidatePool]) -> Tuple[Span, CandidatePool, str, Optional[VerifierError]]:
        sp, pl = item
        try:
            out = verify_span(client, sp, pl)
            return (sp, pl, out, None)
        except VerifierError as err:
            return (sp, pl, "", err)

    verification_results = []
    if spans_to_verify:
        from concurrent.futures import ThreadPoolExecutor
        max_workers = min(4, len(spans_to_verify))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            verification_results = list(executor.map(_verify_item, spans_to_verify))

    qwen_infer_ms += (time.perf_counter() - llm_start) * 1000.0

    # 5. Validation, Gating, and Replacement Application
    val_start = time.perf_counter()
    for span, pool, raw_output, verifier_err in verification_results:
        if verifier_err is not None:
            val_result = ValidationResult(
                accepted=False,
                reason=verifier_err.error_type,
            )
        else:
            val_result = validate_qwen_output(raw_output, pool.candidates)

        decision = decide(val_result, span, pool)

        if decision.accepted:
            corrected_span_text = apply_accepted(span, decision)
            if len(span.span_ids) == 1:
                word_replacements[span.span_ids[0]] = corrected_span_text
            else:
                word_replacements[span.span_ids[0]] = corrected_span_text
                for wid in span.span_ids[1:]:
                    word_replacements[wid] = ""

            cand_sources = pool.sources.get(decision.replacement or "", [])
            corrections.append(
                Correction(
                    original=span.original_text,
                    replacement=corrected_span_text,
                    position=[span.span_ids[0], span.span_ids[-1]],
                    candidate_set=pool.candidates,
                    candidate_source=cand_sources,
                    qwen_confidence=val_result.confidence,
                    threshold_used=decision.threshold_used,
                    threshold_type=decision.threshold_type,
                    accepted=True,
                    rejection_reason=None,
                )
            )
        else:
            cand_sources = pool.sources.get(val_result.selected or span.original_text, ["original"])
            corrections.append(
                Correction(
                    original=span.original_text,
                    replacement=span.original_text,
                    position=[span.span_ids[0], span.span_ids[-1]],
                    candidate_set=pool.candidates,
                    candidate_source=cand_sources,
                    qwen_confidence=val_result.confidence,
                    threshold_used=decision.threshold_used,
                    threshold_type=decision.threshold_type,
                    accepted=False,
                    rejection_reason=decision.reason,
                )
            )

    validation_ms += (time.perf_counter() - val_start) * 1000.0

    # 4. Sentence reconstitution
    corrected_tokens: List[str] = []
    for w in words:
        if w.id in word_replacements:
            repl = word_replacements[w.id]
            if repl:
                corrected_tokens.append(repl)
        else:
            corrected_tokens.append(w.text)

    corrected_text = " ".join(corrected_tokens)
    total_ms = (time.perf_counter() - start_total) * 1000.0

    latency_dict = {
        "candidate_generation_ms": round(cand_gen_ms, 2),
        "qwen_inference_ms": round(qwen_infer_ms, 2),
        "validation_ms": round(validation_ms, 2),
        "total_ms": round(total_ms, 2),
    }

    corrections_applied = sum(1 for c in corrections if c.accepted)
    corrections_rejected = sum(1 for c in corrections if not c.accepted)
    total_tokens = len(words) or 1

    from stage3.schemas import QualityFlags, Stage3Metadata

    quality_flags = QualityFlags(
        avg_confidence=sum(w.confidence for w in words) / total_tokens if words else 1.0,
        min_confidence=min((w.confidence for w in words), default=1.0),
        pct_corrected=round(corrections_applied / total_tokens, 4),
        has_unresolved_uncertainty=corrections_rejected > 0,
    )

    metadata = Stage3Metadata(
        corrections_applied=corrections_applied,
        corrections_rejected=corrections_rejected,
        model_version="qwen3:1.7b",
        threshold_snapshot="thresholds.yaml",
        latency_breakdown=latency_dict,
    )

    return Stage3Result(
        request_id=request_id,
        original_text=original_text,
        final_text=corrected_text,
        corrected_text=corrected_text,
        quality_flags=quality_flags,
        stage3_metadata=metadata,
        corrections=corrections,
        stage3_latency_ms=latency_dict,
    )


def process_ocr_request(
    ocr_json: Dict[str, Any],
    client: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """
    Public API helper taking Stage 2 OCR JSON request and returning full JSON response.
    """
    res = correct_page(ocr_json, client=client)

    corrections_applied = sum(1 for c in res.corrections if c.accepted)
    corrections_rejected = sum(1 for c in res.corrections if not c.accepted)

    return {
        "request_id": res.request_id,
        "original_text": res.original_text,
        "final_text": res.corrected_text,
        "corrections": [c.model_dump() for c in res.corrections],
        "quality_flags": {
            "corrections_applied": corrections_applied,
            "corrections_rejected": corrections_rejected,
            "has_unresolved_uncertainty": corrections_rejected > 0,
        },
        "stage3_metadata": {
            "corrections_applied": corrections_applied,
            "corrections_rejected": corrections_rejected,
            "stage3_latency_ms": res.stage3_latency_ms,
        },
    }
