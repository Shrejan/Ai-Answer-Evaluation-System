"""
stage3/schemas.py — Pydantic v2 schemas for all Stage 3 post-OCR data contracts.
Includes input schemas, internal pipeline objects, validation/decision types,
and the final structured response schema required for evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator


# ==============================================================================
# 1. Input Contracts (Stage 2 Output -> Stage 3 Input)
# ==============================================================================

class Word(BaseModel):
    """Single OCR token with identifier, recognized text, and confidence score."""
    id: int = Field(..., description="0-indexed position within the page token stream")
    text: str = Field(..., description="Recognized string token")
    confidence: float = Field(..., ge=0.0, le=1.0, description="OCR confidence score in [0.0, 1.0]")

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Confidence {v} out of bounds [0.0, 1.0]")
        return v


class OCRInput(BaseModel):
    """Per-page JSON payload provided by Stage 2."""
    request_id: str = Field(..., description="Unique request identifier, e.g. page_001")
    student_id: Optional[str] = Field(None, description="Student ID, e.g. S2024_112")
    question_id: Optional[str] = Field(None, description="Question ID, e.g. Q3")
    page_id: Optional[int] = Field(1, description="Page number")
    text: str = Field(..., description="Full text as concatenated by Stage 2")
    words: List[Word] = Field(..., min_length=1, description="List of recognized word tokens")

    @field_validator("words")
    @classmethod
    def validate_words_non_empty(cls, v: List[Word]) -> List[Word]:
        if not v:
            raise ValueError("Input word list cannot be empty")
        return v


# ==============================================================================
# 2. Internal Pipeline Objects
# ==============================================================================

class Span(BaseModel):
    """Represents a flagged token or multi-token phrase span."""
    span_ids: List[int] = Field(..., description="Indices of word tokens included in this span")
    original_text: str = Field(..., description="Exact raw text of the span")
    prev_context: str = Field("", description="Leading context tokens (typically 5-8 tokens)")
    next_context: str = Field("", description="Trailing context tokens (typically 5-8 tokens)")
    is_phrase: bool = Field(False, description="True if span contains multiple contiguous tokens")
    leading_punct: str = Field("", description="Stripped leading punctuation")
    trailing_punct: str = Field("", description="Stripped trailing punctuation")
    clean_text: str = Field("", description="Normalized alphanumeric text without outer punctuation")
    char_start: int = Field(0, description="Character offset start in original text")
    char_end: int = Field(0, description="Character offset end in original text")


class SuspicionRecord(BaseModel):
    """Emitted by detector.py when a token or phrase triggers one or more suspicion rules."""
    token: str = Field(..., description="Flagged token text")
    span_ids: List[int] = Field(..., description="Token indices")
    reasons: List[str] = Field(..., description="List of rule IDs triggered")
    severity: float = Field(..., ge=0.0, le=1.0, description="Heuristic suspicion score")


class CandidatePool(BaseModel):
    """Aggregated, ranked, and capped pool of candidates for a flagged span."""
    candidates: List[str] = Field(..., description="Ranked list of candidates; original always present")
    sources: Dict[str, List[str]] = Field(default_factory=dict, description="Map of candidate -> source names")


class PromptBundle(BaseModel):
    """Payload passed to the LLM verifier."""
    system: str = Field(..., description="Fixed system instruction")
    user: str = Field(..., description="Context and candidate prompt")
    schema_dict: Dict[str, Any] = Field(..., alias="schema", description="Dynamic JSON-schema enum definition")


class ValidationResult(BaseModel):
    """Result of hard structural checks on LLM verifier output."""
    accepted: bool = Field(..., description="True if LLM output passed all 6 hard checks")
    selected: Optional[str] = Field(None, description="Selected candidate string if accepted")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Reported model confidence")
    reason: Optional[str] = Field(None, description="Rejection reason code, e.g. validation_failed:<check>")


class Decision(BaseModel):
    """Decision produced by gating.py determining whether to apply a candidate replacement."""
    accepted: bool = Field(..., description="True if candidate meets gating threshold")
    threshold_used: float = Field(..., description="Applied numerical threshold")
    threshold_type: Literal["general", "domain"] = Field(..., description="Which threshold was applied")
    reason: Optional[str] = Field(None, description="Rejection reason if accepted is False")
    replacement: Optional[str] = Field(None, description="Candidate to substitute if accepted is True")


# ==============================================================================
# 3. Output Contract & Audit Types
# ==============================================================================

class Correction(BaseModel):
    """Full audit record for a single flagged span."""
    original: str = Field(..., description="Original OCR text")
    replacement: str = Field(..., description="Replacement text (equals original if rejected)")
    position: List[int] = Field(..., min_length=2, max_length=2, description="[char_start, char_end]")
    candidate_set: List[str] = Field(..., description="Bounded candidate pool presented to verifier")
    candidate_source: List[str] = Field(..., description="Sources contributing to candidates")
    qwen_confidence: Optional[float] = Field(None, description="Model-reported confidence score")
    threshold_used: Optional[float] = Field(None, description="Threshold checked against")
    threshold_type: Optional[str] = Field(None, description="'general' or 'domain'")
    accepted: bool = Field(..., description="Whether replacement was approved and applied")
    rejection_reason: Optional[str] = Field(None, description="Rejection reason code or None")


class LowConfidenceSpan(BaseModel):
    """User-requested summary of unresolved or flagged low-confidence spans."""
    text: str = Field(..., description="Span text")
    span: List[int] = Field(..., min_length=2, max_length=2, description="[char_start, char_end]")
    confidence: float = Field(..., description="Confidence score")
    corrected: bool = Field(False, description="True if correction was applied")
    rejection_reason: Optional[str] = Field(None, description="Reason if left uncorrected")


class Segment(BaseModel):
    """Line or sentence segment metadata."""
    segment_id: int = Field(..., description="0-indexed segment number")
    text: str = Field(..., description="Segment text")
    confidence: Optional[float] = Field(None, description="Average confidence of words in segment")
    word_count: Optional[int] = Field(None, description="Number of tokens in segment")


class DocumentStructure(BaseModel):
    """Structural breakdown of the document."""
    paragraphs: List[int] = Field(default_factory=lambda: [0], description="Paragraph start indices")
    line_breaks_preserved: bool = Field(True, description="True if paragraph structure preserved")
    sentence_count: Optional[int] = Field(None, description="Total sentences detected in text")


class QualityFlags(BaseModel):
    """Aggregate quality indicators for Stage 4 evaluation."""
    avg_confidence: float = Field(..., description="Mean confidence of page tokens")
    min_confidence: float = Field(..., description="Minimum confidence observed")
    pct_corrected: float = Field(..., description="Fraction of page tokens altered by Stage 3")
    has_unresolved_uncertainty: bool = Field(..., description="True if low-confidence spans remain uncorrected")


class Stage3Metadata(BaseModel):
    """Provenance and runtime statistics for reproducibility."""
    corrections_applied: int = Field(..., description="Count of accepted edits")
    corrections_rejected: int = Field(..., description="Count of rejected or skipped spans")
    model_version: str = Field(..., description="Identifier of the verifier model")
    threshold_snapshot: str = Field(..., description="Config filename or version hash")
    latency_breakdown: Optional[Dict[str, float]] = Field(None, description="Component latency in ms")


class Stage3Result(BaseModel):
    """Comprehensive Stage 3 result object consumed by Stage 4 evaluation and logs."""
    request_id: str = Field(..., description="Request identifier")
    student_id: Optional[str] = Field(None, description="Student ID")
    question_id: Optional[str] = Field(None, description="Question ID")
    final_text: str = Field(..., description="Reconstituted text with proper sentence formatting")
    original_text: Optional[str] = Field(None, description="Raw OCR input text")
    corrected_text: Optional[str] = Field(None, description="Alias for final_text matching SPEC")
    segments: List[Segment] = Field(default_factory=list, description="Structured text segments")
    low_confidence_spans: List[LowConfidenceSpan] = Field(default_factory=list, description="Low confidence spans")
    structure: DocumentStructure = Field(default_factory=DocumentStructure, description="Paragraph/sentence layout")
    quality_flags: QualityFlags = Field(..., description="Quality metrics")
    stage3_metadata: Stage3Metadata = Field(..., description="Metadata and latency breakdown")
    corrections: List[Correction] = Field(default_factory=list, description="Detailed audit log of all corrections")
    stage3_latency_ms: Optional[Dict[str, float]] = Field(None, description="SPEC-compatible latency breakdown")

    @model_validator(mode="after")
    def sync_corrected_text(self) -> Stage3Result:
        if self.corrected_text is None:
            self.corrected_text = self.final_text
        return self
