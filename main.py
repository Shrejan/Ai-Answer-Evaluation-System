"""
OCR Post-Processing Pipeline for Handwritten Academic Answer Sheets.

Reconstructs noisy OCR text using FLAN-T5-small with strict constraints:
- Fix OCR noise, spelling, and punctuation only
- Never hallucinate, expand, or improve academic quality
- Preserve student intent and meaning
"""

from __future__ import annotations

import logging
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ocr_reconstruction")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "google/flan-t5-small"
MAX_INPUT_TOKENS = 512
MAX_OUTPUT_TOKENS = 256

# T5 generation parameters (as specified)
GENERATION_KWARGS: dict[str, Any] = {
    "max_new_tokens": MAX_OUTPUT_TOKENS,
    "temperature": 0.2,
    "num_beams": 3,
    "do_sample": False,
    "length_penalty": 0.8,
    "repetition_penalty": 1.1,
    "early_stopping": True,
}

# Conservative instruction prompt – explicitly forbids expansion
PROMPT_TEMPLATE = (
    "Correct OCR errors in this student answer. "
    "Fix spelling, punctuation, and OCR noise only. "
    "Do NOT add information, expand, explain, or improve the answer. "
    "Preserve the student's original meaning and academic level. "
    "Keep the same length and scope. "
    "Text: {text}"
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ReconstructRequest(BaseModel):
    """Single-text reconstruction request."""

    text: str = Field(..., min_length=1, description="Raw noisy OCR text")


class ReconstructBatchRequest(BaseModel):
    """Batch reconstruction request."""

    texts: list[str] = Field(..., min_length=1, description="List of raw OCR texts")


class ReconstructResponse(BaseModel):
    """Single reconstruction result."""

    raw: str
    preprocessed: str
    reconstructed: str
    confidence: Optional[float] = Field(
        None, description="Average log-probability confidence score (higher = more confident)"
    )
    processing_time_ms: float


class ReconstructBatchResponse(BaseModel):
    """Batch reconstruction result."""

    results: list[ReconstructResponse]
    total_processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model: str
    device: str
    gpu_available: bool


@dataclass
class ReconstructionResult:
    """Internal reconstruction result."""

    raw: str
    preprocessed: str
    reconstructed: str
    confidence: Optional[float] = None
    processing_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Stage 1: OCR Preprocessing
# ---------------------------------------------------------------------------


class OCRPreprocessor:
    """
    Cleans raw OCR output before model inference.

    - Normalizes whitespace
    - Removes repeated punctuation artifacts
    - Preserves bullet points and list structure
    - Strips common OCR noise characters
    """

    # Characters commonly misread by OCR that should be removed or replaced
    OCR_ARTIFACTS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f|~`^]")
    # Repeated punctuation: "!!!" -> "!", "???" -> "?"
    REPEATED_PUNCT = re.compile(r"([!?.,;:\-])\1{2,}")
    # Multiple spaces/tabs
    MULTI_SPACE = re.compile(r"[ \t]+")
    # Multiple newlines (keep at most double newline for paragraph breaks)
    MULTI_NEWLINE = re.compile(r"\n{3,}")
    # Stray characters between words (e.g., "w o r d" from broken OCR)
    BROKEN_WORD_SPACES = re.compile(r"\b(\w)\s+(?=\w\s+\w\b)")

    # Bullet point patterns to preserve
    BULLET_PATTERN = re.compile(r"^[\s]*([•\-\*\+]|\d+[\.\)])\s+", re.MULTILINE)

    @classmethod
    def preprocess(cls, text: str) -> str:
        """Apply full OCR preprocessing pipeline."""
        if not text or not text.strip():
            return text

        cleaned = text.strip()

        # Remove control characters and OCR artifacts
        cleaned = cls.OCR_ARTIFACTS.sub("", cleaned)

        # Normalize line endings
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse repeated punctuation (keep max 2 for emphasis like "!!")
        cleaned = cls.REPEATED_PUNCT.sub(r"\1\1", cleaned)

        # Normalize whitespace within lines but preserve newlines
        lines = []
        for line in cleaned.split("\n"):
            # Preserve bullet structure
            line = cls.MULTI_SPACE.sub(" ", line.strip())
            lines.append(line)
        cleaned = "\n".join(lines)

        # Limit excessive blank lines
        cleaned = cls.MULTI_NEWLINE.sub("\n\n", cleaned)

        # Fix common OCR substitutions before model (light touch)
        cleaned = cls._fix_common_ocr_substitutions(cleaned)

        return cleaned.strip()

    @staticmethod
    def _fix_common_ocr_substitutions(text: str) -> str:
        """
        Apply safe, deterministic OCR character fixes that do not change meaning.

        These are conservative replacements for well-known OCR confusions.
        """
        substitutions = {
            # Common letter/number confusions in OCR
            r"\b0\b(?=\s|$)": "O",  # standalone zero -> O (rare, context-dependent)
        }
        for pattern, replacement in substitutions.items():
            text = re.sub(pattern, replacement, text)
        return text


# ---------------------------------------------------------------------------
# Stage 2: Prompt Engineering
# ---------------------------------------------------------------------------


class PromptBuilder:
    """Builds conservative FLAN-T5 instruction prompts."""

    @staticmethod
    def build(text: str) -> str:
        """Create the instruction prompt with preprocessed OCR text."""
        return PROMPT_TEMPLATE.format(text=text.strip())


# ---------------------------------------------------------------------------
# Stage 3 & 4: Tokenization and T5 Inference
# ---------------------------------------------------------------------------


class T5Reconstructor:
    """
    FLAN-T5-small inference engine with GPU/CPU fallback.

    Loads model once and reuses for all requests.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        max_input_tokens: int = MAX_INPUT_TOKENS,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        generation_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.generation_kwargs = generation_kwargs or GENERATION_KWARGS.copy()

        self.device = self._resolve_device()
        logger.info("Loading model '%s' on device '%s'...", model_name, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        logger.info("Model loaded successfully.")

    @staticmethod
    def _resolve_device() -> torch.device:
        """Select GPU if available, otherwise CPU."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def reconstruct(
        self, preprocessed_text: str, compute_confidence: bool = True
    ) -> tuple[str, Optional[float]]:
        """
        Run T5 reconstruction on preprocessed text.

        Returns:
            Tuple of (reconstructed_text, confidence_score).
        """
        prompt = PromptBuilder.build(preprocessed_text)

        # Tokenize and truncate to max input tokens
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_input_tokens,
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        confidence: Optional[float] = None

        with torch.no_grad():
            if compute_confidence:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_scores=True,
                    return_dict_in_generate=True,
                    **self.generation_kwargs,
                )
                generated_ids = outputs.sequences
                confidence = self._compute_confidence(outputs)
            else:
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **self.generation_kwargs,
                )

        reconstructed = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        ).strip()

        return reconstructed, confidence

    @staticmethod
    def _compute_confidence(outputs: Any) -> Optional[float]:
        """
        Compute average log-probability across generated tokens.

        Higher values indicate the model is more confident in its output.
        Returns None if scores are unavailable.
        """
        if not hasattr(outputs, "scores") or outputs.scores is None:
            return None

        try:
            log_probs = []
            for step_scores in outputs.scores:
                # step_scores: (batch, vocab_size)
                probs = torch.log_softmax(step_scores, dim=-1)
                max_log_prob = probs.max(dim=-1).values.item()
                log_probs.append(max_log_prob)

            if log_probs:
                return round(sum(log_probs) / len(log_probs), 4)
        except Exception as exc:
            logger.warning("Failed to compute confidence: %s", exc)

        return None


# ---------------------------------------------------------------------------
# Stage 5: Postprocessing
# ---------------------------------------------------------------------------


class OutputPostprocessor:
    """
    Minimal post-inference cleanup.

    - Remove duplicate consecutive words
    - Stabilize punctuation spacing
    - Enforce sentence capitalization (first letter only)
    - No semantic changes
    """

    DUPLICATE_WORDS = re.compile(r"\b(\w+)(\s+\1\b)+", re.IGNORECASE)
    SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,!?;:])")
    MISSING_SPACE_AFTER_PUNCT = re.compile(r"([.,!?;:])([^\s\d])")
    MULTI_SPACE = re.compile(r" {2,}")

    @classmethod
    def postprocess(cls, text: str) -> str:
        """Apply minimal postprocessing to model output."""
        if not text or not text.strip():
            return text

        cleaned = text.strip()

        # Remove duplicate consecutive words (OCR/model artifact)
        cleaned = cls.DUPLICATE_WORDS.sub(r"\1", cleaned)

        # Fix punctuation spacing
        cleaned = cls.SPACE_BEFORE_PUNCT.sub(r"\1", cleaned)
        cleaned = cls.MISSING_SPACE_AFTER_PUNCT.sub(r"\1 \2", cleaned)

        # Collapse multiple spaces
        cleaned = cls.MULTI_SPACE.sub(" ", cleaned)

        # Capitalize first character of text (minimal readability fix)
        cleaned = cls._capitalize_first(cleaned)

        # Ensure text ends with punctuation if it looks like a complete sentence
        cleaned = cls._ensure_terminal_punctuation(cleaned)

        return cleaned.strip()

    @staticmethod
    def _capitalize_first(text: str) -> str:
        """Capitalize the first alphabetic character only."""
        for i, char in enumerate(text):
            if char.isalpha():
                return text[:i] + char.upper() + text[i + 1 :]
        return text

    @staticmethod
    def _ensure_terminal_punctuation(text: str) -> str:
        """
        Add a period at the end if the text is a single sentence fragment
        without terminal punctuation. Conservative: only for short answers.
        """
        if not text:
            return text

        # Do not modify multi-line bullet lists
        if "\n" in text:
            return text

        terminal_chars = ".!?;"
        if text[-1] not in terminal_chars:
            # Only add period for short answer-like text (< 200 chars)
            if len(text) < 200 and text[-1].isalnum():
                return text + "."
        return text


# ---------------------------------------------------------------------------
# Reconstruction Engine (Orchestrator)
# ---------------------------------------------------------------------------


class ReconstructionEngine:
    """
    End-to-end OCR reconstruction pipeline.

    Coordinates preprocessing, inference, and postprocessing stages.
    """

    def __init__(self, reconstructor: T5Reconstructor) -> None:
        self.reconstructor = reconstructor

    def process(
        self, raw_text: str, compute_confidence: bool = True
    ) -> ReconstructionResult:
        """Run the full reconstruction pipeline on a single text."""
        start = time.perf_counter()

        raw = raw_text
        preprocessed = OCRPreprocessor.preprocess(raw)
        model_output, confidence = self.reconstructor.reconstruct(
            preprocessed, compute_confidence=compute_confidence
        )
        reconstructed = OutputPostprocessor.postprocess(model_output)

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Reconstructed %d chars -> %d chars (%.1f ms, confidence=%s)",
            len(raw),
            len(reconstructed),
            elapsed_ms,
            confidence,
        )

        return ReconstructionResult(
            raw=raw,
            preprocessed=preprocessed,
            reconstructed=reconstructed,
            confidence=confidence,
            processing_time_ms=round(elapsed_ms, 2),
        )

    def process_batch(
        self, raw_texts: list[str], compute_confidence: bool = True
    ) -> list[ReconstructionResult]:
        """Process multiple texts sequentially (model is shared)."""
        return [self.process(text, compute_confidence) for text in raw_texts]


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

# Global engine instance – initialized at startup
_engine: Optional[ReconstructionEngine] = None
_reconstructor: Optional[T5Reconstructor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup, release on shutdown."""
    global _engine, _reconstructor

    logger.info("Starting OCR reconstruction service...")
    _reconstructor = T5Reconstructor()
    _engine = ReconstructionEngine(_reconstructor)
    logger.info("Service ready.")

    yield

    logger.info("Shutting down OCR reconstruction service.")
    _engine = None
    _reconstructor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="OCR Reconstruction API",
    description=(
        "Production-grade OCR post-processing for handwritten academic answer sheets. "
        "Corrects OCR noise while strictly preserving student meaning."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _get_engine() -> ReconstructionEngine:
    """Return the initialized engine or raise 503."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Service not ready. Model still loading.")
    return _engine


def _result_to_response(result: ReconstructionResult) -> ReconstructResponse:
    """Convert internal result to API response."""
    return ReconstructResponse(
        raw=result.raw,
        preprocessed=result.preprocessed,
        reconstructed=result.reconstructed,
        confidence=result.confidence,
        processing_time_ms=result.processing_time_ms,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    device = str(_reconstructor.device) if _reconstructor else "unknown"
    return HealthResponse(
        status="ok" if _engine is not None else "loading",
        model=MODEL_NAME,
        device=device,
        gpu_available=torch.cuda.is_available(),
    )


@app.post("/reconstruct", response_model=ReconstructResponse)
async def reconstruct(request: ReconstructRequest) -> ReconstructResponse:
    """
    Reconstruct a single noisy OCR text.

    Returns raw, preprocessed, and final reconstructed text.
    """
    engine = _get_engine()

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        result = engine.process(request.text.strip())
        return _result_to_response(result)
    except Exception as exc:
        logger.exception("Reconstruction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Reconstruction failed: {exc}") from exc


@app.post("/reconstruct_batch", response_model=ReconstructBatchResponse)
async def reconstruct_batch(request: ReconstructBatchRequest) -> ReconstructBatchResponse:
    """
    Reconstruct a batch of noisy OCR texts.

    Processes each text through the full pipeline and returns all results.
    """
    engine = _get_engine()

    texts = [t.strip() for t in request.texts]
    if any(not t for t in texts):
        raise HTTPException(status_code=400, detail="All texts must be non-empty.")

    start = time.perf_counter()

    try:
        results = engine.process_batch(texts)
        total_ms = round((time.perf_counter() - start) * 1000, 2)
        return ReconstructBatchResponse(
            results=[_result_to_response(r) for r in results],
            total_processing_time_ms=total_ms,
        )
    except Exception as exc:
        logger.exception("Batch reconstruction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Batch reconstruction failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
