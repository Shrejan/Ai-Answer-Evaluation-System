"""End-to-end OCR pipeline orchestration."""

from __future__ import annotations

import io
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image

from ocr_pipeline.assembly.response_builder import build_success_response
from ocr_pipeline.config import AppConfig
from ocr_pipeline.correction.protected_vocab import load_protected_vocab
from ocr_pipeline.correction.spell_correct import SpellCorrector
from ocr_pipeline.cropping.pool_manager import crop_lines_parallel
from ocr_pipeline.detection.kraken_detector import detect_lines
from ocr_pipeline.detection.reading_order import assign_reading_order
from ocr_pipeline.diagnostics.timing import StageTimer
from ocr_pipeline.models import InternalWord, LineRecord, RecognitionStatus, image_to_numpy
from ocr_pipeline.recognition.gpu_queue import GpuInferenceQueue
from ocr_pipeline.recognition.trocr_engine import TrocrEngine
from ocr_pipeline.scoring.confidence import compute_word_confidences

logger = logging.getLogger(__name__)

WORD_SPLIT = re.compile(r"\S+")


@dataclass
class PipelineServices:
    config: AppConfig
    trocr: TrocrEngine
    gpu_queue: GpuInferenceQueue
    spell: SpellCorrector
    kraken_model: Optional[Any] = None


def decode_image(image_bytes: bytes) -> np.ndarray:
    try:
        pil = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise ValueError(f"Unreadable image: {exc}") from exc
    return image_to_numpy(pil)


def _word_char_spans(text: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in WORD_SPLIT.finditer(text)]


def _attach_words_to_line(
    line: LineRecord,
    text: str,
    confidences: List[float],
    spell_enabled: bool,
) -> None:
    spans = _word_char_spans(text)
    line.words = []
    for i, (start, end) in enumerate(spans):
        token = text[start:end]
        conf = confidences[i] if i < len(confidences) else 0.5
        corrected = False
        raw = line.raw_text or ""
        if spell_enabled and line.word_corrections:
            for info in line.word_corrections:
                if info.changed and info.corrected_word in token:
                    corrected = True
        line.words.append(
            InternalWord(
                id=-1,
                text=token,
                confidence=conf,
                line_id=line.line_id,
                raw_text=token,
                corrected=corrected,
            )
        )


def _compute_line_confidences(line: LineRecord, bundle) -> List[float]:
    output = line.generation_output
    if output is None or not line.raw_text:
        return [0.5] * len(_word_char_spans(line.raw_text or ""))
    seq = output.sequences[0]
    spans = _word_char_spans(line.raw_text)
    try:
        return compute_word_confidences(
            seq,
            list(output.scores),
            bundle.processor.tokenizer,
            spans,
        )
    except Exception:
        logger.exception("Confidence computation failed for line %d", line.line_id)
        return [0.5] * len(spans)


def run_pipeline(
    services: PipelineServices,
    image_bytes: bytes,
    *,
    request_id: Optional[str] = None,
    page_id=None,
    spell_correction: Optional[bool] = None,
) -> dict:
    req_id = request_id or str(uuid.uuid4())
    spell_enabled = (
        spell_correction
        if spell_correction is not None
        else services.config.spell.enabled
    )
    timer = StageTimer()

    timer.start()
    try:
        original = decode_image(image_bytes)
    except ValueError:
        raise
    timer.record("decode")

    timer.start()
    polygons = detect_lines(
        original,
        services.config.detection,
        kraken_model=services.kraken_model,
    )
    lines = assign_reading_order(polygons, services.config.reading_order)
    timer.record("detection")

    timer.start()
    lines = crop_lines_parallel(original, lines, services.config.crop)
    timer.record("cropping")

    timer.start()
    future = services.gpu_queue.submit(
        services.trocr.recognize_lines_batched,
        lines,
    )
    lines = future.result()
    timer.record("recognition")

    timer.start()
    bundle = services.trocr.bundle
    for line in lines:
        if line.recognition_status != RecognitionStatus.RECOGNIZED:
            continue
        confidences = _compute_line_confidences(line, bundle)
        if spell_enabled:
            corrected, corrections = services.spell.correct(line.raw_text or "")
            line.corrected_text = corrected
            line.word_corrections = corrections
            _attach_words_to_line(line, corrected, confidences, spell_enabled=True)
        else:
            line.corrected_text = line.raw_text
            _attach_words_to_line(line, line.raw_text or "", confidences, spell_enabled=False)
    timer.record("spell_correction")

    response = build_success_response(
        req_id,
        page_id,
        lines,
        timer,
        spell_enabled=spell_enabled,
    )
    logger.debug(
        "request=%s lines=%d recognized=%d words=%d",
        req_id,
        response.line_count,
        response.recognized_line_count,
        len(response.words),
    )
    return response.model_dump()
