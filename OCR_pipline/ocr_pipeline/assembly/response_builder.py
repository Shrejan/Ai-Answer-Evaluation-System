"""Build API text + words from one ordered internal representation (doc 02, 11)."""

from __future__ import annotations

import re
from typing import List, Tuple

from ocr_pipeline.api.schemas import ModelInfo, OCRSuccessResponse, WordRecord
from ocr_pipeline.diagnostics.timing import StageTimer
from ocr_pipeline.models import InternalWord, LineRecord, RecognitionStatus

WORD_SPLIT = re.compile(r"\S+")


def _final_line_text(line: LineRecord, spell_enabled: bool) -> str:
    if line.recognition_status not in (
        RecognitionStatus.RECOGNIZED,
        RecognitionStatus.RECOGNIZED_EMPTY,
    ):
        return ""
    if spell_enabled and line.corrected_text is not None:
        return line.corrected_text
    return line.raw_text or ""


def build_page_words(lines: List[LineRecord], spell_enabled: bool) -> Tuple[str, List[InternalWord]]:
    """Single source of truth for page text and word records."""
    text_lines: List[str] = []
    words: List[InternalWord] = []
    word_id = 0

    for line in lines:
        line_text = _final_line_text(line, spell_enabled)
        if line_text:
            text_lines.append(line_text)

        if not line.words:
            for match in WORD_SPLIT.finditer(line_text):
                token = match.group()
                words.append(
                    InternalWord(
                        id=word_id,
                        text=token,
                        confidence=0.5,
                        line_id=line.line_id,
                        raw_text=token,
                        corrected=False,
                    )
                )
                word_id += 1
        else:
            for w in line.words:
                w.id = word_id
                words.append(w)
                word_id += 1

    return "\n".join(text_lines), words


def count_recognized_lines(lines: List[LineRecord]) -> int:
    return sum(
        1
        for line in lines
        if line.recognition_status == RecognitionStatus.RECOGNIZED
        and (line.raw_text or line.corrected_text)
    )


def build_success_response(
    request_id: str,
    page_id,
    lines: List[LineRecord],
    timer: StageTimer,
    *,
    spell_enabled: bool,
) -> OCRSuccessResponse:
    text, internal_words = build_page_words(lines, spell_enabled)
    word_records = [
        WordRecord(
            id=w.id,
            text=w.text,
            confidence=w.confidence,
            line_id=w.line_id,
            bbox=list(w.bbox) if w.bbox else None,
            raw_text=w.raw_text,
            corrected=w.corrected,
        )
        for w in internal_words
    ]
    return OCRSuccessResponse(
        request_id=request_id,
        page_id=page_id,
        text=text,
        words=word_records,
        line_count=len(lines),
        recognized_line_count=count_recognized_lines(lines),
        timings_ms=timer.to_timings(),
        model=ModelInfo(),
    )
