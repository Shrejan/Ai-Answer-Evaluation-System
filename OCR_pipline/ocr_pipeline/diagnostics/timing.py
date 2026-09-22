"""Per-stage timing helpers (doc 09)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict

from ocr_pipeline.api.schemas import TimingsMs


@dataclass
class StageTimer:
    decode: int = 0
    detection: int = 0
    cropping: int = 0
    recognition: int = 0
    spell_correction: int = 0
    _start: float = field(default=0.0, repr=False)

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> int:
        elapsed = int((time.perf_counter() - self._start) * 1000)
        return elapsed

    def record(self, stage: str) -> int:
        ms = self.stop()
        setattr(self, stage, ms)
        return ms

    def to_timings(self) -> TimingsMs:
        total = (
            self.decode
            + self.detection
            + self.cropping
            + self.recognition
            + self.spell_correction
        )
        return TimingsMs(
            decode=self.decode,
            detection=self.detection,
            cropping=self.cropping,
            recognition=self.recognition,
            spell_correction=self.spell_correction,
            total=total,
        )
