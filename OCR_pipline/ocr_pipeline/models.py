"""Shared internal pipeline data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image

Polygon = List[Tuple[int, int]]
Bounds = Tuple[int, int, int, int]  # x0, y0, x1, y1


class RecognitionStatus(str, Enum):
    PENDING = "pending"
    RECOGNIZED = "recognized"
    RECOGNIZED_EMPTY = "recognized_empty"
    DISCARDED_SMALL = "discarded_small"
    DISCARDED_FAILED = "discarded_failed"


@dataclass
class WordCorrectionInfo:
    raw_word: str
    corrected_word: str
    changed: bool
    edit_distance: int = 0


@dataclass
class InternalWord:
    """Word in the ordered page representation (pre-API)."""

    id: int
    text: str
    confidence: float
    line_id: int
    bbox: Optional[Bounds] = None
    raw_text: Optional[str] = None
    corrected: bool = False


@dataclass
class LineRecord:
    line_id: int
    polygon: Polygon
    bounds: Bounds
    vertical_center: float
    estimated_height: float
    recognition_status: RecognitionStatus = RecognitionStatus.PENDING
    recognized_text: Optional[str] = None
    raw_text: Optional[str] = None
    corrected_text: Optional[str] = None
    crop: Optional[Image.Image] = None
    generation_output: Optional[Any] = None
    line_confidence: Optional[float] = None
    word_corrections: List[WordCorrectionInfo] = field(default_factory=list)
    words: List[InternalWord] = field(default_factory=list)


def polygon_bounds(polygon: Polygon) -> Bounds:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def polygon_area(polygon: Polygon) -> float:
    if len(polygon) < 3:
        return 0.0
    area = 0.0
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0


def bounds_from_polygon(polygon: Polygon) -> Tuple[Bounds, float, float]:
    x0, y0, x1, y1 = polygon_bounds(polygon)
    height = max(y1 - y0, 1)
    vertical_center = (y0 + y1) / 2.0
    return (x0, y0, x1, y1), vertical_center, float(height)


def clip_bounds(bounds: Bounds, width: int, height: int) -> Bounds:
    x0, y0, x1, y1 = bounds
    return (
        max(0, min(x0, width - 1)),
        max(0, min(y0, height - 1)),
        max(0, min(x1, width - 1)),
        max(0, min(y1, height - 1)),
    )


def padded_bounds(bounds: Bounds, padding: int, width: int, height: int) -> Bounds:
    x0, y0, x1, y1 = bounds
    return clip_bounds(
        (x0 - padding, y0 - padding, x1 + padding, y1 + padding),
        width,
        height,
    )


def image_to_numpy(image: Image.Image) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def numpy_to_image(array: np.ndarray) -> Image.Image:
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)
    return Image.fromarray(array)
