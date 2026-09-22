"""Single-line crop: bbox, padding, mask-to-white, RGB conversion (doc 05)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from ocr_pipeline.config import CropConfig
from ocr_pipeline.models import Polygon, RecognitionStatus, padded_bounds, polygon_bounds


@dataclass
class CropJob:
    line_id: int
    polygon: Polygon


@dataclass
class CropResult:
    line_id: int
    crop: Optional[Image.Image]
    status: RecognitionStatus


def _mask_outside_polygon(
    crop_array: np.ndarray,
    polygon: Polygon,
    offset_x: int,
    offset_y: int,
) -> np.ndarray:
    h, w = crop_array.shape[:2]
    local_poly = [(x - offset_x, y - offset_y) for x, y in polygon]
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).polygon(local_poly, fill=255)
    mask_arr = np.array(mask, dtype=bool)
    result = crop_array.copy()
    result[~mask_arr] = 255
    return result


def crop_single_line(
    page_array: np.ndarray,
    polygon: Polygon,
    config: CropConfig,
) -> Tuple[Optional[Image.Image], RecognitionStatus]:
    height, width = page_array.shape[:2]
    bounds = polygon_bounds(polygon)
    crop_bounds = padded_bounds(bounds, config.padding_px, width, height)
    x0, y0, x1, y1 = crop_bounds
    crop_w = x1 - x0 + 1
    crop_h = y1 - y0 + 1

    if crop_w < config.min_crop_size_px or crop_h < config.min_crop_size_px:
        return None, RecognitionStatus.DISCARDED_SMALL

    region = page_array[y0 : y1 + 1, x0 : x1 + 1].copy()
    masked = _mask_outside_polygon(region, polygon, x0, y0)
    pil = Image.fromarray(masked.astype(np.uint8))
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    return pil, RecognitionStatus.PENDING


def process_crop_job(page_array: np.ndarray, job: CropJob, config: CropConfig) -> CropResult:
    try:
        crop, status = crop_single_line(page_array, job.polygon, config)
        return CropResult(line_id=job.line_id, crop=crop, status=status)
    except Exception:
        return CropResult(
            line_id=job.line_id,
            crop=None,
            status=RecognitionStatus.DISCARDED_FAILED,
        )
