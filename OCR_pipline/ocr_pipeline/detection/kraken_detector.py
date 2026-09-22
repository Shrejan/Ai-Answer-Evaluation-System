"""Kraken blla line detection with coordinate restoration (doc 03)."""

from __future__ import annotations

import logging
import math
from typing import List

import numpy as np

from ocr_pipeline.config import DetectionConfig
from ocr_pipeline.detection.downscale import create_detection_image
from ocr_pipeline.models import Polygon, polygon_area, polygon_bounds

logger = logging.getLogger(__name__)


def _restore_coordinate(value: float, inv_scale: float, upper_bound: int) -> int:
    """Round outward slightly so strokes are not clipped inward."""
    restored = value * inv_scale
    if inv_scale >= 1.0:
        return int(max(0, min(round(restored), upper_bound)))
    # When mapping from smaller detection space to original, bias outward.
    return int(max(0, min(math.floor(restored + 0.5), upper_bound)))


def restore_polygon(
    polygon: Polygon,
    scale_factor: float,
    image_width: int,
    image_height: int,
) -> Polygon:
    inv = 1.0 / scale_factor
    restored: Polygon = []
    for x, y in polygon:
        rx = _restore_coordinate(x, inv, image_width - 1)
        ry = _restore_coordinate(y, inv, image_height - 1)
        restored.append((rx, ry))
    return restored


def clip_polygon_to_bounds(
    polygon: Polygon,
    image_width: int,
    image_height: int,
) -> Polygon:
    return [
        (
            max(0, min(x, image_width - 1)),
            max(0, min(y, image_height - 1)),
        )
        for x, y in polygon
    ]


def is_degenerate_polygon(polygon: Polygon) -> bool:
    if len(polygon) < 3:
        return True
    x0, y0, x1, y1 = polygon_bounds(polygon)
    if x1 <= x0 or y1 <= y0:
        return True
    return polygon_area(polygon) < 1.0


def _extract_polygons_from_kraken(detection_image: np.ndarray, kraken_model) -> List[Polygon]:
    from kraken import blla
    from ocr_pipeline.models import numpy_to_image

    pil = numpy_to_image(detection_image)
    seg = blla.segment(pil, model=kraken_model)
    polygons: List[Polygon] = []
    for line in seg.lines:
        boundary = getattr(line, "boundary", None)
        if boundary is None:
            continue
        poly: Polygon = [(int(round(x)), int(round(y))) for x, y in boundary]
        if poly:
            polygons.append(poly)
    return polygons


def detect_lines(
    original_image: np.ndarray,
    config: DetectionConfig,
    *,
    kraken_model=None,
    segment_fn=None,
) -> List[Polygon]:
    """Return polygons in original image coordinate space, clipped to bounds."""
    height, width = original_image.shape[:2]
    downscale = create_detection_image(original_image, config.max_dimension)

    if segment_fn is not None:
        raw_polygons = segment_fn(downscale.detection_image)
    elif kraken_model is not None:
        try:
            raw_polygons = _extract_polygons_from_kraken(
                downscale.detection_image,
                kraken_model,
            )
        except Exception:
            logger.exception("Kraken segmentation failed")
            raise
    else:
        raise RuntimeError(
            "Kraken model not loaded. Ensure blla.mlmodel is present and the "
            "server completed startup, or set OCR_KRAKEN_MODEL_PATH."
        )

    result: List[Polygon] = []
    for poly in raw_polygons:
        restored = restore_polygon(poly, downscale.scale_factor, width, height)
        clipped = clip_polygon_to_bounds(restored, width, height)
        if is_degenerate_polygon(clipped):
            logger.debug("Filtered degenerate polygon: %s", clipped)
            continue
        result.append(clipped)
    return result
