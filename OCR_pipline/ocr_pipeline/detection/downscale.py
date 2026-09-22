"""Detection-image downscaling (doc 03). Downscale only — never for recognition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image

from ocr_pipeline.models import numpy_to_image


@dataclass(frozen=True)
class DownscaleResult:
    detection_image: np.ndarray
    scale_factor: float


def measure_longest_side(image: np.ndarray) -> Tuple[int, int, int]:
    height, width = image.shape[:2]
    return width, height, max(width, height)


def create_detection_image(
    original_image: np.ndarray,
    max_dimension: int,
) -> DownscaleResult:
    """Build the temporary detection copy; original is never modified."""
    width, height, longest = measure_longest_side(original_image)
    if longest <= max_dimension:
        return DownscaleResult(detection_image=original_image, scale_factor=1.0)

    scale_factor = max_dimension / float(longest)
    new_w = max(1, int(round(width * scale_factor)))
    new_h = max(1, int(round(height * scale_factor)))

    pil = numpy_to_image(original_image)
    # Downscale only for Kraken detection — not enhancement upscaling.
    resized = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return DownscaleResult(
        detection_image=np.array(resized),
        scale_factor=scale_factor,
    )
