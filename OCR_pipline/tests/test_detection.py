"""Phase 1 detection tests."""

import numpy as np
import pytest

from ocr_pipeline.config import DetectionConfig, ReadingOrderConfig
from ocr_pipeline.detection.downscale import create_detection_image, measure_longest_side
from ocr_pipeline.detection.kraken_detector import (
    detect_lines,
    is_degenerate_polygon,
    restore_polygon,
)
from ocr_pipeline.detection.reading_order import assign_reading_order


def _blank_page(w: int, h: int) -> np.ndarray:
    return np.full((h, w, 3), 255, dtype=np.uint8)


def test_measure_longest_side():
    img = _blank_page(800, 1200)
    assert measure_longest_side(img) == (800, 1200, 1200)


def test_downscale_when_oversized():
    img = _blank_page(4000, 3000)
    result = create_detection_image(img, max_dimension=2048)
    assert result.scale_factor == pytest.approx(2048 / 4000)
    h, w = result.detection_image.shape[:2]
    assert max(w, h) == 2048
    assert result.detection_image is not img


def test_no_downscale_when_within_limit():
    img = _blank_page(800, 600)
    result = create_detection_image(img, max_dimension=2048)
    assert result.scale_factor == 1.0
    assert result.detection_image is img


def test_restore_polygon_roundtrip():
    poly = [(100, 50), (300, 50), (300, 80), (100, 80)]
    restored = restore_polygon(poly, scale_factor=0.5, image_width=799, image_height=599)
    assert restored[0][0] == 200
    assert restored[0][1] == 100


def test_degenerate_polygon_filtered():
    assert is_degenerate_polygon([(0, 0), (0, 0)])
    assert is_degenerate_polygon([(0, 0), (10, 0), (10, 0)])


def test_detect_lines_with_mock_segment_fn():
    img = _blank_page(1000, 800)

    def mock_segment(detection_image):
        h, w = detection_image.shape[:2]
        return [[(10, 10), (w - 10, 10), (w - 10, 40), (10, 40)]]

    polys = detect_lines(
        img,
        DetectionConfig(max_dimension=2048),
        segment_fn=mock_segment,
    )
    assert len(polys) == 1
    assert polys[0][0][0] == 10


def test_detect_lines_oversized_page_uses_downscale(mock_scale=0.5):
    img = _blank_page(4000, 3000)
    seen = {}

    def mock_segment(detection_image):
        seen["shape"] = detection_image.shape[:2]
        return [[(100, 100), (500, 100), (500, 150), (100, 150)]]

    polys = detect_lines(img, DetectionConfig(max_dimension=2048), segment_fn=mock_segment)
    assert max(seen["shape"]) == 2048
    assert len(polys) == 1
    assert polys[0][1][0] > 500  # coordinates restored to original space


def test_zero_line_page():
    img = _blank_page(400, 300)

    def mock_segment(_):
        return []

    polys = detect_lines(img, DetectionConfig(), segment_fn=mock_segment)
    assert polys == []
    ordered = assign_reading_order(polys, ReadingOrderConfig())
    assert ordered == []


def test_reading_order_top_to_bottom_left_to_right():
    polys = [
        [(10, 200), (100, 200), (100, 230), (10, 230)],
        [(10, 50), (100, 50), (100, 80), (10, 80)],
        [(150, 50), (250, 50), (250, 80), (150, 80)],
    ]
    ordered = assign_reading_order(polys, ReadingOrderConfig())
    assert [r.line_id for r in ordered] == [0, 1, 2]
    assert ordered[0].bounds[1] < ordered[2].bounds[1]
    assert ordered[0].bounds[0] < ordered[1].bounds[0]
