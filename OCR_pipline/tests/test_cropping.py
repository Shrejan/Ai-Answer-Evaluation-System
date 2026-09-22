"""Phase 2 cropping tests."""

import time
from unittest.mock import patch

import numpy as np

from ocr_pipeline.config import CropConfig
from ocr_pipeline.cropping.crop_worker import crop_single_line, process_crop_job, CropJob
from ocr_pipeline.cropping.pool_manager import crop_lines_parallel
from ocr_pipeline.models import LineRecord, RecognitionStatus, bounds_from_polygon


def _page(w: int, h: int) -> np.ndarray:
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    img[50:80, 50:200] = 0
    return img


def test_edge_of_page_polygon_clips_padding():
    poly = [(0, 0), (30, 0), (30, 20), (0, 20)]
    img = _page(100, 100)
    crop, status = crop_single_line(img, poly, CropConfig(padding_px=10))
    assert status == RecognitionStatus.PENDING
    assert crop is not None
    assert crop.width <= 100
    assert crop.height <= 100


def test_below_minimum_size_discarded():
    poly = [(10, 10), (12, 10), (12, 12), (10, 12)]
    img = _page(100, 100)
    crop, status = crop_single_line(img, poly, CropConfig(padding_px=0, min_crop_size_px=8))
    assert crop is None
    assert status == RecognitionStatus.DISCARDED_SMALL


def test_out_of_order_future_reassembly():
    img = _page(300, 200)
    lines = []
    for i, y in enumerate([20, 80, 140]):
        poly = [(20, y), (200, y), (200, y + 30), (20, y + 30)]
        bounds, vc, h = bounds_from_polygon(poly)
        lines.append(
            LineRecord(
                line_id=i,
                polygon=poly,
                bounds=bounds,
                vertical_center=vc,
                estimated_height=h,
            )
        )

    delays = {0: 0.05, 1: 0.0, 2: 0.02}

    def slow_process(page_array, job, config):
        time.sleep(delays.get(job.line_id, 0))
        return process_crop_job(page_array, job, config)

    with patch("ocr_pipeline.cropping.pool_manager.process_crop_job", side_effect=slow_process):
        result = crop_lines_parallel(
            img,
            lines,
            CropConfig(process_pool_workers=1),
        )

    assert [r.line_id for r in result] == [0, 1, 2]
    assert all(r.crop is not None for r in result)
