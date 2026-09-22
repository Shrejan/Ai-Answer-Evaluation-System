"""Parallel crop dispatch with order-safe reassembly by line_id (doc 05)."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

import numpy as np

from ocr_pipeline.config import CropConfig
from ocr_pipeline.cropping.crop_worker import CropJob, CropResult, process_crop_job
from ocr_pipeline.models import LineRecord, RecognitionStatus


def _worker_process(job_data: tuple) -> CropResult:
    page_array, job_dict, config = job_data
    job = CropJob(line_id=job_dict["line_id"], polygon=job_dict["polygon"])
    return process_crop_job(page_array, job, config)


def crop_lines_parallel(
    original_image: np.ndarray,
    ordered_lines: List[LineRecord],
    config: CropConfig,
) -> List[LineRecord]:
    if not ordered_lines:
        return ordered_lines

    jobs = [
        {
            "line_id": line.line_id,
            "polygon": line.polygon,
        }
        for line in ordered_lines
    ]

    results_by_id: dict[int, CropResult] = {}

    if config.process_pool_workers <= 1:
        for job_dict in jobs:
            job = CropJob(line_id=job_dict["line_id"], polygon=job_dict["polygon"])
            result = process_crop_job(original_image, job, config)
            results_by_id[result.line_id] = result
    else:
        payload = [(original_image, j, config) for j in jobs]
        with ProcessPoolExecutor(max_workers=config.process_pool_workers) as pool:
            futures = {pool.submit(_worker_process, p): p[1]["line_id"] for p in payload}
            for future in as_completed(futures):
                result = future.result()
                results_by_id[result.line_id] = result

    for line in ordered_lines:
        result = results_by_id.get(line.line_id)
        if result is None:
            continue
        line.crop = result.crop
        if result.status == RecognitionStatus.DISCARDED_SMALL:
            line.recognition_status = RecognitionStatus.DISCARDED_SMALL
        elif result.status == RecognitionStatus.DISCARDED_FAILED:
            line.recognition_status = RecognitionStatus.DISCARDED_FAILED
        elif result.crop is None:
            line.recognition_status = RecognitionStatus.DISCARDED_SMALL
        else:
            line.recognition_status = RecognitionStatus.PENDING
    return ordered_lines
