"""Health/diagnostics helpers (doc 09)."""

from __future__ import annotations

from typing import Optional

import torch

from ocr_pipeline.api.schemas import GpuHealthInfo, HealthResponse, ModelsHealthInfo
from ocr_pipeline.config import AppConfig


def gpu_health() -> GpuHealthInfo:
    if not torch.cuda.is_available():
        return GpuHealthInfo(available=False)
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
    reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
    return GpuHealthInfo(
        available=True,
        device=f"cuda:{device}",
        allocated_mb=round(allocated, 2),
        reserved_mb=round(reserved, 2),
    )


def build_health_response(
    config: AppConfig,
    *,
    warmed_up: bool,
    queue_depth: int,
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        gpu=gpu_health(),
        models=ModelsHealthInfo(
            detector=config.detection.detector_name,
            recognizer=config.recognition.model_id,
            warmed_up=warmed_up,
        ),
        queue_depth=queue_depth,
    )
