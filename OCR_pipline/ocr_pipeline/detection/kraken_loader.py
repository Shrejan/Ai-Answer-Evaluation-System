"""Load Kraken blla model from disk (avoids Python 3.9 importlib.resources bug)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from kraken.lib.vgsl import TorchVGSLModel

logger = logging.getLogger(__name__)


def resolve_kraken_model_path(model_path: Path) -> Path:
    """Resolve blla.mlmodel from cwd or repo root."""
    if model_path.is_absolute() and model_path.exists():
        return model_path

    candidates = [
        Path.cwd() / model_path,
        Path(__file__).resolve().parents[2] / model_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Kraken blla model not found: {model_path}. "
        "Place blla.mlmodel in the project root or set OCR_KRAKEN_MODEL_PATH."
    )


def load_kraken_model(model_path: Path) -> Any:
    """Load blla segmentation model once at startup."""
    resolved = resolve_kraken_model_path(model_path)
    logger.info("Loading Kraken blla from %s", resolved)
    model = TorchVGSLModel.load_model(str(resolved))
    model.eval()
    if torch.cuda.is_available():
        model.nn.to(torch.device("cuda"))
        logger.info("Kraken blla loaded on cuda")
    else:
        logger.info("Kraken blla loaded on cpu")
    return model
