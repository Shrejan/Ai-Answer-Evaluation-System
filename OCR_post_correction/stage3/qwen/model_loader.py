"""
stage3/qwen/model_loader.py — Warms up local Ollama Qwen3-1.7B model and ensures residency.
"""

from __future__ import annotations

import logging
from stage3.qwen.client import OllamaClient
from stage3.schemas import Span, CandidatePool
from stage3.qwen.prompt_templates import build_prompt_bundle


logger = logging.getLogger(__name__)


def warmup_model(client: OllamaClient) -> bool:
    """Sends a quick warm-up request to Ollama to preload weights into VRAM."""
    span = Span(span_ids=[0], original_text="warmup", clean_text="warmup")
    pool = CandidatePool(candidates=["warmup"], sources={"warmup": ["original"]})
    prompt = build_prompt_bundle(span, pool)
    try:
        raw = client.select(prompt, timeout_s=15.0)
        logger.info(f"Ollama Qwen3 warm-up successful. Raw response: {raw}")
        return True
    except Exception as e:
        logger.warning(f"Ollama warm-up encountered exception: {e}")
        return False
