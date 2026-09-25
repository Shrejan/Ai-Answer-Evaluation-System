"""
stage3/config.py — Strongly typed configuration loader with caching.
Loads model.yaml, thresholds.yaml, and symspell.yaml from the config/ directory.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel, Field


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


class ModelConfig(BaseModel):
    backend: str = "ollama"
    host: str = "http://localhost:11434"
    model: str = "qwen3:1.7b"
    think: bool = False
    temperature: float = 0.0
    seed: int = 42
    num_predict: int = 40
    num_ctx: int = 2048
    keep_alive: str = "30m"
    timeout_s: float = 10.0


class ThresholdsConfig(BaseModel):
    version: str = "thresholds_v2.yaml"
    tau_general: float = 0.90
    tau_domain: float = 0.85
    conf_threshold: float = 0.85
    margin_threshold: float = 0.0
    use_margin: bool = False
    max_false_correction_rate: float = 0.01


class SymSpellConfig(BaseModel):
    max_edit_distance: int = 2
    prefix_length: int = 7
    max_candidates_per_source: int = 6
    context_window_tokens: int = 6
    max_pool_size: int = 6
    ranking_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "frequency": 0.3,
            "edit_distance": 0.4,
            "source_agreement": 0.3,
        }
    )


class Stage3Config(BaseModel):
    model: ModelConfig
    thresholds: ThresholdsConfig
    symspell: SymSpellConfig


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@functools.lru_cache(maxsize=1)
def get_config(config_dir: Optional[Path] = None) -> Stage3Config:
    """Load and cache the complete configuration tree."""
    base = Path(config_dir) if config_dir else CONFIG_DIR
    model_data = _load_yaml(base / "model.yaml")
    thresholds_data = _load_yaml(base / "thresholds.yaml")
    symspell_data = _load_yaml(base / "symspell.yaml")

    return Stage3Config(
        model=ModelConfig(**model_data),
        thresholds=ThresholdsConfig(**thresholds_data),
        symspell=SymSpellConfig(**symspell_data),
    )


def reload_config(config_dir: Optional[Path] = None) -> Stage3Config:
    """Clear config cache and reload from disk."""
    get_config.cache_clear()
    return get_config(config_dir)
