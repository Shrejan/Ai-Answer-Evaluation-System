"""
tests/test_config.py — Tests for cached configuration loader and schema defaults.
"""

from pathlib import Path
from stage3.config import get_config, reload_config, Stage3Config


def test_config_loader():
    cfg = get_config()
    assert isinstance(cfg, Stage3Config)
    assert cfg.model.backend == "ollama"
    assert cfg.model.model == "qwen3:1.7b"
    assert cfg.model.think is False
    assert cfg.thresholds.tau_general == 0.90
    assert cfg.thresholds.tau_domain == 0.85
    assert cfg.symspell.max_edit_distance == 2
    assert cfg.symspell.max_pool_size == 6


def test_config_cache():
    cfg1 = get_config()
    cfg2 = get_config()
    assert cfg1 is cfg2  # Same instance in cache

    cfg3 = reload_config()
    assert cfg3 is not None
