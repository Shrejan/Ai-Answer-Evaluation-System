"""
stage3/qwen/client.py — LLM client implementations adhering to the LLMClient protocol.
Contains OllamaClient for local inference and MockLLMClient for testing.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional
import httpx
from stage3.schemas import PromptBundle
from stage3.config import get_config


class LLMClient:
    """Protocol / base class for verifier LLM clients."""
    def select(self, prompt: PromptBundle, timeout_s: float = 10.0) -> str:
        """Sends verification prompt to model and returns RAW string output only."""
        raise NotImplementedError


class OllamaClient(LLMClient):
    """Client for local Ollama server running Qwen3-1.7B."""
    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        think: Optional[bool] = None,
    ):
        cfg = get_config()
        self.host = host or cfg.model.host
        self.model = model or cfg.model.model
        self.think = think if think is not None else cfg.model.think
        self.url = f"{self.host.rstrip('/')}/api/chat"

    def select(self, prompt: PromptBundle, timeout_s: float = 10.0) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "think": self.think,
            "keep_alive": "30m",
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            "format": prompt.schema_dict,
            "options": {
                "temperature": 0.0,
                "seed": 42,
                "num_predict": 40,
                "num_ctx": 2048,
            },
        }

        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(self.url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")


class MockLLMClient(LLMClient):
    """Mock LLM client for deterministic unit tests and adversarial testing."""
    def __init__(
        self,
        fixed_response: Optional[str] = None,
        raise_timeout: bool = False,
        raise_exception: Optional[Exception] = None,
    ):
        self.fixed_response = fixed_response
        self.raise_timeout = raise_timeout
        self.raise_exception = raise_exception
        self.call_count = 0
        self.last_prompt: Optional[PromptBundle] = None

    def select(self, prompt: PromptBundle, timeout_s: float = 10.0) -> str:
        self.call_count += 1
        self.last_prompt = prompt

        if self.raise_timeout:
            raise TimeoutError("Mock LLM client timed out")
        if self.raise_exception:
            raise self.raise_exception

        if self.fixed_response is not None:
            return self.fixed_response

        # Default smart mock: pick second candidate (first replacement) if available, with 0.95 confidence
        enum_list = prompt.schema_dict.get("properties", {}).get("selected_candidate", {}).get("enum", [])
        if len(enum_list) > 1:
            chosen = enum_list[1]
        elif enum_list:
            chosen = enum_list[0]
        else:
            chosen = "mock_word"

        return json.dumps({"selected_candidate": chosen, "confidence": 0.95})
