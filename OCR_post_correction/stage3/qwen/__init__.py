"""
stage3.qwen — LLM verifier client, prompt builder, and schema-constrained calling.
"""
from typing import Protocol
from stage3.schemas import PromptBundle


class LLMClient(Protocol):
    """Protocol for LLM verifiers."""
    def select(self, prompt: PromptBundle, timeout_s: float) -> str:
        """Execute constrained verification prompt and return raw JSON string."""
        ...
