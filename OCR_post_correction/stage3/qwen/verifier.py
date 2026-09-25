"""
stage3/qwen/verifier.py — Orchestrates prompt formatting and verifier invocation.
Returns RAW model output only. Wraps timeouts and runtime exceptions into typed VerifierError.
DO NOT parse or validate output here; that is strictly the responsibility of validation.py.
"""

from __future__ import annotations

import logging
from stage3.schemas import Span, CandidatePool
from stage3.qwen.client import LLMClient
from stage3.qwen.prompt_templates import build_prompt_bundle


logger = logging.getLogger(__name__)


class VerifierError(Exception):
    """Raised when verifier invocation encounters a network error, timeout, or client exception."""
    def __init__(self, message: str, error_type: str = "llm_exception"):
        super().__init__(message)
        self.error_type = error_type


def verify_span(
    client: LLMClient,
    span: Span,
    pool: CandidatePool,
    timeout_s: float = 10.0,
) -> str:
    """
    Submits a suspicious span and candidate pool to the LLM client.
    Returns the untrusted, unparsed raw string output from the model.
    """
    prompt = build_prompt_bundle(span, pool)

    try:
        raw_output = client.select(prompt, timeout_s=timeout_s)
        return raw_output
    except TimeoutError as e:
        logger.warning(f"Verifier timed out on span '{span.original_text}': {e}")
        raise VerifierError(str(e), error_type="llm_timeout") from e
    except Exception as e:
        logger.warning(f"Verifier error on span '{span.original_text}': {e}")
        raise VerifierError(str(e), error_type="llm_exception") from e
