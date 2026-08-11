"""
A single, small function that makes exactly ONE OpenRouter request to turn
the already-computed similarity results into a human-readable report.
"""

import json
import re

import requests

import config
from prompts import SYSTEM_PROMPT, build_user_prompt


class OpenRouterError(Exception):
    pass


def _strip_code_fences(text: str) -> str:
    """LLMs sometimes wrap JSON in ```json fences even when told not to."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def generate_report(
    question: str,
    reference_answer: str,
    student_answer: str,
    concepts_result: list[dict],
    final_score: float,
) -> dict:
    """
    Makes exactly one OpenRouter chat completion request and returns the
    parsed JSON report. Raises OpenRouterError on any failure (missing key,
    network issue, timeout, invalid JSON) so the caller can decide how to
    degrade gracefully.
    """
    if not config.OPENROUTER_API_KEY:
        raise OpenRouterError("OPENROUTER_API_KEY is not set.")

    user_prompt = build_user_prompt(
        question, reference_answer, student_answer, concepts_result, final_score
    )

    payload = {
        "model": config.OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }

    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            config.OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=config.OPENROUTER_TIMEOUT_SECONDS,
        )
    except requests.exceptions.Timeout:
        raise OpenRouterError("OpenRouter request timed out.")
    except requests.exceptions.RequestException as exc:
        raise OpenRouterError(f"OpenRouter request failed: {exc}")

    if response.status_code == 401:
        raise OpenRouterError("Invalid OpenRouter API key.")
    if response.status_code != 200:
        raise OpenRouterError(
            f"OpenRouter returned status {response.status_code}: {response.text[:300]}"
        )

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as exc:
        raise OpenRouterError(f"Unexpected OpenRouter response shape: {exc}")

    cleaned = _strip_code_fences(content)
    try:
        report = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise OpenRouterError(f"LLM did not return valid JSON: {exc}")

    return report
