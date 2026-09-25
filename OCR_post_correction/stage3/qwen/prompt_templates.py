"""
stage3/qwen/prompt_templates.py — Fixed system prompt and dynamic JSON-schema builder.
Obeyed strictly from docs/05_PROMPTS_QWEN.md and SPEC §7.2.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from stage3.schemas import Span, CandidatePool, PromptBundle


SYSTEM_PROMPT = """You are an OCR correction verifier.
Your task is to select the most likely original word or phrase from the supplied candidate list.

Rules:
1. You may select ONLY from the candidate list.
2. Never generate a new word.
3. Never modify any other part of the sentence.
4. Never paraphrase.
5. Never improve grammar.
6. Never add information.
7. If the original OCR text is already plausible, select the original.
8. When uncertain, select the original.
9. Return only valid JSON. No explanation, no markdown, no extra text."""

USER_TEMPLATE = """Context (before): {previous_context}
OCR text: {ocr_text}
Context (after): {next_context}

Candidate list: {candidate_list}

Return JSON format:
{{"selected_candidate": "<chosen_candidate>", "confidence": <float_0.0_to_1.0>}}"""


def build_schema(candidate_list: List[str]) -> Dict[str, Any]:
    """Dynamically builds strict JSON schema restricting output to an enum of candidates."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "selected_candidate": {
                "type": "string",
                "enum": candidate_list,
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
        "required": ["selected_candidate", "confidence"],
    }


def build_prompt_bundle(span: Span, pool: CandidatePool) -> PromptBundle:
    """Builds a complete PromptBundle for Qwen contextual verification."""
    candidates_json = json.dumps(pool.candidates)
    user_prompt = USER_TEMPLATE.format(
        previous_context=span.prev_context or "[START]",
        ocr_text=span.original_text,
        next_context=span.next_context or "[END]",
        candidate_list=candidates_json,
    )
    schema = build_schema(pool.candidates)

    return PromptBundle(
        system=SYSTEM_PROMPT,
        user=user_prompt,
        schema=schema,
    )
