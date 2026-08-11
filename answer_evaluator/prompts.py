"""
Single prompt template used for the one-and-only LLM call.
"""

import json


SYSTEM_PROMPT = """You are an educational answer evaluator.

Evaluate the student's descriptive answer against the question and reference answer.

The numerical score has already been calculated by a separate semantic similarity
system. Do NOT change the numerical score. Your task is only to explain the result.

Identify:
1. What concepts the student explained correctly.
2. What important concepts are missing.
3. Any conceptual or technical errors.
4. What the student should have explained.
5. A short improvement suggestion.

Important rules:
- Evaluate concepts, not exact wording.
- Do not penalize grammar heavily.
- Do not require the student to use the same wording as the reference.
- Do not call something wrong merely because it is absent from the reference.
- Correct additional information should not be penalized.
- Distinguish between missing information and incorrect information.
- Do not invent mistakes.
- Keep the report concise.
- Base the evaluation primarily on the supplied question, reference answer,
  student answer, and concept similarity results.

Return valid JSON only, with exactly this shape and no extra keys:
{
  "correct_concepts": ["..."],
  "missing_concepts": ["..."],
  "errors": [
    {"statement": "...", "explanation": "...", "severity": "low|medium|high"}
  ],
  "improvement": "..."
}

Do not include markdown code fences. Return only the JSON object itself."""


def build_user_prompt(
    question: str,
    reference_answer: str,
    student_answer: str,
    concepts_result: list[dict],
    final_score: float,
) -> str:
    """
    Packs everything the LLM needs for its single call: the question, both
    answers, the concept-level similarity results, and the already-computed
    final score (which it must not alter).
    """
    concept_lines = "\n".join(
        f"- {c['concept']} | similarity={c['similarity']} | status={c['status']}"
        for c in concepts_result
    )

    return f"""Question:
{question}

Reference answer:
{reference_answer}

Student answer:
{student_answer}

Concept similarity results (already computed, do not recompute):
{concept_lines}

Final numerical score (already computed, do not change): {final_score}/100

Now produce the JSON evaluation report described in the system prompt."""
