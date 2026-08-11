"""
Core scoring logic. This is the ONLY place the numerical score is
calculated — the LLM never touches these numbers.
"""

import numpy as np

import config
from embeddings import embed_texts, split_into_sentences, cosine_similarity_matrix


def status_for_similarity(similarity: float) -> str:
    if similarity >= config.COVERED_THRESHOLD:
        return "covered"
    if similarity >= config.PARTIAL_THRESHOLD:
        return "partial"
    return "missing"


def credit_for_status(status: str) -> float:
    """Per-concept credit used to build the concept_coverage fraction."""
    return {"covered": 1.0, "partial": 0.5, "missing": 0.0}[status]


def evaluate(question: str, reference_answer: str, student_answer: str) -> dict:
    """
    Runs the full Stage 1 (Sentence Transformer) evaluation:
      1. Split the reference answer into concept sentences.
      2. Split the student answer into sentences.
      3. For each reference concept, find its max similarity against any
         student sentence -> covered / partial / missing.
      4. Compute concept_coverage as the average per-concept credit.
      5. Compute overall_similarity between the full reference and full
         student answer.
      6. Combine both into a final 0-100 score using configurable weights.

    Returns a dict with everything needed for both the JSON output and the
    single LLM call.
    """
    reference_concepts = split_into_sentences(reference_answer)
    student_sentences = split_into_sentences(student_answer)

    if not reference_concepts:
        raise ValueError("Reference answer produced no usable concepts (is it empty?).")
    if not student_sentences:
        raise ValueError("Student answer is empty or produced no usable sentences.")

    # Batch-embed everything in as few calls as possible for speed.
    concept_embeddings = embed_texts(reference_concepts)
    student_sentence_embeddings = embed_texts(student_sentences)
    overall_embeddings = embed_texts([reference_answer, student_answer])

    # --- Concept-level coverage -------------------------------------------------
    sim_matrix = cosine_similarity_matrix(concept_embeddings, student_sentence_embeddings)
    max_similarities = sim_matrix.max(axis=1)  # best matching student sentence per concept

    concepts_result = []
    credits = []
    for concept_text, sim in zip(reference_concepts, max_similarities):
        sim = float(sim)
        status = status_for_similarity(sim)
        credits.append(credit_for_status(status))
        concepts_result.append(
            {
                "concept": concept_text,
                "similarity": round(sim, 4),
                "status": status,
            }
        )

    concept_coverage = float(np.mean(credits))

    # --- Overall answer-to-answer similarity ------------------------------------
    overall_similarity = float(
        cosine_similarity_matrix(overall_embeddings[0:1], overall_embeddings[1:2])[0][0]
    )

    # --- Final score --------------------------------------------------------------
    final_score = (
        overall_similarity * config.SIMILARITY_WEIGHT
        + concept_coverage * config.CONCEPT_WEIGHT
    ) * 100

    return {
        "score": round(final_score, 1),
        "similarity": round(overall_similarity, 4),
        "concept_coverage": round(concept_coverage, 4),
        "concepts": concepts_result,
    }
