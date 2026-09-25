"""
stage3/candidates/aggregator.py — Merges, ranks, and caps multi-source candidate proposals.
Guarantees that the original OCR text is ALWAYS present in the candidate pool.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional
from rapidfuzz.distance import Levenshtein
from stage3.schemas import Span, CandidatePool
from stage3.resources import load_general_dictionary, load_domain_terms
from stage3.config import get_config


def score_candidate(
    candidate: str,
    original: str,
    sources_count: int,
    total_sources: int,
    gen_dict: Dict[str, int],
    domain_terms: set[str],
    weights: Dict[str, float],
) -> float:
    """
    Computes a composite ranking score for a candidate:
    Score = w_dist * s_dist + w_freq * s_freq + w_agree * s_agree
    """
    cand_lower = candidate.lower()
    orig_lower = original.lower()

    # 1. Edit distance score (1.0 for distance 0, decaying as distance grows)
    dist = Levenshtein.distance(cand_lower, orig_lower)
    max_len = max(len(cand_lower), len(orig_lower), 1)
    s_dist = max(0.0, 1.0 - (dist / max_len))

    # 2. Frequency score (log-normalized frequency)
    freq = gen_dict.get(cand_lower, 0)
    if cand_lower in domain_terms:
        freq = max(freq, 2_000_000)  # Strong domain boost
    
    # Scale log(freq) into [0, 1] assuming max freq ~ 10^10
    s_freq = min(1.0, math.log10(freq + 1) / 10.0) if freq > 0 else 0.0

    # 3. Source agreement score [0, 1]
    s_agree = (sources_count / max(total_sources, 1))

    w_dist = weights.get("edit_distance", 0.4)
    w_freq = weights.get("frequency", 0.3)
    w_agree = weights.get("source_agreement", 0.3)

    return (w_dist * s_dist) + (w_freq * s_freq) + (w_agree * s_agree)


def build_candidate_pool(
    span: Span,
    sources_output: Dict[str, List[str]],
    max_pool_size: Optional[int] = None,
) -> CandidatePool:
    """
    Aggregates candidates from all sources, ranks them deterministically,
    and returns a CandidatePool of length at most max_pool_size.
    INVARIANT: The original text is ALWAYS present in the pool.
    """
    cfg = get_config()
    cap = max_pool_size or cfg.symspell.max_pool_size
    weights = cfg.symspell.ranking_weights

    gen_dict = load_general_dictionary()
    domain_terms = load_domain_terms()

    original = span.original_text.strip()
    clean_orig = (span.clean_text or original).strip()

    # Build map: candidate -> set of source names
    candidate_sources: Dict[str, List[str]] = {}

    # Register original
    candidate_sources[clean_orig] = ["original"]

    total_sources = len(sources_output)
    for src_name, cands in sources_output.items():
        for cand in cands:
            cand_norm = cand.strip()
            if not cand_norm:
                continue
            if cand_norm not in candidate_sources:
                candidate_sources[cand_norm] = []
            if src_name not in candidate_sources[cand_norm]:
                candidate_sources[cand_norm].append(src_name)

    # Compute scores for all unique candidates
    scored: List[tuple[str, float]] = []
    for cand, src_list in candidate_sources.items():
        s = score_candidate(
            candidate=cand,
            original=clean_orig,
            sources_count=len(src_list),
            total_sources=total_sources,
            gen_dict=gen_dict,
            domain_terms=domain_terms,
            weights=weights,
        )
        scored.append((cand, s))

    # Deterministic sort: descending by score, ascending by candidate string (tie-breaker)
    scored.sort(key=lambda x: (-x[1], x[0]))
    ranked_candidates = [cand for cand, _ in scored]

    # Apply cap while strictly ensuring clean_orig is included
    if len(ranked_candidates) > cap:
        final_pool = ranked_candidates[:cap]
        if clean_orig not in final_pool:
            # Replace the last ranked candidate with original
            final_pool[-1] = clean_orig
    else:
        final_pool = ranked_candidates

    # Final assertion of invariant
    assert clean_orig in final_pool, f"Original '{clean_orig}' must be in candidate pool"

    # Filter candidate sources map to only include pool members
    pool_sources = {c: candidate_sources.get(c, ["unknown"]) for c in final_pool}

    return CandidatePool(candidates=final_pool, sources=pool_sources)


def collect_and_build_candidate_pool(span: Span) -> CandidatePool:
    """Convenience helper: queries all 4 candidate sources and builds ranked candidate pool."""
    from stage3.candidates.symspell_source import SymSpellSource
    from stage3.candidates.confusion_source import ConfusionSource
    from stage3.candidates.dictionary_source import DictionarySource
    from stage3.candidates.phrase_retrieval_source import PhraseRetrievalSource
    from stage3.candidates.phonetic_source import PhoneticDomainSource

    sources = [
        SymSpellSource(),
        ConfusionSource(),
        DictionarySource(),
        PhraseRetrievalSource(),
        PhoneticDomainSource(),
    ]

    sources_output: Dict[str, List[str]] = {}
    for src in sources:
        try:
            sources_output[src.name] = src.propose(span)
        except Exception:
            sources_output[src.name] = []

    return build_candidate_pool(span, sources_output)

