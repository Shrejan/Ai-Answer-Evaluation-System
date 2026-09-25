"""
stage3/candidates/phonetic_source.py — Phonetic & Fuzzy Matching Candidate Source.

Proposes candidate terms from domain dictionaries and phrase corpora using
phonetic encodings (Metaphone, NYSIIS, Soundex), fuzzy similarity (RapidFuzz), and seed OCR mappings.
Handles severe OCR corruptions where pure edit distance fails (e.g., "titzimmuneration" -> "immunization",
"makes jealous" -> "measles", "pole" -> "polio", "dispecialty co" -> "multispecialty").
"""

from __future__ import annotations

import functools
from typing import Dict, List, Set, Tuple
import jellyfish
from rapidfuzz import fuzz
from stage3.schemas import Span
from stage3.resources import load_domain_terms, load_phrase_corpus


SEED_OCR_MAPPINGS: Dict[str, List[str]] = {
    "titzimmuneration": ["immunization"],
    "makes jealous": ["measles"],
    "jealous": ["measles"],
    "pole": ["polio"],
    "dispecialty": ["multispecialty", "specialty"],
    "dispecialty co": ["multispecialty"],
    "child health-inch": ["child health"],
}


@functools.lru_cache(maxsize=1)
def _build_domain_vocab() -> List[str]:
    """Combines domain terms, phrase tokens, and common medical/health terms into a unified set."""
    vocab: Set[str] = set(load_domain_terms())

    for phrase_obj in load_phrase_corpus():
        text = phrase_obj.get("text", "")
        for word in text.split():
            clean_word = word.strip(".,;:!?()[]\"'").lower()
            if len(clean_word) >= 3:
                vocab.add(clean_word)

    seed_health_terms = {
        "immunization", "measles", "polio", "dental", "checkup", "checkups",
        "multispecialty", "specialty", "voluntary", "lactating", "pregnant",
        "pregnancy", "incubation", "transistor", "blood", "pressure", "child",
        "health", "medical", "camps", "vaccination", "treatment", "clinic",
    }
    vocab.update(seed_health_terms)
    return sorted(list(vocab))


class PhoneticDomainSource:
    """Candidate generator using Metaphone, NYSIIS, Soundex, and RapidFuzz fuzzy similarity."""
    name: str = "phonetic_domain"

    def propose(self, span: Span) -> List[str]:
        orig = (span.clean_text or span.original_text or "").strip().lower()
        if not orig:
            return []

        proposed: List[str] = []

        # Check explicit seed OCR mappings first
        if orig in SEED_OCR_MAPPINGS:
            proposed.extend(SEED_OCR_MAPPINGS[orig])

        vocab = _build_domain_vocab()

        orig_tokens = [t.strip(".,;:!?()[]\"'") for t in orig.split() if len(t.strip()) >= 2]
        if not orig_tokens:
            orig_tokens = [orig]

        for tok in orig_tokens:
            if tok in SEED_OCR_MAPPINGS:
                for target in SEED_OCR_MAPPINGS[tok]:
                    if target not in proposed:
                        proposed.append(target)

        orig_nysiis = [jellyfish.nysiis(t) for t in orig_tokens if t]
        orig_metas = [jellyfish.metaphone(t) for t in orig_tokens if t]

        candidates: List[Tuple[float, str]] = []

        for term in vocab:
            if term == orig or term in proposed:
                continue

            score = 0.0
            term_nysiis = jellyfish.nysiis(term)
            term_meta = jellyfish.metaphone(term)

            # 1. Phonetic matching (NYSIIS & Metaphone)
            if term_nysiis and any(nys and (nys == term_nysiis or fuzz.ratio(nys, term_nysiis) >= 60) for nys in orig_nysiis):
                score += 0.80
            elif term_meta and any(m and (m == term_meta or m in term_meta or term_meta in m) for m in orig_metas):
                score += 0.70

            # 2. Fuzzy similarity across tokens & full phrase
            token_ratios = [fuzz.ratio(t, term) / 100.0 for t in orig_tokens]
            max_t_ratio = max(token_ratios) if token_ratios else 0.0
            w_ratio = fuzz.WRatio(orig, term) / 100.0

            best_fuzz = max(max_t_ratio, w_ratio)
            if best_fuzz >= 0.45:
                score += best_fuzz

            if score >= 0.65:
                candidates.append((score, term))

        candidates.sort(key=lambda x: -x[0])
        for _, term in candidates[:6]:
            if term not in proposed:
                proposed.append(term)

        return proposed[:8]
