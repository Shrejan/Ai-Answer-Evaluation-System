"""
stage3/candidates/dictionary_source.py — Candidate source C: Domain technical vocabulary fuzzy matcher.
Matches suspicious tokens against curated ECE/VTU technical vocabulary using Levenshtein distance
and phonetic similarity.
"""

from __future__ import annotations

from typing import List, Set
from rapidfuzz.distance import Levenshtein
import jellyfish
from stage3.schemas import Span
from stage3.resources import load_domain_terms


class DictionarySource:
    name: str = "domain_dictionary"

    def __init__(self, max_edit_distance: int = 2):
        self.max_edit_distance = max_edit_distance
        self._domain_terms: Set[str] = set()
        self._initialized = False

    def _ensure_initialized(self):
        if not self._initialized:
            self._domain_terms = load_domain_terms()
            self._initialized = True

    def propose(self, span: Span) -> List[str]:
        self._ensure_initialized()
        query = (span.clean_text or span.original_text or "").strip().lower()
        if not query:
            return []

        # If length is very short, restrict edit distance to 1
        max_dist = 1 if len(query) <= 4 else self.max_edit_distance

        matched: List[tuple[str, int]] = []
        query_soundex = jellyfish.soundex(query) if len(query) >= 3 else ""

        for term in self._domain_terms:
            # Length filter optimization: skip terms with length delta > max_dist
            if abs(len(term) - len(query)) > max_dist:
                continue

            dist = Levenshtein.distance(query, term)
            if dist <= max_dist and dist > 0:
                matched.append((term, dist))
            elif dist == max_dist + 1 and query_soundex and jellyfish.soundex(term) == query_soundex:
                # Phonetic match bonus
                matched.append((term, dist))

        # Sort by distance
        matched.sort(key=lambda x: x[1])
        return [term for term, _ in matched]
