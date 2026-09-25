"""
stage3/candidates/symspell_source.py — Candidate source A: SymSpell fuzzy spell checking.
Uses symmetric delete spelling correction with general English and domain term frequencies.
"""

from __future__ import annotations

from typing import List
from symspellpy import Verbosity
from stage3.schemas import Span
from stage3.resources import get_symspell_engine


class SymSpellSource:
    name: str = "symspell"

    def __init__(self, max_edit_distance: int = 2, prefix_length: int = 7):
        self.max_edit_distance = max_edit_distance
        self.prefix_length = prefix_length
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            self._engine = get_symspell_engine(
                max_edit_distance=self.max_edit_distance,
                prefix_length=self.prefix_length,
            )
        return self._engine

    def propose(self, span: Span) -> List[str]:
        query = (span.clean_text or span.original_text or "").strip().lower()
        if not query:
            return []

        suggestions = self.engine.lookup(
            query,
            Verbosity.ALL,
            max_edit_distance=self.max_edit_distance,
            transfer_casing=False,
        )

        results: List[str] = []
        for s in suggestions:
            term = s.term.lower()
            if term not in results:
                results.append(term)
        return results
