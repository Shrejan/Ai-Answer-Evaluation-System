"""
stage3/candidates/confusion_source.py — Candidate source B: Empirical character confusion variants.
Applies observed OCR substitution and multi-character error patterns in reverse,
filtering generated forms against known vocabulary so only valid words are proposed.
"""

from __future__ import annotations

from typing import List, Set
from stage3.schemas import Span
from stage3.resources import load_confusion_matrix, load_domain_terms, load_general_dictionary


class ConfusionSource:
    name: str = "confusion_matrix"

    def __init__(self, top_k_patterns: int = 15):
        self.top_k_patterns = top_k_patterns
        self._vocab: Set[str] = set()
        self._reverse_substitutions: List[tuple[str, str]] = []
        self._reverse_multichar: List[tuple[str, str]] = []
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return
        
        # Valid vocabulary: general words + domain terms
        self._vocab = set(load_general_dictionary().keys()) | load_domain_terms()

        # Build reverse substitutions from confusion matrix:
        # If ground truth 'gt' was misrecognized as 'pred' (gt->pred),
        # then when we see 'pred' in OCR output, we should test replacing it with 'gt'.
        matrix = load_confusion_matrix()
        subs = matrix.get("substitutions", {})
        for pair_str, count in sorted(subs.items(), key=lambda x: x[1], reverse=True)[:self.top_k_patterns]:
            if "->" in pair_str:
                gt_char, pred_char = pair_str.split("->", 1)
                if gt_char and pred_char:
                    self._reverse_substitutions.append((pred_char, gt_char))

        multi = matrix.get("multi_char_patterns", {})
        for pair_str, count in sorted(multi.items(), key=lambda x: x[1], reverse=True)[:self.top_k_patterns]:
            if "->" in pair_str:
                gt_pat, pred_pat = pair_str.split("->", 1)
                if gt_pat and pred_pat:
                    self._reverse_multichar.append((pred_pat, gt_pat))

        # Also add common OCR character collapses
        common_reversals = [
            ("rn", "m"), ("cl", "d"), ("vv", "w"), ("ii", "u"),
            ("1", "l"), ("0", "o"), ("5", "s"), ("8", "b"),
            ("ue", "ant"), # worked example: pregnue -> pregnant
        ]
        for pred, gt in common_reversals:
            if (pred, gt) not in self._reverse_multichar:
                self._reverse_multichar.append((pred, gt))

        self._initialized = True

    def propose(self, span: Span) -> List[str]:
        self._ensure_initialized()
        query = (span.clean_text or span.original_text or "").strip().lower()
        if not query:
            return []

        candidates: List[str] = []

        # 1. Test multi-character reversals
        for pred_pat, gt_pat in self._reverse_multichar:
            if pred_pat in query:
                replaced = query.replace(pred_pat, gt_pat)
                if replaced in self._vocab and replaced != query and replaced not in candidates:
                    candidates.append(replaced)

        # 2. Test single-character substitutions
        for pred_char, gt_char in self._reverse_substitutions:
            if pred_char in query:
                # Try replacing each occurrence
                for idx, c in enumerate(query):
                    if c == pred_char:
                        variant = query[:idx] + gt_char + query[idx + 1:]
                        if variant in self._vocab and variant != query and variant not in candidates:
                            candidates.append(variant)

        return candidates
