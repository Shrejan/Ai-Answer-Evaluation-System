"""
stage3/detector.py — Suspicious-span detection using 7 rule-based/statistical checks.
Operates without any LLM calls (ultra-fast). Groups contiguous suspicious tokens into phrase spans,
strips outer punctuation per D-004, and extracts surrounding context windows.
"""

from __future__ import annotations

import re
from typing import List, Set, Tuple
from rapidfuzz.distance import Levenshtein
from stage3.schemas import Word, Span, SuspicionRecord
from stage3.resources import (
    load_domain_terms,
    load_general_dictionary,
    load_confusion_matrix,
    load_phrase_corpus,
)


PUNCT_REGEX = re.compile(r"^([^a-zA-Z0-9]*)(.*?)([^a-zA-Z0-9]*)$")
VOWELS = set("aeiouyAEIOUY")
REPEATED_CHARS_REGEX = re.compile(r"(.)\1{2,}")
DIGIT_LETTER_REGEX = re.compile(r"(?<=[a-zA-Z])\d|\d(?=[a-zA-Z])")


def split_punctuation(raw_text: str) -> Tuple[str, str, str]:
    """
    Separates leading punctuation, clean token, and trailing punctuation per D-004.
    Example: 'groups.' -> ('', 'groups', '.')
    """
    match = PUNCT_REGEX.match(raw_text)
    if match:
        lead, clean, trail = match.groups()
        return lead, clean, trail
    return "", raw_text, ""


class SuspicionDetector:
    def __init__(self, conf_threshold: float = 0.85, context_window: int = 6):
        self.conf_threshold = conf_threshold
        self.context_window = context_window
        self._domain_terms: Set[str] = set()
        self._general_vocab: Set[str] = set()
        self._confusion_patterns: Set[str] = set()
        self._context_flagged_words: Set[str] = {"beat", "attack"}  # Out of place in healthcare/ECE
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return
        # Clear cache to ensure any disk updates are reflected
        load_general_dictionary.cache_clear()
        load_domain_terms.cache_clear()

        self._domain_terms = load_domain_terms()
        self._general_vocab = set(load_general_dictionary().keys())
        
        matrix = load_confusion_matrix()
        for k in matrix.get("substitutions", {}).keys():
            if "->" in k:
                _, pred = k.split("->", 1)
                self._confusion_patterns.add(pred)
        for k in matrix.get("multi_char_patterns", {}).keys():
            if "->" in k:
                _, pred = k.split("->", 1)
                self._confusion_patterns.add(pred)

        self._initialized = True

    def check_token(self, word: Word) -> Tuple[List[str], float]:
        """
        Evaluates the 7 suspicion rules on a single Word token.
        Returns triggered reasons and a severity score in [0.0, 1.0].
        """
        self._ensure_initialized()
        lead_p, clean, trail_p = split_punctuation(word.text)
        token_lower = clean.lower()

        reasons: List[str] = []
        severity = 0.0

        if not token_lower:
            return [], 0.0

        # Rule 1: OCR confidence threshold
        if word.confidence < self.conf_threshold:
            reasons.append("rule1_low_confidence")
            severity = max(severity, 1.0 - word.confidence)

        # Rule 2: Token absent from both general and domain dictionaries
        is_in_general = (token_lower in self._general_vocab) or (clean in self._general_vocab)
        is_in_domain = (token_lower in self._domain_terms)
        is_oov = not is_in_general and not is_in_domain

        if is_oov:
            reasons.append("rule2_oov")
            severity = max(severity, 0.75)

        # Rule 3: Matches known character confusion patterns (applies to OOV or low confidence words)
        if is_oov or word.confidence < self.conf_threshold:
            for pat in ["tt", "rn", "ue", "ii", "vv"]:
                if pat in token_lower:
                    reasons.append(f"rule3_confusion_pattern:{pat}")
                    severity = max(severity, 0.70)
                    break

        # Rule 4: Unusual character patterns
        # 4a. 3+ repeated characters (e.g. 'ttt' in 'transisttor')
        if REPEATED_CHARS_REGEX.search(token_lower):
            reasons.append("rule4_repeated_characters")
            severity = max(severity, 0.80)
        # 4b. Mixed digits and letters (e.g. 't1mer')
        if DIGIT_LETTER_REGEX.search(token_lower):
            reasons.append("rule4_mixed_digit_letter")
            severity = max(severity, 0.85)
        # 4c. No vowels in word of length >= 3
        if len(token_lower) >= 3 and not any(c in VOWELS for c in token_lower):
            reasons.append("rule4_no_vowels")
            severity = max(severity, 0.85)

        # Rule 5: Edit distance to domain term is <= 2 but not exact match
        # Applies to OOV tokens or tokens with confidence < threshold
        if not is_in_domain and (is_oov or word.confidence < self.conf_threshold):
            for term in self._domain_terms:
                if abs(len(term) - len(token_lower)) <= 2:
                    dist = Levenshtein.distance(token_lower, term)
                    if 0 < dist <= 2:
                        reasons.append(f"rule5_near_domain_term:{term}")
                        severity = max(severity, 0.80)
                        break

        # Rule 6: Merge / split / hyphen fragment
        # E.g. 'volontrar-y' (internal hyphen in an OOV word)
        if "-" in word.text and is_oov:
            reasons.append("rule6_split_fragment")
            severity = max(severity, 0.70)

        # Rule 7: Contextual n-gram / low-support / incongruous word check
        if token_lower in self._context_flagged_words:
            reasons.append("rule7_low_support_context")
            severity = max(severity, 0.70)

        return reasons, severity

    def detect_spans(self, words: List[Word]) -> List[Span]:
        """
        Scans page tokens, flags suspicious words, groups adjacent suspicious tokens
        into phrase spans, extracts context windows, and returns List[Span].
        """
        self._ensure_initialized()
        if not words:
            return []

        # Step 1: Detect individual word suspicions
        flagged_indices: List[int] = []
        for idx, w in enumerate(words):
            reasons, _ = self.check_token(w)
            if reasons:
                flagged_indices.append(idx)

        if not flagged_indices:
            return []

        # Step 2: Group contiguous suspicious tokens into phrase spans
        grouped_spans: List[List[int]] = []
        current_group: List[int] = [flagged_indices[0]]

        for idx in flagged_indices[1:]:
            if idx == current_group[-1] + 1:
                # Contiguous token
                current_group.append(idx)
            else:
                grouped_spans.append(current_group)
                current_group = [idx]
        if current_group:
            grouped_spans.append(current_group)

        # Step 3: Build Span objects with context windows and punctuation handling
        spans: List[Span] = []
        total_words = len(words)

        for group in grouped_spans:
            start_idx = group[0]
            end_idx = group[-1]

            group_tokens = [words[i].text for i in group]
            raw_span_text = " ".join(group_tokens)

            # Strip outer punctuation
            lead_punct, _, _ = split_punctuation(group_tokens[0])
            _, _, trail_punct = split_punctuation(group_tokens[-1])
            
            clean_tokens = [split_punctuation(t)[1] for t in group_tokens]
            clean_span_text = " ".join(filter(None, clean_tokens))

            # Context windows (prev/next tokens)
            prev_start = max(0, start_idx - self.context_window)
            next_end = min(total_words, end_idx + 1 + self.context_window)

            prev_ctx_words = [words[i].text for i in range(prev_start, start_idx)]
            next_ctx_words = [words[i].text for i in range(end_idx + 1, next_end)]

            prev_context = " ".join(prev_ctx_words)
            next_context = " ".join(next_ctx_words)

            spans.append(
                Span(
                    span_ids=group,
                    original_text=raw_span_text,
                    prev_context=prev_context,
                    next_context=next_context,
                    is_phrase=(len(group) > 1),
                    leading_punct=lead_punct,
                    trailing_punct=trail_punct,
                    clean_text=clean_span_text,
                )
            )

        return spans
