"""
stage3/candidates/phrase_retrieval_source.py — Candidate source D: Phrase-level domain retrieval.
Retrieves plausible replacements using TF-IDF over the domain phrase corpus,
and performs token merge/split candidate generation for segmentation errors.
"""

from __future__ import annotations

import re
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz.distance import Levenshtein
import jellyfish
from stage3.schemas import Span
from stage3.resources import load_phrase_corpus, load_domain_terms, load_general_dictionary


class PhraseRetrievalSource:
    name: str = "phrase_retrieval"

    def __init__(self, top_k: int = 3, sim_threshold: float = 0.20):
        self.top_k = top_k
        self.sim_threshold = sim_threshold
        self._corpus: List[str] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self._vocab = None
        self._corpus_words = set()
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return

        corpus_data = load_phrase_corpus()
        self._corpus = [item.get("text", "") for item in corpus_data if item.get("text")]
        
        if self._corpus:
            self._vectorizer = TfidfVectorizer(ngram_range=(1, 3), lowercase=True)
            self._tfidf_matrix = self._vectorizer.fit_transform(self._corpus)
            for doc in self._corpus:
                for w in re.findall(r"\b\w+\b", doc.lower()):
                    self._corpus_words.add(w)

        self._vocab = set(load_general_dictionary().keys()) | load_domain_terms() | self._corpus_words
        self._initialized = True

    def propose(self, span: Span) -> List[str]:
        self._ensure_initialized()
        raw_text = span.original_text.strip()
        clean_text = (span.clean_text or raw_text).lower()

        if not clean_text:
            return []

        candidates: List[str] = []

        # 1. Split handling (hyphenation / split boundary, e.g. "volontrar-y" -> "voluntary")
        if "-" in raw_text:
            dehyphen = raw_text.replace("-", "").strip().lower()
            if dehyphen in self._vocab and dehyphen not in candidates:
                candidates.append(dehyphen)
            for term in self._vocab:
                if abs(len(term) - len(dehyphen)) <= 2 and Levenshtein.distance(term, dehyphen) <= 2:
                    if term not in candidates:
                        candidates.append(term)

        # 2. Merge handling (adjacent words, e.g. "local lacking" -> "lactating")
        if " " in raw_text or span.is_phrase:
            merged = re.sub(r"\s+", "", raw_text).lower()
            if merged in self._vocab and merged not in candidates:
                candidates.append(merged)
            
            # Match merged token against vocabulary (distance <= 4 or phonetic or corpus word similarity)
            merged_soundex = jellyfish.soundex(merged) if len(merged) >= 4 else ""
            for term in self._vocab:
                if abs(len(term) - len(merged)) <= 4:
                    dist = Levenshtein.distance(term, merged)
                    if dist <= 4:
                        if term not in candidates:
                            candidates.append(term)
                    elif merged_soundex and jellyfish.soundex(term) == merged_soundex:
                        if term not in candidates:
                            candidates.append(term)

            # Check similarity against domain phrase corpus vocabulary
            from rapidfuzz import fuzz
            for cw in self._corpus_words:
                if fuzz.ratio(merged, cw) >= 55.0 and cw not in candidates:
                    candidates.append(cw)

        # 3. Truncated prefix / suffix / OCR prefix corruption recovery (e.g. "nubation" -> "incubation")
        if len(clean_text) >= 5:
            for term in self._vocab:
                if abs(len(term) - len(clean_text)) <= 3:
                    # Check if term is an edit distance <= 2 from clean_text (e.g. nubation -> incubation dist 2)
                    if Levenshtein.distance(term, clean_text) <= 2:
                        if term not in candidates:
                            candidates.append(term)
                # Also check suffix match
                if term.endswith(clean_text) and term not in candidates:
                    candidates.append(term)

        # 4. Contextual TF-IDF phrase retrieval
        if self._vectorizer is not None and self._tfidf_matrix is not None:
            query = f"{span.prev_context} {raw_text} {span.next_context}".strip().lower()
            if query:
                q_vec = self._vectorizer.transform([query])
                sims = cosine_similarity(q_vec, self._tfidf_matrix).flatten()
                
                top_indices = sims.argsort()[::-1][:self.top_k]
                for idx in top_indices:
                    if sims[idx] >= self.sim_threshold:
                        matched_doc = self._corpus[idx]
                        tokens_raw = raw_text.lower().split()
                        if len(tokens_raw) >= 2:
                            words_in_doc = matched_doc.split()
                            doc_len = len(words_in_doc)
                            n = len(tokens_raw)
                            for i in range(doc_len - n + 1):
                                sub = " ".join(words_in_doc[i:i + n])
                                if tokens_raw[1:] == words_in_doc[i + 1:i + n]:
                                    candidate_phrase = sub.strip(".,;:!?").lower()
                                    if candidate_phrase not in candidates:
                                        candidates.append(candidate_phrase)

        return candidates
