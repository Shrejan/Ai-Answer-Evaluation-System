"""
stage3/resources.py — Cached loaders for domain dictionaries, word frequencies,
phrase corpora, character confusion matrices, and the SymSpell engine.
Zero network calls; all loaded locally once and cached.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from symspellpy import SymSpell, Verbosity


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@functools.lru_cache(maxsize=1)
def load_domain_terms(path: Optional[Path] = None) -> Set[str]:
    """Load curated ECE/VTU domain terms as a lowercase set."""
    file_path = path or (DATA_DIR / "domain_dictionary" / "ece_terms.txt")
    if not file_path.exists():
        return set()

    terms = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                terms.add(line.lower())
    return terms


@functools.lru_cache(maxsize=1)
def load_general_dictionary(path: Optional[Path] = None) -> Dict[str, int]:
    """Load general English word frequency dictionary (word -> frequency)."""
    file_path = path or (DATA_DIR / "domain_dictionary" / "general_dictionary.txt")
    if not file_path.exists():
        return {}

    freqs: Dict[str, int] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                word = parts[0].strip().lower()
                try:
                    count = int(parts[1].strip())
                    freqs[word] = count
                except ValueError:
                    continue
            elif len(parts) == 1:
                freqs[parts[0].strip().lower()] = 1000
    return freqs


@functools.lru_cache(maxsize=1)
def load_phrase_corpus(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load domain phrases from JSONL corpus."""
    file_path = path or (DATA_DIR / "phrase_corpus" / "domain_phrases.jsonl")
    if not file_path.exists():
        return []

    phrases = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                phrases.append(json.loads(line))
    return phrases


@functools.lru_cache(maxsize=1)
def load_confusion_matrix(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load character confusion matrix JSON."""
    file_path = path or (DATA_DIR / "confusion_matrix" / "char_confusion.json")
    if not file_path.exists():
        return {
            "substitutions": {
                "i->e": 49, "i->a": 29, "e->o": 26, "r->n": 23,
                "o->a": 20, "a->o": 19, "c->o": 18, "i->o": 14
            },
            "multi_char_patterns": {"tt->t": 12, "rn->m": 15},
            "insertions": {},
            "deletions": {},
            "metadata": {"note": "seed fallback"}
        }

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def get_symspell_engine(
    max_edit_distance: int = 2,
    prefix_length: int = 7,
) -> SymSpell:
    """Initialize and populate SymSpell with general and domain vocabulary."""
    sym_spell = SymSpell(
        max_dictionary_edit_distance=max_edit_distance,
        prefix_length=prefix_length,
    )

    # 1. Load general dictionary frequencies
    gen_dict = load_general_dictionary()
    for word, count in gen_dict.items():
        sym_spell.create_dictionary_entry(word, count)

    # 2. Load domain technical vocabulary with high frequency boost
    domain_terms = load_domain_terms()
    for term in domain_terms:
        # Give technical terms strong frequency so they rank competitive against typos
        sym_spell.create_dictionary_entry(term, 2_000_000)

    return sym_spell
