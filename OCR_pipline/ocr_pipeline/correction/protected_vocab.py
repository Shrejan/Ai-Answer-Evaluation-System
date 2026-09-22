"""Protected vocabulary loader (doc 07)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Set


def load_protected_vocab(path: Optional[Path]) -> Set[str]:
    if path is None or not path.exists():
        return set()
    words: Set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        word = line.strip()
        if word and not word.startswith("#"):
            words.add(word)
            words.add(word.lower())
    return words
