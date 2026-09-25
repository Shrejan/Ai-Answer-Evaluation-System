"""Conservative SymSpell line correction (doc 07)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Set, Tuple

from symspellpy import SymSpell, Verbosity

from ocr_pipeline.config import SpellConfig
from ocr_pipeline.models import WordCorrectionInfo

WORD_PATTERN = re.compile(r"(\w+(?:'\w+)?|[^\w\s]|\s+)")


def _build_symspell(config: SpellConfig) -> SymSpell:
    sym = SymSpell(max_dictionary_edit_distance=config.max_edit_distance, prefix_length=7)
    if config.dictionary_path and config.dictionary_path.exists():
        sym.load_dictionary(str(config.dictionary_path), term_index=0, count_index=1)
    else:
        # Built-in frequency dictionary shipped with symspellpy.
        from importlib.resources import files

        dict_path = files("symspellpy").joinpath("frequency_dictionary_en_82_765.txt")
        sym.load_dictionary(str(dict_path), term_index=0, count_index=1)
    return sym


def _preserve_case(original: str, corrected: str) -> str:
    if original.isupper():
        return corrected.upper()
    if original[0].isupper() and len(original) > 1:
        return corrected.capitalize()
    if original[0].isupper():
        return corrected.capitalize()
    return corrected


def _split_word_parts(token: str) -> Tuple[str, str, str]:
    leading = ""
    core = token
    trailing = ""
    m = re.match(r"^([^\w]*)([\w']+?)([^\w]*)$", token)
    if m:
        leading, core, trailing = m.group(1), m.group(2), m.group(3)
    return leading, core, trailing


def spell_correct_line(
    raw_text: str,
    protected_vocab: Set[str],
    config: SpellConfig,
    sym: Optional[SymSpell] = None,
) -> Tuple[str, List[WordCorrectionInfo]]:
    if not config.enabled or not raw_text.strip():
        return raw_text, []

    if sym is None:
        sym = _build_symspell(config)

    tokens = WORD_PATTERN.findall(raw_text)
    corrected_parts: List[str] = []
    corrections: List[WordCorrectionInfo] = []

    for token in tokens:
        if not re.match(r"^[\w']+$", token):
            corrected_parts.append(token)
            continue

        leading, core, trailing = _split_word_parts(token)
        if not core or len(core) < config.min_word_length:
            corrected_parts.append(token)
            continue

        core_lower = core.lower()
        if core_lower in protected_vocab or core in protected_vocab:
            corrected_parts.append(token)
            corrections.append(
                WordCorrectionInfo(raw_word=core, corrected_word=core, changed=False)
            )
            continue

        suggestions = sym.lookup(
            core_lower,
            Verbosity.CLOSEST,
            max_edit_distance=config.max_edit_distance,
        )
        if not suggestions:
            corrected_parts.append(token)
            corrections.append(
                WordCorrectionInfo(raw_word=core, corrected_word=core, changed=False)
            )
            continue

        best = suggestions[0]
        if best.distance > config.max_edit_distance or best.term == core_lower:
            corrected_parts.append(token)
            corrections.append(
                WordCorrectionInfo(
                    raw_word=core,
                    corrected_word=core,
                    changed=False,
                    edit_distance=best.distance,
                )
            )
            continue

        fixed_core = _preserve_case(core, best.term)
        corrected_parts.append(f"{leading}{fixed_core}{trailing}")
        corrections.append(
            WordCorrectionInfo(
                raw_word=core,
                corrected_word=fixed_core,
                changed=True,
                edit_distance=best.distance,
            )
        )

    return "".join(corrected_parts), corrections


class SpellCorrector:
    def __init__(self, config: SpellConfig, protected_vocab: Optional[Set[str]] = None):
        self.config = config
        self.protected_vocab = protected_vocab or set()
        self._sym: Optional[SymSpell] = None

    @property
    def sym(self) -> SymSpell:
        if self._sym is None:
            self._sym = _build_symspell(self.config)
        return self._sym

    def correct(self, raw_text: str) -> Tuple[str, List[WordCorrectionInfo]]:
        return spell_correct_line(
            raw_text,
            self.protected_vocab,
            self.config,
            sym=self.sym,
        )
