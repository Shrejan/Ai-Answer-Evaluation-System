"""Phase 5 spell correction tests."""

from pathlib import Path

from ocr_pipeline.config import SpellConfig
from ocr_pipeline.correction.protected_vocab import load_protected_vocab
from ocr_pipeline.correction.spell_correct import SpellCorrector, spell_correct_line


def test_protected_word_untouched(tmp_path):
    vocab_file = tmp_path / "protected.txt"
    vocab_file.write_text("mitochondria\nATP\n", encoding="utf-8")
    protected = load_protected_vocab(vocab_file)
    config = SpellConfig(enabled=True, max_edit_distance=2)
    corrected, info = spell_correct_line("The mitochondria is here.", protected, config)
    assert "mitochondria" in corrected
    assert not any(i.changed for i in info if i.raw_word == "mitochondria")


def test_punctuation_preserved():
    config = SpellConfig(enabled=True, max_edit_distance=2)
    corrected, _ = spell_correct_line("Hello, world.", set(), config)
    assert corrected.endswith(".")
    assert "," in corrected


def test_correction_disabled_returns_raw():
    config = SpellConfig(enabled=False)
    raw = "Teh mitochondria"
    corrected, info = spell_correct_line(raw, set(), config)
    assert corrected == raw
    assert info == []


def test_below_threshold_word_unchanged(monkeypatch):
    config = SpellConfig(enabled=True, max_edit_distance=1, min_word_length=4)

    class FakeSuggestion:
        def __init__(self, term, distance):
            self.term = term
            self.distance = distance

    class FakeSym:
        def lookup(self, word, verbosity, max_edit_distance):
            return [FakeSuggestion("completelydifferent", 2)]

    corrected, info = spell_correct_line(
        "zzzzterm",
        set(),
        config,
        sym=FakeSym(),
    )
    assert "zzzzterm" in corrected
    assert not info[0].changed
