"""
tests/test_detector.py — Tests for SuspicionDetector:
7 suspicion rules, phrase span grouping, context window extraction,
worked examples detection, and zero-span clean sentence test.
"""

from stage3.schemas import Word
from stage3.detector import SuspicionDetector


def test_clean_sentence_yields_zero_spans():
    detector = SuspicionDetector(conf_threshold=0.85)
    sentence = "General health medical camps provide basic check-ups for all age groups."
    words = [
        Word(id=i, text=tok, confidence=0.98)
        for i, tok in enumerate(sentence.split())
    ]
    spans = detector.detect_spans(words)
    assert len(spans) == 0, "A fully clean high-confidence sentence MUST yield zero spans"


def test_detector_flags_worked_examples():
    detector = SuspicionDetector(conf_threshold=0.85)

    worked_words = [
        ("denital", 0.72),       # low conf + near domain term
        ("pregnue", 0.90),       # OOV + confusion pattern
        ("volontrar-y", 0.92),   # hyphen split fragment + OOV
        ("nubation", 0.95),      # near domain term (incubation) / OOV
        ("transisttor", 0.90),   # 3 t's + near domain term
        ("beat", 0.95),          # rule 7 incongruous context word
        ("attack", 0.95),        # rule 7 incongruous context word
    ]

    for text, conf in worked_words:
        w = Word(id=0, text=text, confidence=conf)
        reasons, severity = detector.check_token(w)
        assert len(reasons) > 0, f"Worked example '{text}' should trigger suspicion rules"
        assert severity > 0.0


def test_detector_contiguous_phrase_grouping():
    detector = SuspicionDetector(conf_threshold=0.85)
    # "Nutritional support for pregnant and local lacking mothers"
    # "local" and "lacking" both have low confidence or form a suspicious phrase
    words = [
        Word(id=0, text="Nutritional", confidence=0.98),
        Word(id=1, text="support", confidence=0.97),
        Word(id=2, text="for", confidence=0.99),
        Word(id=3, text="pregnant", confidence=0.96),
        Word(id=4, text="and", confidence=0.99),
        Word(id=5, text="local", confidence=0.60),   # low conf
        Word(id=6, text="lacking", confidence=0.62), # low conf
        Word(id=7, text="mothers.", confidence=0.96),
    ]

    spans = detector.detect_spans(words)
    assert len(spans) == 1
    span = spans[0]
    assert span.is_phrase is True
    assert span.span_ids == [5, 6]
    assert span.original_text == "local lacking"
    assert "pregnant and" in span.prev_context
    assert "mothers." in span.next_context


def test_detector_punctuation_handling():
    detector = SuspicionDetector(conf_threshold=0.85)
    words = [
        Word(id=0, text="clinics,", confidence=0.98),
        Word(id=1, text="denital.", confidence=0.65),
    ]
    spans = detector.detect_spans(words)
    assert len(spans) == 1
    assert spans[0].clean_text == "denital"
    assert spans[0].trailing_punct == "."
