# Spell Correction (SymSpell, Conservative)

## Placement
After TrOCR recognition, before final word-record construction. Toggleable per request and via a global config default.

## Flow

1. Keep the raw recognized line text untouched (`raw_text`).
2. Split each line into candidate words while retaining punctuation boundaries (don't lose sentence punctuation in the split/rejoin).
3. If correction is enabled AND the dictionary is available, run SymSpell per candidate word.
4. Preserve capitalization and punctuation as far as possible when reconstructing the corrected word (e.g. don't lowercase a proper-noun-cased word just because the dictionary match is lowercase).
5. Store `corrected_text` on the line separately from `raw_text`.
6. `corrected_text` (when correction is enabled) feeds the final `text` and `words` output; `raw_text` remains available for diagnostics.

## Conservatism requirements

Handwritten answer scripts contain names, abbreviations, scientific/domain-specific terms that a general English dictionary won't know. To avoid mangling legitimate content:

- Support a **protected vocabulary** list (config-loadable, e.g. a text file of domain terms / common names) that SymSpell must never touch.
- Support a **correction distance/confidence threshold** — only apply a correction when SymSpell's edit distance and frequency ranking clear a configured bar; otherwise leave the word as recognized.
- Make the dictionary itself swappable/extendable per subject (a physics answer script and a history answer script have different domain vocabularies) — this can be a config path, doesn't need a UI.

## Confidence interaction — important

**Correction confidence must never be confused with OCR confidence.** If SymSpell changes a word:
- The word keeps its original OCR confidence (computed from TrOCR token probabilities — see `08_CONFIDENCE_SCORING.md`).
- Optionally attach an internal `correction_applied: true` / edit-distance value for diagnostics, but do not let a dictionary match inflate the reported confidence — a corrected word is not "more visually certain," it's differently spelled.

## Function signature sketch

```python
def spell_correct_line(
    raw_text: str,
    protected_vocab: set[str],
    config: SpellConfig,
) -> tuple[str, list[WordCorrectionInfo]]:
    """Returns corrected_text and per-word correction metadata (raw word, corrected word,
    whether changed, edit distance)."""
```

## Testing checklist

- A protected-vocabulary word is never altered even if it superficially resembles a common misspelling.
- Punctuation attached to a word (e.g. trailing comma/period) survives correction unchanged.
- Correction can be disabled per request and the response then reflects raw TrOCR text with `corrected: false` on every word.
- A domain term not in the general dictionary and not in protected vocab, but which fails the correction-confidence threshold, is left unchanged rather than replaced with a low-confidence "closest" dictionary word.
