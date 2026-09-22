# Confidence Scoring

## Why this is nontrivial
TrOCR is a sequence-generation model, not a detector with a built-in per-token confidence score exposed by default. Confidence has to be computed from the model's own generation output.

## Approach

1. Call `.generate()` with `output_scores=True, return_dict_in_generate=True` (and beam search params from `06_RECOGNITION_TROCR.md`).
2. From the returned scores, compute normalized token probabilities for the generated sequence.
3. Exclude special tokens (BOS/EOS/PAD) from the confidence calculation.
4. Align token-level scores to decoded words:
   - A practical first version: use the tokenizer's decoded token sequence and aggregate the token probabilities that fall within each word's character span.
   - Word confidence = **geometric mean** (or length-normalized mean — pick one and be consistent) of that word's constituent token probabilities.
   - Clip the final value to `[0, 1]`.
5. Attach this as `confidence` on the word record.

## Correction vs. OCR confidence
As stated in `07_SPELL_CORRECTION.md`: if spell correction changes a word, `confidence` still reflects OCR/visual evidence, not dictionary certainty. Don't let a "clean" dictionary match make a low-confidence OCR guess look falsely confident.

## Calibration disclaimer (document this in code comments and any output docs)
These confidence values are **model-derived estimates, not calibrated probabilities**, until validated against a labeled dataset. Calibration (e.g. temperature scaling or isotonic regression against a held-out labeled set of handwritten pages) is a follow-up validation step (see `10_VALIDATION_PLAN.md`), not part of the initial implementation.

## Function signature sketch

```python
def compute_word_confidences(
    generated_sequence: torch.Tensor,
    token_scores: list[torch.Tensor],
    tokenizer,
    word_boundaries: list[tuple[int, int]],  # char spans per word
) -> list[float]:
    """Returns one confidence value per word, in the same order as word_boundaries."""
```

## Testing checklist

- A short, unambiguous crop (mock generation scores near 1.0 for all tokens) yields a word confidence close to 1.0.
- A crop with one clearly uncertain token pulls that word's confidence down without affecting neighboring words' scores.
- Confidence values are always within `[0, 1]` even with adversarial/mocked score inputs (e.g. verify the clipping actually triggers on out-of-range mock scores).
