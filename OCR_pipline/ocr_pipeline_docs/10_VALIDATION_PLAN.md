# Validation Plan

Evaluate accuracy and speed **per stage**, not just end-to-end, so a regression can be traced to the responsible component.

## Metrics by stage

1. **Detection recall** — % of ground-truth handwritten lines actually found by Kraken.
2. **Crop integrity** — % of polygons whose strokes are not clipped by the crop rectangle (visually spot-check a sample; automate a stroke-touches-border heuristic if feasible).
3. **Recognition quality** — Character Error Rate (CER) and Word Error Rate (WER) on raw TrOCR output vs. ground truth.
4. **Spell-correction impact** — change in WER after correction (should be non-negative improvement on average, and must not regress domain-term accuracy — track this separately using the protected-vocabulary test set).
5. **Confidence quality** — correlation between predicted confidence and actual correctness; this is where calibration (temperature scaling / isotonic regression, deferred from `08_CONFIDENCE_SCORING.md`) gets evaluated.
6. **End-to-end latency** — per-stage and total timing distributions (p50/p95), not just averages.
7. **GPU behavior** — peak VRAM, reserved VRAM, OOM frequency under load.
8. **Concurrency behavior** — queue latency and throughput under multiple simultaneous page requests.

## Required test-fixture coverage

Build a small labeled test set spanning:
- Clean handwriting
- Faint handwriting
- Slanted lines
- Multiple rows
- Multi-column layout
- Punctuation-heavy text
- Numbers / equations
- Domain-specific terms (tie this to the protected-vocabulary list)
- Pages larger than `DETECTION_MAX_DIMENSION` (to exercise the downscale path)
- A genuinely blank page (to exercise the zero-lines path)

## Suggested test file mapping (matches `01_ARCHITECTURE.md` layout)

- `tests/test_detection.py` → metric 1, plus the downscale/oversized-page edge case.
- `tests/test_cropping.py` → metric 2, plus discard-small-crop and parallel-reassembly-order cases.
- `tests/test_recognition.py` → metric 3, plus OOM-backoff behavior (mocked).
- `tests/test_spell_correction.py` → metric 4, plus protected-vocabulary and threshold cases.
- `tests/test_confidence.py` → metric 5 groundwork (unit-level sanity, not full calibration).
- `tests/test_end_to_end.py` → metrics 6–8, plus the full request/response contract (including the empty-page and error-response shapes from `02_REQUEST_RESPONSE_CONTRACT.md`).

## Definition of "validated enough to move to the next thesis stage"
- CER/WER on the labeled set meets whatever threshold the supervisor has set (confirm this number before treating the module as complete — it isn't specified here).
- No enhancement-preprocessing calls anywhere in the codebase (grep-verifiable).
- End-to-end runs without OOM at the default batch size on the target 4 GB GPU across the full fixture set.
