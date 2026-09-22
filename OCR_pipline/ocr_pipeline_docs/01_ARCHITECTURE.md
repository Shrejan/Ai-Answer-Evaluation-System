# Architecture

## Two logical images, always

Every page exists in two forms through the pipeline. Never let these collapse into one:

- **Detection representation** — the (possibly downscaled) copy of the page handed to Kraken for line finding.
- **Recognition source** — the original decoded image, full resolution, always used for actual cropping and TrOCR. This is never replaced by the downscaled copy.

## Staged pipeline (implement as separate, independently testable functions/modules)

1. **Request & metadata validation** — validate `request_id`, `page_id`, image payload.
2. **Image decoding** — decode upload into the original-resolution recognition source.
3. **Temporary detection-image downscaling** — only if the page exceeds a configured max dimension.
4. **Kraken line segmentation** — run blla on the detection representation.
5. **Polygon coordinate restoration** — map polygons back to original-resolution coordinate space.
6. **Reading-order reconstruction** — sort lines top-to-bottom, left-to-right.
7. **Original-resolution polygon masking + cropping** — extract line images from the recognition source.
8. **Batched TrOCR recognition** — no preprocessing beyond the model's own mandatory tensor prep.
9. **Optional spell correction** — SymSpell pass, conservative, toggleable.
10. **Word segmentation + confidence construction** — build word-level records from corrected line text.
11. **Structured response assembly** — page text + word records built from the same ordered internal representation.
12. **Timing, health & diagnostics reporting** — attach timings and expose a health/diagnostics endpoint.

## Suggested module layout

```
ocr_pipeline/
  api/
    routes.py              # HTTP endpoints (stage 1, 11, 12 glue)
    schemas.py             # pydantic request/response models (see 02_REQUEST_RESPONSE_CONTRACT.md)
  detection/
    kraken_detector.py      # stages 3-6
    downscale.py
    reading_order.py
  cropping/
    crop_worker.py          # stage 7, runs in ProcessPoolExecutor workers
    pool_manager.py
  recognition/
    trocr_engine.py          # stage 8, model load, warm-up, batching, OOM backoff
    gpu_queue.py             # single-worker bounded inference queue
  correction/
    spell_correct.py        # stage 9
    protected_vocab.py
  scoring/
    confidence.py            # stage 10 confidence math
  assembly/
    response_builder.py      # stage 11
  diagnostics/
    timing.py
    health.py
  config.py                  # all tunables in one place (see below)
  main.py                    # app startup: load models, warm up, mount routes
tests/
  fixtures/                  # sample page images: clean, faint, slanted, multi-column, oversized
  test_detection.py
  test_cropping.py
  test_recognition.py
  test_spell_correction.py
  test_confidence.py
  test_end_to_end.py
```

## Config-driven tunables (put these in `config.py`, not hardcoded)

- `DETECTION_MAX_DIMENSION` — longest-side threshold above which downscaling kicks in.
- `CROP_PADDING_PX`
- `MIN_CROP_SIZE_PX`
- `TROCR_BATCH_SIZE` (default conservative for 4 GB VRAM)
- `TROCR_BEAM_WIDTH`
- `PROCESS_POOL_WORKERS`
- `SPELL_CORRECTION_ENABLED` (default true, but overridable per request)
- `ROW_GROUPING_TOLERANCE_FRACTION` (fraction of median line height)

## Hard architectural rules

- The detection representation must never be passed to TrOCR.
- No function in `cropping/` or `recognition/` may call any image-enhancement routine (denoise, CLAHE, threshold, deskew, upscale, sharpen, morphology). Enforce this with a code-review checklist item and, ideally, a lint rule / grep-based CI check.
- Page text and word list must be derived from the *same* ordered internal line/word representation — never regenerated independently from each other (this is what guarantees they can't disagree).
- Ordered line index must remain stable across cropping, recognition, and correction, even when some crops are discarded (discarded slots are recorded, not silently dropped from the index).
