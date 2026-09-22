# Implementation Progress

Checklist mirrors `ocr_pipeline_docs/11_IMPLEMENTATION_TASKS.md`. Read this file first after a context reset.

## Phase 0 — Scaffolding
- [x] Create the repo layout from `01_ARCHITECTURE.md`.
- [x] Create `config.py` with every tunable listed across files `03`–`09`, with sane defaults for a 4 GB GPU.
- [x] Add `.gitignore`, `requirements.txt` / `pyproject.toml`.
- [x] Stub the pydantic schemas from `02_REQUEST_RESPONSE_CONTRACT.md` in `api/schemas.py`.

**Notes:** Config via `pydantic-settings` with `OCR_` env prefix. Python 3.9 compatible type hints.

## Phase 1 — Detection
- [x] Implement `detection/downscale.py` (measure + downscale + scale factor).
- [x] Implement `detection/kraken_detector.py` (run blla, restore coordinates, clip to bounds).
- [x] Implement `detection/reading_order.py` per `04_READING_ORDER.md`.
- [x] Write `tests/test_detection.py` covering: normal page, oversized page (downscale path), zero-line page, degenerate-polygon filtering.

**Notes:** Added `models.py` for shared `LineRecord`/`Polygon` types. Kraken calls injectable via `segment_fn` for tests.

## Phase 2 — Cropping
- [x] Implement `cropping/crop_worker.py` (single-crop logic: bbox + padding + mask-to-white + convert).
- [x] Implement `cropping/pool_manager.py` (bounded `ProcessPoolExecutor`, dispatch + order-safe reassembly by `line_id`).
- [x] Write `tests/test_cropping.py` covering: edge-of-page polygon, below-minimum-size discard, out-of-order future completion reassembled correctly.
- [x] **Checkpoint**: grep — no forbidden preprocessing calls in `ocr_pipeline/`.

## Phase 3 — Recognition
- [x] Implement `recognition/trocr_engine.py`: model load, warm-up, FP16-on-CUDA, beam search, inference_mode, batching.
- [x] Implement OOM backoff/retry logic.
- [x] Implement `recognition/gpu_queue.py`: single-worker bounded queue serializing GPU access.
- [x] Write `tests/test_recognition.py` covering: batch success, mocked OOM triggers backoff, line_id association survives a failed crop mid-batch.

## Phase 4 — Confidence
- [x] Implement `scoring/confidence.py` per `08_CONFIDENCE_SCORING.md` (token-to-word alignment, geometric mean, clipping).
- [x] Write `tests/test_confidence.py` covering: near-certain crop, one-uncertain-token crop, out-of-range mock scores get clipped.

**Notes:** Token alignment uses char-span overlap, not proportional mapping. Tests use logit-style score tensors (not pseudo-probabilities).

## Phase 5 — Spell Correction
- [x] Implement `correction/protected_vocab.py` (loadable vocabulary list).
- [x] Implement `correction/spell_correct.py` per `07_SPELL_CORRECTION.md`.
- [x] Write `tests/test_spell_correction.py` covering: protected word untouched, punctuation preserved, correction disabled per request, below-threshold word left unchanged.

## Phase 6 — Assembly & API
- [x] Implement `assembly/response_builder.py`: build `text` and `words` from the same ordered line/word representation.
- [x] Implement `diagnostics/timing.py` (per-stage timers feeding `timings_ms`).
- [x] Implement `diagnostics/health.py` per the `/health` shape in `09`.
- [x] Implement `api/routes.py`: OCR + health endpoints, structured 4xx/5xx error handling.
- [x] Wire `main.py`: load + warm up models at startup, mount routes, start bounded inference queue.
- [x] Added `pipeline.py` orchestrating all stages end-to-end.

**Notes:** `create_app(skip_model_load=True)` for tests. Startup skips init if `app.state.services` pre-set.

## Phase 7 — End-to-end validation
- [x] Assemble fixture set in `tests/fixtures/` (blank, clean, faint, slanted, multi_row, multi_column, punctuation, numbers, domain_terms, oversized).
- [x] Write `tests/test_end_to_end.py` covering the full contract.
- [x] Concurrent request test (4 threads) confirms bounded queue handles parallel posts.
- [x] Final grep check for forbidden preprocessing calls (Python-based, no `rg` dependency).
- [x] CER utility test on fixture labels — synthetic pairs yield CER=0.0.

**Validation numbers (synthetic labeled pairs, not live TrOCR):** CER=0.0 on mock "hello world" / "the mitochondria" pairs. Full CER/WER on real labeled handwriting requires running against GPU + Kraken with ground-truth transcripts (not run in CI — models not loaded in test suite).

## Test status
**36/36 tests passing** (`python -m pytest tests/ -q`)

## Run the service
```bash
cd C:\Users\User\code_file_folder\python\OCR_pipline
pip install -r requirements.txt
uvicorn ocr_pipeline.main:app --host 0.0.0.0 --port 8000
```
