# Implementation Tasks (work through in order)

Instructions for Cursor: complete phases in order. Within a phase, write the module, then its corresponding test file, before moving to the next phase. Do not skip ahead to recognition before detection + cropping are tested. After each phase, run the phase's tests before continuing.

## Phase 0 — Scaffolding
- [ ] Create the repo layout from `01_ARCHITECTURE.md`.
- [ ] Create `config.py` with every tunable listed across files `03`–`09`, with sane defaults for a 4 GB GPU.
- [ ] Add `.gitignore`, `requirements.txt` / `pyproject.toml` with: `kraken`, `transformers`, `torch`, `symspellpy` (or chosen equivalent), `fastapi`, `uvicorn`, `pillow`, `numpy`, `pytest`.
- [ ] Stub the pydantic schemas from `02_REQUEST_RESPONSE_CONTRACT.md` in `api/schemas.py`.

## Phase 1 — Detection
- [ ] Implement `detection/downscale.py` (measure + downscale + scale factor).
- [ ] Implement `detection/kraken_detector.py` (run blla, restore coordinates, clip to bounds).
- [ ] Implement `detection/reading_order.py` per `04_READING_ORDER.md`.
- [ ] Write `tests/test_detection.py` covering: normal page, oversized page (downscale path), zero-line page, degenerate-polygon filtering.

## Phase 2 — Cropping
- [ ] Implement `cropping/crop_worker.py` (single-crop logic: bbox + padding + mask-to-white + convert).
- [ ] Implement `cropping/pool_manager.py` (bounded `ProcessPoolExecutor`, dispatch + order-safe reassembly by `line_id`).
- [ ] Write `tests/test_cropping.py` covering: edge-of-page polygon, below-minimum-size discard, out-of-order future completion reassembled correctly.
- [ ] **Checkpoint**: grep the codebase for denoise/CLAHE/threshold/deskew/upscale/sharpen/morphology calls — there should be none. Do this again at the end of every subsequent phase.

## Phase 3 — Recognition
- [ ] Implement `recognition/trocr_engine.py`: model load, warm-up, FP16-on-CUDA, beam search, inference_mode, batching.
- [ ] Implement OOM backoff/retry logic.
- [ ] Implement `recognition/gpu_queue.py`: single-worker bounded queue serializing GPU access.
- [ ] Write `tests/test_recognition.py` covering: batch success, mocked OOM triggers backoff, line_id association survives a failed crop mid-batch.

## Phase 4 — Confidence
- [ ] Implement `scoring/confidence.py` per `08_CONFIDENCE_SCORING.md` (token-to-word alignment, geometric mean, clipping).
- [ ] Write `tests/test_confidence.py` covering: near-certain crop, one-uncertain-token crop, out-of-range mock scores get clipped.

## Phase 5 — Spell Correction
- [ ] Implement `correction/protected_vocab.py` (loadable vocabulary list).
- [ ] Implement `correction/spell_correct.py` per `07_SPELL_CORRECTION.md` (threshold, protected vocab, punctuation/case preservation, raw vs corrected separation).
- [ ] Write `tests/test_spell_correction.py` covering: protected word untouched, punctuation preserved, correction disabled per request, below-threshold word left unchanged.

## Phase 6 — Assembly & API
- [ ] Implement `assembly/response_builder.py`: build `text` and `words` from the same ordered line/word representation — no independent regeneration.
- [ ] Implement `diagnostics/timing.py` (per-stage timers feeding `timings_ms`).
- [ ] Implement `diagnostics/health.py` per the `/health` shape in `09_PERFORMANCE_AND_RELIABILITY.md`.
- [ ] Implement `api/routes.py`: the OCR endpoint (request validation → pipeline call → response), the health endpoint, and structured 4xx/5xx error handling per `02_REQUEST_RESPONSE_CONTRACT.md`.
- [ ] Wire `main.py`: load + warm up models at startup, mount routes, start the bounded inference queue consumer.

## Phase 7 — End-to-end validation
- [ ] Assemble the fixture set described in `10_VALIDATION_PLAN.md` (clean/faint/slanted/multi-column/punctuation/numbers/domain-terms/oversized/blank).
- [ ] Write `tests/test_end_to_end.py` covering the full contract: normal page, blank page, oversized page (exercise downscale), invalid input (4xx), forced processing failure (5xx).
- [ ] Run a manual load test with several concurrent requests against one GPU; confirm the bounded queue prevents OOM and confirm p50/p95 latency is recorded.
- [ ] Final grep check for forbidden preprocessing calls across the entire repo.
- [ ] Compute CER/WER on the labeled fixture set and record the numbers (feeds back into the thesis validation chapter, not just this repo).

## Out of scope reminders (do not let Cursor wander into these)
- No NLP semantic correction/evaluation logic — that's a separate downstream module.
- No scoring/grading logic.
- No image-enhancement preprocessing, ever, in any phase.
- No UI beyond the API/health endpoints.
