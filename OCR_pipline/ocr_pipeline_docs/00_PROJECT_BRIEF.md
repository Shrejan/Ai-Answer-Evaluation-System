# Project Brief — Handwritten Answer-Script OCR Pipeline

## Read order for Cursor
This is file 1 of 11. Read all files in this folder in numeric order before writing any code. Together they form the full spec. Do not start implementing after only this file.

1. `00_PROJECT_BRIEF.md` (this file)
2. `01_ARCHITECTURE.md`
3. `02_REQUEST_RESPONSE_CONTRACT.md`
4. `03_DETECTION_KRAKEN.md`
5. `04_READING_ORDER.md`
6. `05_CROPPING_AND_PARALLELISM.md`
7. `06_RECOGNITION_TROCR.md`
8. `07_SPELL_CORRECTION.md`
9. `08_CONFIDENCE_SCORING.md`
10. `09_PERFORMANCE_AND_RELIABILITY.md`
11. `10_VALIDATION_PLAN.md`
12. `11_IMPLEMENTATION_TASKS.md` — the actual build checklist, in order

## What this project is

A backend OCR service that extracts handwritten text from scanned/photographed answer-script pages. It is one module inside a larger academic pipeline ("AI-Driven Descriptive Answer Script Evaluation System") whose later stages do NLP post-correction and semantic scoring against a rubric — this repo only owns OCR. Keep the module boundary clean: input is a page image, output is structured text + word-level data, nothing about scoring or evaluation lives here.

## Core objective

Fast, reliable handwritten-line OCR that:
- Uses **Kraken** for line detection.
- Uses **TrOCR** (`microsoft/trocr-base-handwritten`) for line recognition.
- Downscales images **only for detection**, never for recognition.
- Performs **no image enhancement preprocessing** (no denoise, CLAHE, deskew, threshold/binarize, upscaling, sharpening, morphology) anywhere in the pipeline.
- Returns both full page text and word-level records with confidence.
- Runs safely on a **4 GB VRAM GPU (RTX 2050)** via controlled batching and a single GPU worker.

Guiding principle: **reduce resolution only where page-level line detection can tolerate it; preserve full original resolution wherever character recognition happens.**

## Explicit non-goals (do not build these here)

- No NLP correction beyond basic spell-check (SymSpell) — semantic correction is a downstream module.
- No answer scoring, grading, or rubric matching.
- No image enhancement/preprocessing pipeline of any kind.
- No UI. This is a backend service exposing a clear function/API boundary.
- No training or fine-tuning of Kraken or TrOCR in this phase — inference only, with model paths/config left swappable for later fine-tuned checkpoints.

## Target environment / constraints

- GPU: 4 GB VRAM class card (e.g. RTX 2050). All batch sizes, worker counts, and queueing must default to values safe for this card, with config overrides for bigger hardware.
- Single GPU, single model-serving worker per GPU — never spin up multiple unbounded concurrent TrOCR calls against the same GPU.
- Python backend (FastAPI assumed unless the existing codebase says otherwise — confirm before adding a web framework).
- Must degrade gracefully: OOM should trigger batch-size backoff and retry, never a hard crash.

## Tech stack (confirm/adjust against existing repo before assuming)

- Detection: `kraken` (blla segmentation).
- Recognition: HuggingFace `transformers` — `TrOCRProcessor` + `VisionEncoderDecoderModel` (`microsoft/trocr-base-handwritten`).
- Spell correction: SymSpell (`symspellpy` or equivalent), optional/toggleable.
- Parallel cropping: `concurrent.futures.ProcessPoolExecutor`.
- Web layer: FastAPI (adjust if the project already uses Flask/Django).
- Image handling: Pillow / NumPy, no OpenCV enhancement calls.

## Definition of done for this module

- A page image in → deterministic JSON out (schema in `02_REQUEST_RESPONSE_CONTRACT.md`), for both text-bearing pages and blank pages.
- No enhancement preprocessing anywhere in the code path — this should be checkable by grep (no calls to denoise/CLAHE/threshold/deskew/upscale functions).
- Runs end-to-end on a 4 GB GPU without OOM under the configured default batch size.
- Timing, model version, and confidence are present in every successful response.
- Validation metrics in `10_VALIDATION_PLAN.md` can be computed against a small labeled test set.
