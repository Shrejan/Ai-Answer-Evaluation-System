# OCR Pipeline — Spec Pack for Cursor

This folder is a complete, modular spec for building the handwritten answer-script OCR pipeline (Kraken detection + TrOCR recognition, no image-enhancement preprocessing, 4 GB VRAM safe).

## How to use this with Cursor

1. Create your project repo and copy this whole `docs/` folder (and `.cursorrules` at the repo root) into it before writing any code.
2. Open Cursor in that repo. Start a new agent/chat and say something like:
   > "Read every file in docs/ in numeric order, then follow docs/11_IMPLEMENTATION_TASKS.md phase by phase, writing tests alongside each phase."
3. Let it work phase by phase — don't let it jump straight to a monolithic implementation. The `.cursorrules` file will keep reminding it of the hard constraints (no preprocessing, stable ordering, single GPU worker, etc.) throughout the session.
4. After each phase, review the diff yourself before telling it to continue — this is a thesis-grade module, worth reading, not just accepting.

## File map

| File | Purpose |
|---|---|
| `00_PROJECT_BRIEF.md` | Goals, non-goals, constraints, tech stack, definition of done |
| `01_ARCHITECTURE.md` | Staged pipeline, module layout, hard architectural rules |
| `02_REQUEST_RESPONSE_CONTRACT.md` | API request/response schemas, success + error shapes |
| `03_DETECTION_KRAKEN.md` | Kraken detection with adaptive downscaling |
| `04_READING_ORDER.md` | Row-grouping and reading-order algorithm |
| `05_CROPPING_AND_PARALLELISM.md` | No-preprocessing cropping + parallel crop execution |
| `06_RECOGNITION_TROCR.md` | TrOCR batching, GPU safety, OOM backoff |
| `07_SPELL_CORRECTION.md` | Conservative SymSpell correction |
| `08_CONFIDENCE_SCORING.md` | Token-to-word confidence math |
| `09_PERFORMANCE_AND_RELIABILITY.md` | Startup, concurrency, observability checklist |
| `10_VALIDATION_PLAN.md` | Per-stage metrics and test-fixture coverage |
| `11_IMPLEMENTATION_TASKS.md` | The actual ordered build checklist |
| `.cursorrules` | Enforcement layer — keep at repo root, not inside `docs/` |

Place `.cursorrules` at the repository root (Cursor reads it from there); everything else can live under `docs/`.
