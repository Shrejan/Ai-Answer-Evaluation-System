# AGENTS.md — Stage 3 Post-OCR Correction (Project Constitution)

You are building **Stage 3** of a handwritten answer-script evaluation system:
`Kraken → fine-tuned TrOCR (Stage 2) → Stage 3 (this repo) → Stage 4 evaluation`.

## Source of truth (read in this order before any task)
1. `docs/SPEC_ORIGINAL.md` — the author's full specification (authoritative for behaviour)
2. `docs/01_PRD.md` — scope and goals
3. `docs/02_ARCHITECTURE.md` — modules, data flow
4. `docs/03_DATA_CONTRACTS.md` — schemas
5. `docs/DECISIONS.md` — resolved ambiguities (overrides SPEC where they conflict, and says so)
6. `docs/PROGRESS.md` — current milestone

## Design law (never violate)
- The LLM is a **selector over a bounded candidate set**, never a generator.
- Every accepted correction satisfies `selected_candidate in candidate_set`.
- Any failure (parse, validation, timeout, low confidence, exception) → **keep original OCR text**.
- Applying a correction is reachable ONLY through a successful `ValidationResult`.

## Environment
- Python 3.11, Windows/Linux laptop, RTX 2050 (4 GB VRAM), model = `qwen3:1.7b` served by **Ollama** at `http://localhost:11434`.
- No cloud/API calls at runtime. Everything local.

## Working protocol
1. Work on **one milestone at a time** (`prompts/MILESTONES.md`). Never start the next one unasked.
2. Produce a short plan first; wait for approval on anything that touches `validation.py`, `gating.py`, `pipeline.py`.
3. Tests first or alongside code. A milestone is done only when its acceptance checks pass (`pytest -q`).
4. Never fabricate data, metrics, or thresholds. If data is missing, say so and use clearly labelled fixtures.
5. Update `docs/PROGRESS.md` and append to `docs/DECISIONS.md` at the end of every milestone.
6. Small, reviewable diffs. Commit message: `M<nn>: <summary>`.
7. Ask before adding any dependency not listed in `docs/04_TECH_STACK.md`.
