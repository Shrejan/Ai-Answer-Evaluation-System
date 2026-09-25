# 00 — Pipeline Creation Roadmap (Antigravity Vibe-Coding Plan)

## 1. How to use this kit
1. Create an empty repo folder `stage3_post_ocr/`; copy this kit's contents into its root (keep `.agent/`, `docs/`, `prompts/`, `AGENTS.md`).
2. Open the folder in Antigravity. In *Customizations*, set the four `.agent/rules/*` files to **Always On**.
3. Install Ollama, run `ollama pull qwen3:1.7b`, confirm `ollama run qwen3:1.7b` works.
4. Use **Planning mode**, terminal policy **Request Review**, and start **one conversation per milestone**.
5. Paste the milestone's prompt from `prompts/MILESTONES.md`. Review the plan artifact, approve, then review the diff and tests.
6. Run `/checkpoint`, commit, open a fresh conversation for the next milestone.

> Folder names (`.agent/rules`, `.agent/workflows`) follow current Antigravity conventions; if your version uses different paths, keep the files and only move them.

## 2. Phases and milestones

| Phase | Milestone | Deliverable | Depends on | Est. effort |
|---|---|---|---|---|
| **A. Foundation** | M00 | Environment + Ollama smoke test | — | 0.5 h |
| | M01 | Scaffolding, schemas, config loader | M00 | 1 h |
| | M02 | Fixtures + dictionaries (seed) | M01 | 1–2 h |
| **B. Data-driven assets** | M03 | `build_confusion_matrix.py` → `char_confusion.json` | M01, Stage 2 pairs | 1–2 h |
| **C. Candidate engine** | M04 | Sources A–D with unit tests | M02, M03 | 3–4 h |
| | M05 | `aggregator.py` | M04 | 1 h |
| | M06 | `detector.py` | M02, M03 | 2 h |
| **D. Safety core** | M07 | Ollama client + prompts + verifier (mock first) | M01 | 2 h |
| | M08 | `validation.py` + adversarial tests **(GATE)** | M07 | 2 h |
| | M09 | `gating.py` (placeholder τ) | M08 | 0.5 h |
| **E. Integration** | M10 | `pipeline.py` end-to-end on one page | M05, M06, M09 | 2 h |
| | M11 | `correction_log.py` | M10 | 1 h |
| **F. Science** | M12 | Threshold calibration | M11, real data | 2 h |
| | M13 | Metrics + 4-way report | M12 | 2 h |
| | M14 | 5-way ablation + E1/E2 contrast | M13 | 2 h |
| | M15 | Latency benchmark (<2 s/page) | M10 | 1 h |
| | M16 | Safety audit + Definition of Done | all | 1 h |

## 3. Hard gates
- **Gate 1 (after M08):** all adversarial tests pass with the LLM mocked. Real Ollama may not be wired into `pipeline.py` before this.
- **Gate 2 (after M10):** one real page runs end-to-end, zero invariant violations.
- **Gate 3 (after M12):** thresholds come from a logged calibration run, never hand-picked.
- **Gate 4 (M16):** every box in SPEC §16 is ticked with evidence.

## 4. Inputs YOU must supply (agent cannot invent these)
| Asset | Used by | Notes |
|---|---|---|
| Stage 2 `(ground_truth, prediction)` pairs, train + val + test splits | M03, M12–M14 | CSV/JSONL: `id, gt_text, pred_text, split` |
| Stage 2 page JSONs with word confidences | M10+ | Format in `03_DATA_CONTRACTS.md` |
| ECE/VTU term list | M02, M04 | Start with ~200 terms; expand |
| Domain phrase corpus | M04-D | From answer dataset + textbooks |
| General English wordlist + frequencies | M02 | e.g. a public frequency list |

## 5. Vibe-coding tactics for a 1.7B local model project
- Keep prompts to the agent narrow: one module + its tests per request.
- Ask for the **plan first**; reject plans that touch files outside the milestone.
- After each milestone, run `/verify-invariants`.
- If the agent wants to "improve" the prompt template or relax validation, decline; log it in `DECISIONS.md` instead.
