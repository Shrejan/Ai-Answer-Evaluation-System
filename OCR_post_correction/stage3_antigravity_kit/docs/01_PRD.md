# 01 — Product Requirements (Stage 3)

## Goal
Reduce residual OCR errors (baseline WER 0.6143, CER 0.2632, EM 0.0470) in fine-tuned TrOCR output on handwritten ECE/VTU answers, **without introducing hallucinated corrections**.

## Scope
**In:** suspicious-span detection, multi-source bounded candidate generation, Qwen3-1.7B selection, hard validation, confidence gating, audit logging, evaluation, ablation, latency benchmark.
**Out:** Stage 1/2/4 changes, fine-tuning Qwen, cloud LLMs, UI.

## Functional requirements
| ID | Requirement |
|---|---|
| FR1 | `correct_page(ocr_json) -> Stage3Result` per SPEC §1 |
| FR2 | Only flagged spans reach the LLM |
| FR3 | Candidate pool ≤ 6, original always included |
| FR4 | LLM returns `{"selected_candidate","confidence"}` only |
| FR5 | All LLM outputs pass `validate_qwen_output` |
| FR6 | Thresholds `tau_general`, `tau_domain` calibrated from data |
| FR7 | Full JSONL audit log incl. rejected corrections |

## Non-functional requirements
| ID | Requirement |
|---|---|
| NFR1 | Stage 3 latency < 2 s/page (mean, report P95) on RTX 2050 |
| NFR2 | Fully offline, deterministic (temp 0, seed 42) |
| NFR3 | Zero invariant violations over the full test corpus |
| NFR4 | Reproducible: config snapshot stored with every log line |

## Success criteria
See SPEC §13 metrics and §16 Definition of Done. Results are reported honestly even if Stage 3 does not improve CER/WER.
