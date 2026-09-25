# 07 — Evaluation Plan (from SPEC §13)

## Splits
Reuse Stage 2 splits. Calibration uses **val**; every reported number uses **test** only. Never tune on test.

## Arms
A Ground truth · B Raw TrOCR · C Fine-tuned TrOCR · D C + Stage 3.

## Metrics (`eval/metrics.py`)
CER, WER, exact match, correction precision, false-correction rate, domain-term recovery, correction recall, merge/split recovery, phrase recovery, latency (mean, P95).

Metric definitions to implement precisely (align pred vs GT at token level):
- *Correction precision* = accepted corrections whose replacement equals GT token(s) / accepted corrections.
- *False-correction rate* = tokens correct before Stage 3 and changed by Stage 3 / tokens correct before Stage 3.
- *Correction recall* = wrong tokens fixed / wrong tokens that had GT in the candidate pool **and** overall (report both; the first isolates candidate-generation coverage).
- Also report **oracle candidate coverage**: % of wrong tokens whose GT appears in the pool (upper bound for the LLM).

## Ablation (`run_ablation.py`)
A FT-TrOCR · B +SymSpell · C +Domain dict · D +Confusion · E Full+Qwen strict · E1 Qwen free-form (comparison only, isolated script, never imported by production code) · E2 = E.
For B–D, replace Qwen by "top-ranked candidate if score ≥ threshold" so that the non-LLM arms are well-defined (D-005).

## Reporting
`eval/report_builder.py` emits Markdown + LaTeX tables and a per-condition breakdown (clean / mild / severe; technical vs general). Report negative results honestly.
