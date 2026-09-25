# DECISIONS (append-only ADR log)

| ID | Decision | Rationale | Status |
|---|---|---|---|
| D-001 | Use Ollama (`qwen3:1.7b`) instead of bitsandbytes/HF in-process | Matches your laptop setup; SPEC §7.1 allows GGUF via ollama | Accepted |
| D-002 | Candidate match is exact, case-sensitive, after Unicode NFC + `strip()`; original token case is preserved on replacement (lowercase lookup, restore capitalisation pattern) | Removes ambiguity in SPEC §8 check 3 | Proposed — confirm |
| D-003 | Confidence = model-reported value for v1; evaluate logprob margin later | Small models are miscalibrated | Open |
| D-004 | Punctuation is split from tokens before lookup and re-attached | `volontrar-y`, `camps.` cases | Proposed — confirm |
| D-005 | Non-LLM ablation arms auto-accept the top-ranked candidate above a score threshold | Needed to define arms B–D | Proposed — confirm |
| D-006 | Context window = 6 tokens each side; max pool = 6; `num_predict` = 40 | SPEC ranges | Accepted |
| D-007 | SPEC cross-reference fixes: correction schema = §10, latency = §12, safety audit = §13.5, config = §14 | SPEC internal section numbers are inconsistent | Accepted |
| D-008 | Qwen3 thinking disabled; `<think>` output rejected | Latency + safety | Accepted |
