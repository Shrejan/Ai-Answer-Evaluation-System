# Milestone Prompts (paste one per Antigravity conversation)

**Common preamble — prepend to every prompt:**
```
Read AGENTS.md, docs/PROGRESS.md, docs/DECISIONS.md and the sections of docs/SPEC_ORIGINAL.md named below.
Work ONLY on this milestone. First output a plan artifact (files to create/edit, tests, risks) and stop for my approval.
After approval implement, run pytest -q, then run /checkpoint. Do not fabricate data or metrics.
```

---
## M00 — Environment + Ollama smoke test
**Files:** `requirements.txt`, `scripts/smoke_ollama.py`, `README.md` (setup section)
```
Create a Python 3.11 venv setup and requirements.txt from docs/04_TECH_STACK.md.
Write scripts/smoke_ollama.py that calls qwen3:1.7b via Ollama with think=false and a JSON-schema `format`
(enum of ["dental","denital"]) and prints the raw output and latency for 5 runs (cold + warm).
Report whether output contained any <think> text.
```
**Acceptance:** script runs; valid JSON every time; warm latency printed.

## M01 — Scaffolding, schemas, config
**Spec:** §2, §10, §14 · **Docs:** 03_DATA_CONTRACTS
```
Create the full directory layout from SPEC §2 (empty modules with docstrings), config/*.yaml per docs/04_TECH_STACK.md,
stage3/schemas.py (pydantic v2, all models in docs/03_DATA_CONTRACTS.md), and stage3/config.py (typed loader, cached).
Add tests for schema validation (bad confidence, empty words) and config loading.
```
**Acceptance:** `pytest` green; `from stage3.schemas import Stage3Result` works.

## M02 — Fixtures + seed dictionaries
**Docs:** 06_TESTING, 08_DATA_PREP
```
Create tests/fixtures/worked_examples.yaml from docs/06_TESTING.md, a tiny synthetic sample page JSON,
a seed ece_terms.txt (from the SPEC §5C list plus any I provide), and a loader module stage3/resources.py
that loads dictionaries once. Mark synthetic files with a header comment "SYNTHETIC FIXTURE".
```
**Acceptance:** loader tests pass; no network calls.

## M03 — Confusion matrix builder
**Spec:** §6
```
Implement scripts/build_confusion_matrix.py reading data/stage2_pairs/*.jsonl (gt_text, pred_text), aligning at
character level (Levenshtein opcodes), counting substitutions, insertions, deletions and multi-char patterns
(tt->t, rn->m, etc.), normalising to probabilities, and writing data/confusion_matrix/char_confusion.json with a
metadata block (source file hash, pair count, date). Include a smoke test using the SPEC §0 seed confusions
(i->e 49, i->a 29, e->o 26, r->n 23, o->a 20, a->o 19, c->o 18, i->o 14) as a labelled sample only.
```
**Acceptance:** JSON matches SPEC §6 schema; test passes on a 5-pair fixture with known counts.

## M04 — Candidate sources A–D
**Spec:** §5
```
Implement candidates/symspell_source.py, confusion_source.py, dictionary_source.py, phrase_retrieval_source.py
each implementing the CandidateSource protocol in docs/02_ARCHITECTURE.md.
Confusion source: apply top-probability substitutions/collapses, keep only dictionary words.
Phrase source: TF-IDF over domain_phrases.jsonl, plus explicit merge (join adjacent tokens) and split (try boundaries) candidates.
One test file section per source, using the worked examples. Each source must return [] (never raise) on empty input.
```
**Acceptance:** worked-example targets appear in the relevant source output; latency per call logged in tests.

## M05 — Aggregator
**Spec:** §5 Aggregation
```
Implement candidates/aggregator.py build_candidate_pool(span, sources_output, max_pool_size) returning ranked candidates
and a candidate→sources map. Original always present and never dropped by the cap. Ranking = combined score of frequency,
edit distance, source agreement (weights in config). Tests: cap respected, original kept when it ranks last, dedupe, deterministic order.
```
**Acceptance:** property test — original ∈ pool for random inputs.

## M06 — Detector
**Spec:** §4
```
Implement detector.py with the 7 suspicion rules (no LLM), SuspicionRecord output, and grouping of contiguous suspicious
tokens into phrase spans. Punctuation handling per D-004. Provide context windows (prev/next 6 tokens). Rule 7 may start as a
TF-IDF/n-gram low-support check against the phrase corpus. Test on: beat, attack, pregnue, local lacking, volontrar-y, denital, nubation.
Also test that a fully clean sentence yields zero spans.
```
**Acceptance:** worked examples flagged; clean sentence yields none.

## M07 — Ollama client + prompts + verifier (mock first)
**Spec:** §7 · **Docs:** 05_PROMPTS_QWEN
```
Implement qwen/client.py (OllamaClient and MockLLMClient behind the LLMClient protocol), qwen/prompt_templates.py
(exact system prompt from docs/05_PROMPTS_QWEN.md; dynamic JSON-schema enum), qwen/model_loader.py (warm-up call, keep_alive),
qwen/verifier.py (returns RAW string only; wraps timeout/exception into a typed VerifierError that downstream treats as rejection).
Do NOT parse or interpret output here.
```
**Acceptance:** tests with mock; one `@pytest.mark.ollama` integration test.

## M08 — Validation + adversarial tests  (GATE 1)
**Spec:** §8, §13.5 · **Docs:** 06_TESTING
```
Implement validation.py validate_qwen_output(raw_output, candidate_set) -> ValidationResult with the 6 ordered, short-circuiting checks in SPEC §8,
the normalisation policy D-002, <think> rejection, NaN rejection, and reason strings "validation_failed:<check>".
Implement tests/test_validation.py and tests/test_adversarial_qwen_outputs.py covering every case in docs/06_TESTING.md, plus a property test.
Achieve 100% line+branch coverage of validation.py. Do not touch any other module.
```
**Acceptance:** all adversarial tests pass; coverage 100 %. **Stop here for my manual review.**

## M09 — Gating
**Spec:** §9
```
Implement gating.py: decide(validation_result, span, pool, thresholds) -> Decision. Domain vs general threshold selection
(domain if any candidate came from the domain dictionary), optional margin hook (disabled by default), and a factory that is the ONLY way
to create an AcceptedDecision. Use placeholder thresholds from config. 100% coverage.
```
**Acceptance:** rejected reasons match enum in docs/03_DATA_CONTRACTS.md.

## M10 — Pipeline end-to-end  (GATE 2)
**Spec:** §1, §3
```
Implement pipeline.py correct_page(ocr_json) -> Stage3Result exactly following SPEC §3. Only apply_accepted() may build corrected_text,
and it accepts only AcceptedDecision. Time each stage. Skip the LLM when the pool has one candidate. Preserve punctuation and spacing.
Write tests using MockLLMClient: happy path, garbage LLM output (text unchanged), timeouts, single-candidate skip.
Then run one real fixture page through real Ollama and show me the correction objects.
```
**Acceptance:** garbage-mock run returns `corrected_text == original_text`.

## M11 — Correction log
**Spec:** §11 (log) · **Docs:** 03_DATA_CONTRACTS
```
Implement correction_log.py: append-only JSONL, one line per page, including rejected corrections, latency and config snapshot
(thresholds version, model name, prompt hash, dictionary version, confusion-matrix hash). Add a small reader utility and tests.
```
**Acceptance:** round-trip test; file never rewritten, only appended.

## M12 — Threshold calibration  (GATE 3)
**Spec:** §9
```
Implement scripts/calibrate_thresholds.py: grid search tau_general and tau_domain on the VAL split only, maximising correction precision and
domain recovery subject to a false-correction-rate cap I will specify (default ≤1%). Write chosen values plus the calibration metrics
to config/thresholds.yaml and calibration_runs/<timestamp>.json. Also output a reliability table (confidence bin vs accuracy) for Qwen.
```
**Acceptance:** run logged; thresholds not hand-set.

## M13 — Metrics + 4-way report
**Spec:** §13.1–13.3 · **Docs:** 07_EVALUATION
```
Implement eval/metrics.py (all 10 metrics, incl. oracle candidate coverage) and eval/report_builder.py producing the GT / raw TrOCR / FT TrOCR / +Stage3
table (Markdown + LaTeX) on the TEST split, with per-condition breakdowns. Unit-test metrics on hand-computed examples.
```

## M14 — Ablation
**Spec:** §13.4
```
Implement scripts/run_ablation.py for arms A–E plus E1/E2. Put the E1 free-form Qwen path in scripts/experimental/ only,
never importable from stage3/. Output one comparison table.
```

## M15 — Latency benchmark
**Spec:** §12
```
Implement scripts/benchmark_latency.py: per-page candidate_generation, qwen_inference, validation, total; mean and P95 over the test pages;
cold vs warm; report GPU memory. Apply levers in order (skip non-suspicious, resident model, cap tokens, cache repeated lookups).
Report whether <2 s/page is met and, if not, where time goes.
```

## M16 — Safety audit + Definition of Done  (GATE 4)
**Spec:** §13.5, §16
```
Implement scripts/run_safety_audit.py asserting selected_candidate ∈ candidate_set for every accepted correction in all logs,
and that no rejected correction changed text. Then produce docs/DEFINITION_OF_DONE.md ticking each SPEC §16 item with evidence
(file/test/metric). Mark any unmet item as FAILED, not done.
```
