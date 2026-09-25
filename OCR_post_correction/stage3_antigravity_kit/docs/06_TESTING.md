# 06 — Testing Strategy

## Layers
| Layer | Location | LLM |
|---|---|---|
| Unit | `tests/test_*.py` | mocked |
| Adversarial | `tests/test_adversarial_qwen_outputs.py` | mocked |
| Pipeline | `tests/test_pipeline_end_to_end.py` | mocked + scripted |
| Integration | `tests/integration/` `@pytest.mark.ollama` | real Ollama |
| Audit | `scripts/run_safety_audit.py` | logs |

## Worked-example fixtures (`tests/fixtures/worked_examples.yaml`)
| Input | Expected candidate present | Kind |
|---|---|---|
| `transisttor` | `transistor` | SymSpell / domain |
| `denital` | `dental` | domain / SymSpell |
| `pregnue` | `pregnant` (or agreed target) | confusion / SymSpell |
| `nubation` | `incubation` (agreed target) | phrase / SymSpell |
| `attack blood pressure` | `check blood pressure` | phrase retrieval |
| `local lacking` | `lactating` | merge |
| `volontrar-y` | `voluntary` | split/punctuation |
| `beat`, `attack` | detector flags in context | detector |
Confirm each expected target with the author in `DECISIONS.md` before hard-coding.

## Mandatory adversarial cases (all must → reject + keep original)
1. Candidate plus extra words (`"transistor amplifier"`)
2. Prose/explanation instead of JSON
3. Malformed JSON; JSON with trailing prose
4. Candidate not in set
5. Missing keys; extra keys; wrong types (confidence as string)
6. Confidence < 0 or > 1; NaN
7. Empty string, null, `None`
8. Timeout; runtime exception
9. `<think>...</think>` block in output
10. Case-variant candidate (`Dental` vs `dental`) per normalisation policy D-002
11. Unicode lookalike / trailing whitespace variants
12. Fenced JSON (```json ... ```) is accepted only if the inner JSON is valid and exact

## Property tests
- For random candidate sets and random garbage outputs, `validate_qwen_output(...).accepted` implies `selected in candidate_set`.
- `pipeline.correct_page` with a mock that returns garbage always returns `corrected_text == original_text`.

## Coverage targets
`validation.py` and `gating.py` 100 % line + branch; overall ≥ 85 %.
