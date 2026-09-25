# 02 — Architecture

## Data flow
```
Stage 2 page JSON
   → detector.py            (cheap rules; no LLM)          → SuspicionRecord[]
   → candidates/*           (A SymSpell, B confusion, C domain dict, D phrase retrieval)
   → aggregator.py          (union, original injected, rank, cap 6)
   → [|pool|==1 ? skip]
   → qwen/verifier.py       (Ollama, JSON-schema constrained, untrusted output)
   → validation.py          (hard gate)                     → ValidationResult
   → gating.py              (tau_general / tau_domain)      → Decision
   → pipeline.apply_accepted (ONLY place corrected_text changes)
   → correction_log.py      (JSONL)
   → Stage3Result → Stage 4
```

## Module responsibilities
| Module | Responsibility | Must not |
|---|---|---|
| `detector.py` | Flag suspicious tokens / group contiguous spans | Call LLM |
| `candidates/*` | Return `list[str]` per source | Return the LLM's opinion |
| `aggregator.py` | Merge, dedupe, rank, cap; inject original | Drop the original |
| `qwen/client.py` | HTTP to Ollama, timeout, retries=0 | Interpret output |
| `qwen/verifier.py` | Build prompt, call client, return raw string | Trust output |
| `validation.py` | Parse + check output against candidate set | Have side effects |
| `gating.py` | Threshold and margin logic | Modify text |
| `pipeline.py` | Orchestrate; assemble text | Bypass validation |
| `correction_log.py` | Append-only JSONL | Mutate results |

## Key interfaces (Protocols)
```python
class LLMClient(Protocol):
    def select(self, prompt: PromptBundle, timeout_s: float) -> str: ...   # raw text only

class CandidateSource(Protocol):
    name: str
    def propose(self, span: Span) -> list[str]: ...
```
The `LLMClient` protocol lets tests inject a `MockLLMClient` for adversarial cases.

## Safety structure
`apply_accepted(text_tokens, decision: AcceptedDecision)` accepts only an `AcceptedDecision` that can be constructed solely by `gating.decide()` after `ValidationResult.accepted is True`. Use a private constructor / factory to make bypass impossible by type.

## Tokenisation policy
Whitespace tokens from Stage 2 `words[]`. Leading/trailing punctuation is split off before lookup and re-attached after replacement (see DECISIONS D-004). Character offsets in the correction log refer to `original_text`.
