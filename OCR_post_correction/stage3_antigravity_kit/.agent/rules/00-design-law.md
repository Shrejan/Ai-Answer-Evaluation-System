# Rule: Design Law (Always On)
- LLM output is untrusted input. Parse → validate → gate → only then apply.
- `selected_candidate in candidate_set` is an invariant; assert it in code paths and in tests.
- The original token is ALWAYS in the candidate set.
- If `len(candidate_set) == 1`, skip the LLM.
- Never send the full page to the LLM; send a window of 5–8 tokens each side, one span per call.
- On any exception, timeout, or ambiguity: keep original.
- No code path may write LLM free text into `corrected_text`.
