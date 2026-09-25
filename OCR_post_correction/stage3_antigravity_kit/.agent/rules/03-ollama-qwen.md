# Rule: Ollama + Qwen3-1.7B (Always On)
- Call `/api/chat` with: `stream=false`, `think=false`, `format=<JSON schema>`, `options={temperature:0, seed:42, num_predict:40, num_ctx:2048}`, `keep_alive="30m"`.
- Qwen3 defaults to "thinking" mode: it must be disabled, and any `<think>...</think>` text in output must cause **rejection**, not stripping.
- The JSON schema constrains `selected_candidate` to an `enum` of the candidate list, but `validation.py` re-checks anyway (defence in depth).
- Model self-reported confidence is weakly calibrated; treat it as a gating feature only after calibration (M12).
- Hard client timeout (default 10 s) → rejection.
