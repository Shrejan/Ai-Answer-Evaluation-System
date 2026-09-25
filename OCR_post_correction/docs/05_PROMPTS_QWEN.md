# 05 — Qwen Prompt Specification

## System prompt (fixed; from SPEC §7.2, do not edit without a DECISIONS entry)
```
You are an OCR correction verifier.
Your task is to select the most likely original word or phrase from the supplied candidate list.

Rules:
1. You may select ONLY from the candidate list.
2. Never generate a new word.
3. Never modify any other part of the sentence.
4. Never paraphrase.
5. Never improve grammar.
6. Never add information.
7. If the original OCR text is already plausible, select the original.
8. When uncertain, select the original.
9. Return only valid JSON. No explanation, no markdown, no extra text.
```

## User template
```
Context (before): {previous_context}
OCR text: {ocr_text}
Context (after): {next_context}

Candidate list: {candidate_list}

Return exactly:
{"selected_candidate": "...", "confidence": 0.0}
```
`candidate_list` is rendered as a JSON array of strings.

## Ollama request
```json
{"model":"qwen3:1.7b","stream":false,"think":false,"keep_alive":"30m",
 "messages":[{"role":"system","content":"..."},{"role":"user","content":"..."}],
 "format":{"type":"object","additionalProperties":false,
   "properties":{"selected_candidate":{"type":"string","enum":["<c1>","<c2>"]},
                 "confidence":{"type":"number","minimum":0,"maximum":1}},
   "required":["selected_candidate","confidence"]},
 "options":{"temperature":0,"seed":42,"num_predict":40,"num_ctx":2048}}
```
Build `enum` dynamically from the candidate pool. `validation.py` still re-checks everything.

## Rules for the agent
- One span per call. Never batch spans into one prompt.
- Do not add few-shot examples unless a DECISIONS entry approves them (they can leak candidates).
- Any output containing `<think>` is rejected.
- Prompt hash is stored in the config snapshot of every log line.

## Known limitation to document in the paper
Small-model verbalised confidence is poorly calibrated. Calibration (M12) must evaluate confidence vs. correctness; if weak, consider adding a margin check using Ollama logprobs (if the installed version exposes them) as a second gating feature (D-003).
