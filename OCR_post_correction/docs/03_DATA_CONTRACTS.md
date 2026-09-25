# 03 — Data Contracts

## Input (per page) — from Stage 2
```json
{"request_id":"page_001","page_id":1,"text":"General health medical camps ...",
 "words":[{"id":0,"text":"General","confidence":0.98}]}
```
Validation: `words` non-empty; `confidence` in [0,1]; `" ".join(w.text)` should equal `text` (log a warning if not).

## Internal objects (`stage3/schemas.py`, pydantic v2)
| Model | Fields |
|---|---|
| `Word` | id:int, text:str, confidence:float |
| `Span` | span_ids:list[int], original_text:str, prev_context:str, next_context:str, is_phrase:bool |
| `SuspicionRecord` | token:str, span_ids:list[int], reasons:list[str], severity:float |
| `CandidatePool` | candidates:list[str], sources:dict[str,list[str]] (candidate → source names) |
| `PromptBundle` | system:str, user:str, schema:dict |
| `ValidationResult` | accepted:bool, selected:str\|None, confidence:float\|None, reason:str\|None |
| `Decision` | accepted:bool, threshold_used:float, threshold_type:"general"\|"domain", reason:str\|None |
| `Correction` | see below |
| `Stage3Result` | request_id, original_text, corrected_text, corrections:list[Correction], stage3_latency_ms |

## Correction object (SPEC §10)
```json
{"original":"denital","replacement":"dental","position":[12,19],
 "candidate_set":["denital","dental"],"candidate_source":["domain_dictionary","symspell"],
 "qwen_confidence":0.94,"threshold_used":0.90,"threshold_type":"domain",
 "accepted":true,"rejection_reason":null}
```
Rejection reasons (enum): `validation_failed:<check>`, `below_confidence_threshold`, `below_margin`, `llm_timeout`, `llm_exception`, `single_candidate_skip`.

## Latency object
`candidate_generation_ms, qwen_inference_ms, validation_ms, total_ms` (floats, ms).

## Log line (JSONL, one per page)
`request_id, page_id, original_text, corrected_text, corrections[], stage3_latency_ms, config_snapshot{thresholds_version, model, dictionary_version, confusion_matrix_hash}`.

## Files and formats
| File | Format |
|---|---|
| `data/domain_dictionary/ece_terms.txt` | one term per line, lowercase |
| `data/domain_dictionary/general_dictionary.txt` | `word<TAB>frequency` per line |
| `data/confusion_matrix/char_confusion.json` | SPEC §6 schema |
| `data/phrase_corpus/domain_phrases.jsonl` | `{"id":..,"text":..,"source":..}` |
| `data/stage2_pairs/{train,val,test}.jsonl` | `{"id","gt_text","pred_text"}` |
