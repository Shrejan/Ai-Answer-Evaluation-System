# Request / Response Contract

## Request

Fields:
- `request_id` (string, optional) — client-supplied identifier. If absent, the service generates a UUID.
- `page_id` (string or int, required) — identifies the page within a document; validate per the surrounding application's convention (stable label or integer).
- `image` (file/bytes, required) — the page image.
- `document_id` / `tenant_id` (optional) — passthrough metadata if the wider system needs it. Do not use these for any logic inside this module.
- `spell_correction` (bool, optional, default = config default) — per-request override.

Example (multipart or JSON+base64, pick one convention consistent with the rest of the codebase):

```json
{
  "request_id": "req_123",
  "page_id": "page_001",
  "spell_correction": true
}
```

## Response — success

```json
{
  "request_id": "req_123",
  "page_id": "page_001",
  "text": "The mitochondria is the powerhouse of the cell.\nIt produces ATP through respiration.",
  "words": [
    {
      "id": 0,
      "text": "The",
      "confidence": 0.94,
      "line_id": 0,
      "bbox": [120, 45, 168, 70],
      "raw_text": "The",
      "corrected": false
    }
  ],
  "line_count": 2,
  "recognized_line_count": 2,
  "timings_ms": {
    "decode": 12,
    "detection": 180,
    "cropping": 40,
    "recognition": 310,
    "spell_correction": 8,
    "total": 550
  },
  "model": {
    "detector": "kraken-blla",
    "recognizer": "microsoft/trocr-base-handwritten",
    "spell_correction": "symspell"
  },
  "status": "ok"
}
```

Notes:
- `bbox`, `line_id`, `raw_text`, `corrected` on a word record are optional/diagnostic — keep them, but don't let their absence break consumers.
- `text` and `words` must be built from the same ordered representation (see `01_ARCHITECTURE.md`).
- An empty page (no lines detected) must still return this shape: `"text": ""`, `"words": []`, `"line_count": 0`, `"recognized_line_count": 0`, valid `timings_ms`, `"status": "ok"`. It is not an error for a page to be blank.

## Response — failure

Two distinct failure classes, don't conflate them:

1. **Invalid input** (bad/missing image, invalid page_id, unreadable file) → standard HTTP 4xx with a structured error body.
2. **Processing failure** (model unavailable, detection crashed, GPU OOM after retries exhausted) → structured processing-error body, HTTP 5xx.

```json
{
  "request_id": "req_123",
  "page_id": "page_001",
  "status": "error",
  "error": {
    "type": "processing_error",
    "stage": "recognition",
    "message": "TrOCR inference failed after batch-size backoff retries"
  }
}
```

Never return a "successful-looking" empty response (`status: ok`, empty text) when the real cause is decode failure or model unavailability — that must be an explicit error.

## Word record fields (canonical list)

| Field | Required | Notes |
|---|---|---|
| `id` | yes | zero-based, unique within page, in reading order |
| `text` | yes | final word text (post-correction if enabled) |
| `confidence` | yes | float 0–1, OCR-derived (see `08_CONFIDENCE_SCORING.md`) |
| `line_id` | no | internal ordered line index |
| `bbox` | no | pixel box in original-image coordinates |
| `raw_text` | no | pre-correction text, for diagnostics |
| `corrected` | no | bool, whether spell correction changed this word |

## Reading order guarantee

`words` must follow top-to-bottom line order, then left-to-right within each line — matching the line order used to assemble `text`.
