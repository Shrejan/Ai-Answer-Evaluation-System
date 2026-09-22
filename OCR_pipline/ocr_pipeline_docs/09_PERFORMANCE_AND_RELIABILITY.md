# Performance & Reliability Controls

Treat this as a checklist to implement and keep visible (e.g. as a health/diagnostics endpoint plus startup log lines), not just prose to read once.

## Startup
- [ ] Load Kraken and TrOCR models once at application startup — never per-request.
- [ ] Warm up TrOCR (one dummy inference) before accepting real traffic.

## Concurrency & GPU safety
- [ ] Exactly one GPU worker per GPU.
- [ ] Bound the OCR request queue (reject or queue-wait past a configured depth rather than accepting unbounded concurrent work).
- [ ] Bound `ProcessPoolExecutor` worker count for cropping.
- [ ] Never allow two unbounded concurrent calls into the same TrOCR model instance.

## Image handling
- [ ] Detection downscale copy is kept fully separate from the recognition-source pixels (see `01_ARCHITECTURE.md`) — verify no function accidentally receives the downscaled copy where it expected the original.
- [ ] Crop padding and minimum crop size are config values, not hardcoded numbers scattered through the code.

## GPU memory / batching
- [ ] Conservative default CUDA batch size for 4 GB VRAM.
- [ ] OOM triggers batch-size backoff + retry (see `06_RECOGNITION_TROCR.md`), not a crash.
- [ ] Health/diagnostics endpoint exposes current GPU memory usage (allocated + reserved).

## Observability
- [ ] Record per-stage timings: detection, cropping, recognition, spell-correction, total (already part of the response contract — make sure they're actually measured, not stubbed).
- [ ] Log detected / cropped / recognized / corrected / returned word counts per request, at least at debug level.

## Storage / concurrency correctness
- [ ] Do **not** write all requests to a single shared file (e.g. `extracted_text.txt`) in any code path meant for production use — concurrent requests will overwrite each other's output. Persist per-request (request-specific file/DB row/object storage) or just return results directly to the caller.

## Failure semantics
- [ ] Invalid input → HTTP 4xx with structured error body.
- [ ] Model/infrastructure failure → structured 5xx processing error.
- [ ] Never return `status: ok` with empty text when the actual cause was decode failure or model unavailability.

## Suggested `/health` response shape

```json
{
  "status": "ok",
  "gpu": {
    "available": true,
    "device": "cuda:0",
    "allocated_mb": 812,
    "reserved_mb": 1024
  },
  "models": {
    "detector": "kraken-blla",
    "recognizer": "microsoft/trocr-base-handwritten",
    "warmed_up": true
  },
  "queue_depth": 0
}
```
