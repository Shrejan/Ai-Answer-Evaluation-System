# Recognition — Original-Resolution TrOCR

## Model
`TrOCRProcessor` + `VisionEncoderDecoderModel` from `microsoft/trocr-base-handwritten`. Load once at application startup, not per-request.

## Required runtime behavior

- GPU execution when CUDA is available; CPU fallback otherwise (log which path is active).
- FP16 on CUDA.
- Conservative batch size by default, sized for 4 GB VRAM (`TROCR_BATCH_SIZE` in config — pick a small starting value like 4–8 and treat it as a deployment knob, not a fixed constant; profile and adjust).
- Beam search with `TROCR_BEAM_WIDTH` (config) and early stopping enabled.
- Model warm-up at startup: run one dummy inference before accepting real traffic so the first real request isn't penalized by lazy CUDA init / cudnn autotune.
- Use a dedicated CUDA stream only if profiling shows a measurable throughput benefit — don't add complexity speculatively.
- Wrap all inference in `torch.inference_mode()` (gradients disabled).
- **One model-serving worker per GPU** — never instantiate multiple concurrent model instances competing for the same 4 GB.

## Batching and OOM handling

- Batch size is a deployment/perf setting, never something that silently changes output correctness.
- On CUDA OOM: catch it, clear the cache, retry with a smaller batch (halve it, down to a floor of 1), and only surface a processing error if the floor batch size still OOMs.
- Concurrent requests sharing one GPU must be serialized behind a **bounded inference queue** — do not let multiple unrestricted async calls hit the same model concurrently (this is a common cause of unpredictable latency/OOM). A single-worker queue consumer is the simplest correct implementation.

## Output per crop

For each successful crop, produce:
- `raw_text` — the decoded TrOCR string, kept for diagnostics even after correction.
- Generation-derived scores needed for confidence (see `08_CONFIDENCE_SCORING.md`) — request `output_scores=True` / `return_dict_in_generate=True` from `.generate()`.

## Line association rules

- Each result maps back to its original `line_id` slot — discarded/failed crops must not shift the `line_id` of lines that come after them.
- An empty decoded string is not promoted to a text line in the final output, but its `recognition_status` (e.g. `"recognized_empty"`) stays available for diagnostics.
- If TrOCR itself throws on a given crop (rare, but plan for it), mark that line `recognition_status = "discarded_failed"` and continue — one bad line must not fail the whole page.

## Function signature sketch

```python
def recognize_lines_batched(
    ordered_lines: list[LineRecord],   # only entries with a valid .crop
    config: RecognitionConfig,
) -> list[LineRecord]:
    """Fills .raw_text and .generation_scores on each line record, batching internally
    according to config.batch_size, with OOM backoff."""
```

## Testing checklist

- A batch that triggers synthetic OOM (mock it) correctly backs off and retries rather than crashing the request.
- Warm-up actually executes before the first real request is served (check via a log line or a startup flag).
- Two lines' results never get swapped when batch order differs from submission order.
