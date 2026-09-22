# Cropping (No Preprocessing) + Parallel Execution

## Per-line cropping steps

For every ordered line record:
1. Compute a bounding rectangle from the original-resolution polygon.
2. Add `CROP_PADDING_PX` (config) — just enough geometric padding to avoid cutting ascenders/descenders/edge strokes. This is spatial padding only, not enhancement.
3. Crop that rectangle from the **original** image (never the detection copy).
4. Mask pixels outside the polygon (but inside the padded rectangle) to white.
5. Convert to whatever image representation `TrOCRProcessor` expects (PIL RGB is the usual choice).

## Explicitly forbidden in this stage (and everywhere else in the pipeline)

Denoising · CLAHE / contrast enhancement · deskewing · thresholding / binarization · artificial upscaling · sharpening · morphological operations.

The only image operations allowed here are: rectangle crop, polygon mask-to-white, and the color-space/tensor conversion TrOCR's own processor mandates. If a code reviewer (or Cursor) is tempted to "improve" recognition by adding any of the above, that's a scope violation — flag it, don't silently add it.

## Discarding invalid crops

- Any crop below `MIN_CROP_SIZE_PX` (config, both width and height) is discarded as detection noise.
- Discarded lines get `recognition_status = "discarded_small"` on their line record — **the line_id slot stays in the ordered list**, it just won't produce recognized text or word records. Downstream code must not compact/reindex around discarded slots.

## Parallel crop execution

- Use a bounded `ProcessPoolExecutor` (`PROCESS_POOL_WORKERS` from config).
- Each worker receives only what it needs for one crop: the (read-only) page array reference and one polygon's data — not the whole line-record object graph.
- Worker returns either an RGB crop array or a failure marker (never raises uncaught across the process boundary).
- Submit all crop jobs together; as futures complete, **write results back into the ordered line-record list by `line_id`**, not by completion order — this is what restores correct ordering after parallel execution.
- Be mindful of memory: passing a large full-page array to many worker processes multiplies memory use. Keep `PROCESS_POOL_WORKERS` conservative by default; document that shared-memory transport (e.g. `multiprocessing.shared_memory`) is a future optimization if serialization cost proves high, not something to build in v1 unless profiling shows it's needed.

## Function signature sketch

```python
def crop_lines_parallel(
    original_image: np.ndarray,
    ordered_lines: list[LineRecord],
    config: CropConfig,
) -> list[LineRecord]:
    """Mutates/returns ordered_lines with .crop (or None) and .recognition_status set,
    in the same line_id order they came in."""
```

## Testing checklist for this stage

- A line polygon at the very edge of the page doesn't crash on padding (clip to image bounds).
- A degenerate/zero-area polygon is discarded, not passed to the worker pool.
- Worker pool result reassembly is order-correct even when jobs complete out of submission order (test by injecting artificial per-worker delay).
