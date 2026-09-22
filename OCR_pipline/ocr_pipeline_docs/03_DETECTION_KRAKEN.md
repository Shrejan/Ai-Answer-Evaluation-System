# Detection — Kraken with Adaptive Downscaling

## Goal
Fast, robust line detection without sacrificing the recognition-stage image quality.

## Steps

1. Decode the upload into the original-resolution image. This is the recognition source — keep a reference to it untouched for the rest of the pipeline.
2. Measure width/height; compute `longest_side = max(width, height)`.
3. Compare against `DETECTION_MAX_DIMENSION` (config).
   - If `longest_side > DETECTION_MAX_DIMENSION`: create a **temporary** downscaled copy for Kraken. Record the exact `scale_factor = DETECTION_MAX_DIMENSION / longest_side`.
   - Else: run Kraken directly on the original image; `scale_factor = 1.0`.
4. Run Kraken `blla` segmentation on the detection copy only.
5. For every polygon Kraken returns, multiply all coordinates by `1 / scale_factor` and round to the nearest pixel in a way that's safe (avoid off-by-one truncation that clips strokes — round outward slightly rather than inward when in doubt).
6. **Clip all restored coordinates to the bounds of the original image** before anything downstream touches them (cropping will otherwise index out of range or clip incorrectly).
7. Discard the temporary downscaled copy — it must never be passed to TrOCR or stored as "the" page image.

## Edge cases to handle explicitly

- Page smaller than `DETECTION_MAX_DIMENSION`: no downscaling, `scale_factor = 1.0`, straightforward passthrough.
- Kraken returns zero lines: this is a valid "blank page" result, not an error — propagate an empty line list downstream.
- Kraken returns overlapping or degenerate (zero-area) polygons: filter these out before reading-order reconstruction; log them as detection noise, don't crash.
- Extremely large pages (well above typical scan size): confirm downscaling actually bounds Kraken's runtime — this is the whole point of the stage, so add a timing assertion in tests.

## What NOT to do here

- Do not run any denoise/contrast/threshold/deskew step before handing the image to Kraken. Kraken should see the plain decoded image (downscaled or not) — nothing else.
- Do not let the downscaled copy leak into any later stage's function signature — recognition and cropping should only ever receive the original-resolution image plus restored-coordinate polygons.

## Function signature sketch

```python
def detect_lines(original_image: np.ndarray, config: DetectionConfig) -> list[Polygon]:
    """Returns polygons in ORIGINAL image coordinate space, already clipped to bounds."""
```

Keep the downscale-and-restore logic entirely inside this function so callers never see the intermediate representation.
