# Combined OCR Pipeline Plan

## 1. Objective

The target system is a fast, reliable handwritten-answer OCR service that combines the strongest architectural properties of `ocr_ptest.py` and `testing.py` while deliberately excluding image enhancement preprocessing.

The final pipeline should:

- Extract handwritten text accurately from page images.
- Use Kraken for robust line detection.
- Downscale only the image used for detection when the page is large.
- Use original-resolution polygons and pixels for TrOCR recognition.
- Avoid denoising, CLAHE, deskewing, thresholding, and artificial upscaling.
- Recognize lines in GPU batches.
- Apply optional spell correction after recognition.
- Return both complete page text and word-level records.
- Preserve reading order and page metadata.
- Report useful timing, model, and confidence information.
- Remain safe on a 4 GB RTX 2050 through controlled batching and one GPU worker.

The central principle is: **reduce resolution only where page-level detection can tolerate it; preserve original detail where character recognition needs it.**

## 2. Recommended architecture

The combined architecture is a staged service:

1. Request and metadata validation.
2. Image decoding.
3. Temporary detection-image downscaling.
4. Kraken line segmentation.
5. Polygon coordinate restoration.
6. Reading-order reconstruction.
7. Original-resolution polygon masking and cropping.
8. Batched TrOCR recognition without extra image preprocessing.
9. Optional spell correction.
10. Word segmentation and confidence construction.
11. Structured response assembly.
12. Timing, health, and diagnostics reporting.

The page image exists in two logical forms:

- The **detection representation**, potentially downscaled for Kraken.
- The **recognition source**, always the original decoded image.

The detection representation must never replace the original recognition source.

## 3. Request contract

Each request should carry or be assigned:

- `request_id`, identifying the OCR request.
- `page_id`, identifying the page within a document.
- The uploaded image.
- Optional document or tenant metadata if required by the client application.

If the client does not provide an identifier, the service should generate a unique request ID. The page ID should be validated as an integer or stable page label according to the surrounding application contract.

The response must be deterministic in structure even when no text is found. An empty page should return an empty `text` value and an empty `words` array, together with zero recognized lines and valid timing information.

## 4. Detection plan: Kraken with adaptive downscaling

The service first decodes the upload into an original-resolution image. It measures the width and height and compares the longest side with a configured detection maximum.

If the page exceeds that maximum, a temporary downscaled copy is created for Kraken. The scale factor is retained. Kraken runs only on this copy, reducing the number of pixels processed and lowering detection latency.

Every polygon returned by Kraken is multiplied by the inverse of the downscale factor and rounded safely to the coordinate space of the original image. Coordinates should be clipped to image bounds before cropping.

If the page is already within the configured size, Kraken can run on the original image and the scale factor is one.

This approach retains the speed improvement requested from `ocr_ptest.py` while ensuring that TrOCR receives the best available character detail.

## 5. Reading-order plan

Kraken line polygons are normalized into line records. Each record should contain:

- An internal line ID.
- The restored original-resolution polygon.
- Polygon bounds.
- Vertical center and estimated height.
- Recognition status.
- Recognized text.
- Optional line confidence.

The records are sorted top to bottom and left to right. The adaptive row grouping from both existing files is appropriate: estimate the median polygon height, use a configurable fraction as the vertical tolerance, group nearby centers into rows, and sort each row horizontally.

The internal ordered line position must remain stable throughout cropping, recognition, spell correction, and response generation.

## 6. Cropping plan with no image preprocessing

For every ordered line, calculate a bounding rectangle from the original-resolution polygon. Add only the configured geometric padding needed to avoid cutting off ascenders, descenders, or edge strokes.

Create a crop from the original image, mask pixels outside the polygon to white, and convert the result to the image representation required by TrOCR.

This is not enhancement preprocessing. It is spatial extraction and isolation of the detected line. The combined pipeline must not perform:

- Denoising.
- CLAHE or other contrast enhancement.
- Deskewing.
- Thresholding or binarization.
- Artificial upscaling.
- Sharpening.
- Morphological operations.

The reason is consistency with the data-collection appearance and the original-resolution design of `ocr_ptest.py`. The TrOCR processor remains responsible only for the model's mandatory tensor conversion, resizing, normalization, and batching behavior.

Very small or invalid crops should be discarded as detection noise. Discarded line slots must be recorded so the response can distinguish detected lines from successfully recognized lines.

## 7. Parallel crop execution

Use a bounded `ProcessPoolExecutor` for independent polygon extraction and masking. The worker should receive only the page array and polygon data required for one crop and return an RGB crop or a failure marker.

All crop jobs can be submitted together, but the parent process must restore the ordered line index after futures complete. The number of processes should be configurable and conservative because passing a large full-page array to many processes can consume memory.

For repeated high-throughput deployment, the implementation should consider shared-memory image transport or a carefully bounded queue. The initial version can retain the existing process-pool design, measure its serialization cost, and reduce worker count if memory pressure outweighs CPU parallelism.

## 8. Recognition plan: original-resolution TrOCR

The successful original-resolution crops are sent to `TrOCRProcessor` and then to `microsoft/trocr-base-handwritten`.

Recognition should use:

- GPU execution when CUDA is available.
- FP16 on CUDA.
- A conservative batch size for 4 GB VRAM.
- Beam search with the configured beam width.
- Early stopping.
- Model warm-up at startup.
- A dedicated CUDA stream where it improves measured throughput.
- Inference mode with gradients disabled.
- One model-serving worker per GPU.

Batch size must be treated as a deployment setting, not an accuracy setting. If an out-of-memory error occurs, the service should reduce the batch size or retry with a smaller configured batch rather than allowing the process to crash.

Concurrent OCR requests should be serialized or placed behind a bounded inference queue when they share one GPU. Running multiple unrestricted `run_in_executor` calls against the same model can cause memory spikes and unpredictable latency.

## 9. Recognition output and line association

Each decoded TrOCR result is associated with the original ordered line slot. Crops that were rejected or failed do not shift the IDs of later lines.

The system should retain both:

- The raw TrOCR text, useful for diagnostics.
- The corrected text, used for the final response when spell correction is enabled.

Empty decoded strings should not appear as text lines, but their recognition status can remain available in internal diagnostics.

## 10. Spell-correction plan

Spell correction is placed after TrOCR recognition and before final word records are produced.

The recommended flow is:

1. Keep the raw recognized line text.
2. Split each line into candidate words while retaining punctuation boundaries.
3. Apply SymSpell only when the dictionary is available and correction is enabled.
4. Preserve capitalization and punctuation as far as possible.
5. Store the corrected line text separately from the raw text.
6. Use corrected text for the final page-level `text` and `words` output.

Spell correction should be conservative. Handwritten answers may contain names, abbreviations, scientific terms, and domain-specific vocabulary that are absent from a general English dictionary. A configurable dictionary, protected vocabulary, or correction threshold should be supported so valid specialized words are not changed unnecessarily.

Correction confidence must not be confused with OCR confidence. A word that is changed by SymSpell should retain its OCR confidence and may additionally carry a correction indicator internally.

## 11. Word-level response plan

The final page response should have the following conceptual fields:

- `request_id`: request identifier such as `page_001`.
- `page_id`: numeric or stable page identifier.
- `text`: final corrected page text, with lines separated consistently.
- `words`: ordered word records.

Each word record should contain:

- A zero-based `id` unique within the page.
- The final word `text`.
- A confidence estimate between 0 and 1.
- Optionally, line ID, character span, bounding box, raw text, or correction status for future diagnostics.

The word list must follow the same reading order as the page text: top-to-bottom lines and left-to-right words within each line.

## 12. Confidence strategy

TrOCR is a sequence-generation model, so word confidence is not automatically available as a simple detector score. The combined implementation should request generation scores and calculate token-level confidence from the model's normalized token probabilities. Special tokens should be excluded.

Token scores must then be aligned to decoded words. A practical first version can use the processor tokenizer's decoded token sequence and aggregate the token probabilities belonging to each word. The word confidence can be the geometric mean or length-normalized mean of its token probabilities, clipped to the range 0 to 1.

If a word is changed by spell correction, the response should continue to represent OCR evidence rather than falsely claiming that the dictionary increased visual certainty. The system may expose separate internal values for OCR confidence and correction confidence.

Confidence values should be documented as model-derived estimates. They are not calibrated probabilities until evaluated against a labeled validation set. Calibration can later be performed using held-out handwritten pages.

## 13. Recommended response behavior

For a successful page, the response should conceptually be:

- Metadata identifying the request and page.
- The complete corrected text.
- Word records in reading order.
- Optionally, line counts, timings, and processing status.

The sample result therefore represents one page whose final text is assembled from recognized and optionally corrected line outputs. The service should not regenerate page text independently from the word list; both should come from the same ordered internal representation so they cannot disagree.

For failures, return a clear HTTP error for invalid input and a structured processing error for model or infrastructure failures. Do not return a successful-looking empty response when the image could not be decoded or the model was unavailable.

## 14. Performance and reliability controls

The combined pipeline should use the following controls:

- Load models once during application startup.
- Warm up TrOCR before accepting traffic.
- Use one GPU worker per GPU.
- Bound the OCR request queue.
- Bound process-pool workers.
- Keep detection downscale separate from recognition pixels.
- Keep crop padding and minimum crop sizes configurable.
- Use conservative CUDA batch sizes.
- Retry recognition with a smaller batch after an out-of-memory event when safe.
- Record detection, cropping, recognition, spell-correction, and total timings.
- Expose GPU memory usage through health diagnostics.
- Log detected, cropped, recognized, corrected, and returned word counts.
- Avoid writing all requests to one shared `extracted_text.txt` file in production, because concurrent requests can overwrite each other. Persist results with request-specific storage or return them directly to the client.

## 15. Validation plan

Accuracy and speed should be evaluated separately for each stage:

1. Detection recall: percentage of handwritten lines found by Kraken.
2. Crop integrity: percentage of polygons whose strokes are not clipped.
3. Recognition quality: character error rate and word error rate on raw TrOCR output.
4. Spell-correction impact: changes in word error rate after correction.
5. Confidence quality: correlation and calibration of confidence against correct words.
6. End-to-end latency: detection, crop, recognition, correction, and total time.
7. GPU behavior: peak VRAM, reserved VRAM, and out-of-memory frequency.
8. Concurrency behavior: queue latency and throughput under multiple page requests.

Test pages should include clean handwriting, faint handwriting, slanted lines, multiple rows, columns, punctuation, numbers, domain terms, and pages larger than the detection limit.

## 16. Final recommended workflow

The final workflow is:

1. Receive `request_id`, `page_id`, and image upload.
2. Validate and decode the original image.
3. Create a temporary downscaled detection copy only when needed.
4. Run Kraken blla on that copy.
5. Map all polygons back to original coordinates.
6. Sort polygons into reading order.
7. Extract white-background polygon crops from the original image.
8. Do not apply image enhancement preprocessing.
9. Run TrOCR on original-resolution crops in controlled GPU batches.
10. Collect raw line text and generation-derived confidence information.
11. Apply optional conservative spell correction.
12. Split corrected lines into ordered words.
13. Assign stable word IDs and word confidence estimates.
14. Build page text from the same ordered words and lines.
15. Return the metadata, page text, words, counts, and diagnostics to the client.

This combination preserves the best speed property of `ocr_ptest.py`, the recognition and spell-correction workflow of `testing.py`, and the requested no-preprocessing behavior. It is therefore well aligned with fast, original-detail handwritten text extraction rather than generic image enhancement.