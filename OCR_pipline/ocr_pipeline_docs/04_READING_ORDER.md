# Reading-Order Reconstruction

## Line record schema (internal, not the API response)

Each detected line becomes a record with:
- `line_id` (int) — internal, stable once assigned.
- `polygon` — restored original-resolution polygon.
- `bounds` — bounding box derived from the polygon.
- `vertical_center` — for row grouping.
- `estimated_height` — for row-grouping tolerance.
- `recognition_status` — e.g. `pending / recognized / discarded_small / discarded_failed`.
- `recognized_text` — filled in after stage 8.
- `line_confidence` — optional, filled in after stage 8/10.

## Algorithm

1. Compute `estimated_height` per polygon (bounding-box height is fine as a first pass).
2. Compute `median_height` across all polygons on the page.
3. Set `vertical_tolerance = ROW_GROUPING_TOLERANCE_FRACTION * median_height` (config, e.g. 0.5).
4. Group polygons into rows: sort by `vertical_center`, then greedily assign a polygon to an existing row if its center is within `vertical_tolerance` of that row's running average center; otherwise start a new row.
5. Within each row, sort polygons left-to-right by their leftmost x-coordinate.
6. Concatenate rows top-to-bottom to produce the final ordered line list; assign `line_id` sequentially at this point.

## Why this matters

This ordering is the backbone of the whole response: `line_id` must stay attached to a line through cropping, recognition, and correction even when some crops fail or get discarded (see `05_CROPPING_AND_PARALLELISM.md`). Never re-sort or re-derive order after this stage — everything downstream trusts `line_id` ordering as final.

## Edge cases

- Single-column vs multi-column pages: the row-grouping-then-left-to-right approach handles simple multi-column layouts reasonably but is not a full layout analyzer — document this limitation; if the thesis eventually needs multi-column tables, that's a separate future enhancement, not part of this pipeline's scope.
- Zero lines: return an empty ordered list; downstream stages must handle this without special-casing (they should already, since it's just "loop over zero items").
- Two polygons with identical vertical center but overlapping horizontally (rare detector artifact): break ties by leftmost x, then by polygon area descending (prefer the larger, more likely genuine detection) — flag this scenario in logs since it usually indicates a detection issue worth reviewing.
