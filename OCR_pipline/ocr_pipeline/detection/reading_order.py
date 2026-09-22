"""Reading-order reconstruction via row grouping (doc 04)."""

from __future__ import annotations

import logging
from statistics import median
from typing import List

from ocr_pipeline.config import ReadingOrderConfig
from ocr_pipeline.models import LineRecord, Polygon, bounds_from_polygon, polygon_area

logger = logging.getLogger(__name__)


def _tie_break_key(line: LineRecord) -> tuple:
    x0 = line.bounds[0]
    area = polygon_area(line.polygon)
    return (x0, -area)


def assign_reading_order(
    polygons: List[Polygon],
    config: ReadingOrderConfig,
) -> List[LineRecord]:
    if not polygons:
        return []

    records: List[LineRecord] = []
    for poly in polygons:
        bounds, vcenter, height = bounds_from_polygon(poly)
        records.append(
            LineRecord(
                line_id=-1,
                polygon=poly,
                bounds=bounds,
                vertical_center=vcenter,
                estimated_height=height,
            )
        )

    records.sort(key=lambda r: r.vertical_center)
    median_height = median(r.estimated_height for r in records) or 1.0
    tolerance = config.row_grouping_tolerance_fraction * median_height

    rows: List[List[LineRecord]] = []
    row_centers: List[float] = []

    for record in records:
        assigned = False
        for idx, center in enumerate(row_centers):
            if abs(record.vertical_center - center) <= tolerance:
                rows[idx].append(record)
                row_centers[idx] = sum(r.vertical_center for r in rows[idx]) / len(
                    rows[idx]
                )
                assigned = True
                break
        if not assigned:
            rows.append([record])
            row_centers.append(record.vertical_center)

    ordered: List[LineRecord] = []
    line_id = 0
    for row in rows:
        row.sort(key=_tie_break_key)
        for record in row:
            if len(row) > 1:
                same_center = [
                    r for r in row if abs(r.vertical_center - record.vertical_center) < 1.0
                ]
                if len(same_center) > 1:
                    logger.debug(
                        "Tie-break applied for overlapping detections at y=%.1f",
                        record.vertical_center,
                    )
            record.line_id = line_id
            ordered.append(record)
            line_id += 1
    return ordered
