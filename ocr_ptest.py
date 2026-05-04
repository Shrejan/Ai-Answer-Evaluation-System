"""
Handwritten OCR Pipeline — Kraken blla + TrOCR-base
Optimised for RTX 2050 (4 GB VRAM)
Key improvements over v2:
  - Kraken fed a downscaled copy (max 1200 px) for speed
  - TrOCR gets original‑resolution polygon crops, white background
  - No extra preprocessing – matches data‑collection pipeline behaviour
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Kraken
from kraken import blla
from kraken.lib.vgsl import TorchVGSLModel

# ── logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("ocr_pipeline")


# ════════════════════════════════════════════════════════════════════
#  CONFIGURATION  ← tweak everything here
# ════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    # ── Model ────────────────────────────────────────────────────────
    trocr_model: str = "microsoft/trocr-base-handwritten"

    # ── Inference ────────────────────────────────────────────────────
    max_new_tokens: int = 128
    beam_size: int = 4
    batch_size_cuda: int = 12   # safe for 4 GB VRAM
    batch_size_cpu: int = 4

    # ── Kraken speedup ───────────────────────────────────────────────
    kraken_max_side: int = 1200   # downscale copy if longer side > this

    # ── Polygon crop (minimal, data‑collection‑style) ────────────────
    crop_padding_x: int = 10
    crop_padding_y: int = 4
    min_crop_h: int = 12
    min_crop_w: int = 24

    # ── Reading order ────────────────────────────────────────────────
    row_tolerance: float = 0.55

    # ── Parallelism ──────────────────────────────────────────────────
    preproc_workers: int = 2


CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16 = DEVICE.type == "cuda"

log.info(f"Device: {DEVICE}  |  FP16: {USE_FP16}  |  Preproc workers: {CFG.preproc_workers}")


# ════════════════════════════════════════════════════════════════════
#  DATA TYPES
# ════════════════════════════════════════════════════════════════════
@dataclass
class TextLine:
    index: int
    polygon: np.ndarray            # (N, 2) pixel coords
    crop_pil: Optional[Image.Image] = None
    text: str = ""


@dataclass
class OCRResult:
    lines_detected: int
    lines_recognized: int
    text: str
    timings_ms: dict = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════
#  GLOBAL MODEL STATE
# ════════════════════════════════════════════════════════════════════
class _Models:
    kraken_model: TorchVGSLModel = None
    trocr_processor: TrOCRProcessor = None
    trocr_model: VisionEncoderDecoderModel = None
    cuda_stream: Optional[torch.cuda.Stream] = None
    preproc_pool: ProcessPoolExecutor = None


G = _Models()


# ════════════════════════════════════════════════════════════════════
#  STARTUP / SHUTDOWN
# ════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    log.info("Loading models …")

    # Kraken
    G.kraken_model = TorchVGSLModel.load_model("blla.mlmodel")
    G.kraken_model.eval()
    if DEVICE.type == "cuda":
        G.kraken_model.nn.to(DEVICE)
    log.info("Kraken blla loaded ✓")

    # TrOCR
    G.trocr_processor = TrOCRProcessor.from_pretrained(CFG.trocr_model)
    G.trocr_model = VisionEncoderDecoderModel.from_pretrained(CFG.trocr_model)
    G.trocr_model.to(DEVICE).eval()
    if USE_FP16:
        G.trocr_model = G.trocr_model.half()
        torch.backends.cudnn.benchmark = True

    # torch.compile for extra throughput
    try:
        G.trocr_model = torch.compile(G.trocr_model, mode="reduce-overhead")
        log.info("torch.compile() enabled ✓")
    except Exception as e:
        log.warning(f"torch.compile() skipped: {e}")

    if DEVICE.type == "cuda":
        G.cuda_stream = torch.cuda.Stream()

    # Preprocessing pool
    G.preproc_pool = ProcessPoolExecutor(max_workers=CFG.preproc_workers)

    # Warm-up
    _warmup()
    log.info(f"Kraken is on {next(G.kraken_model.nn.parameters()).device}")
    log.info(f"TrOCR is on {next(G.trocr_model.parameters()).device}")
    log.info(f"Pipeline ready in {(time.perf_counter()-t0)*1000:.0f} ms")
    yield

    G.preproc_pool.shutdown(wait=False)
    log.info("Shutdown complete.")


def _warmup():
    dummy = Image.fromarray(np.full((64, 256, 3), 200, dtype=np.uint8))
    inputs = G.trocr_processor(images=[dummy], return_tensors="pt")
    pv = inputs.pixel_values.to(DEVICE)
    if USE_FP16:
        pv = pv.half()
    with torch.inference_mode():
        G.trocr_model.generate(pv, max_new_tokens=4)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    log.info("Warm-up complete ✓")


# ════════════════════════════════════════════════════════════════════
#  STEP 1 — DETECTION  (Kraken blla, with optional downscale for speed)
# ════════════════════════════════════════════════════════════════════
def detect_lines(image_pil: Image.Image) -> List[TextLine]:
    """
    Run Kraken segmentation.
    If any side > kraken_max_side, feed a downscaled copy to Kraken,
    then map polygons back to original coordinates.
    """
    w, h = image_pil.size
    longest = max(w, h)
    scale = 1.0
    kraken_input = image_pil

    if longest > CFG.kraken_max_side:
        scale = longest / CFG.kraken_max_side
        kraken_input = image_pil.resize(
            (int(w / scale), int(h / scale)), Image.LANCZOS
        )

    seg = blla.segment(kraken_input, model=G.kraken_model)

    lines: List[TextLine] = []
    for i, line in enumerate(seg.lines):
        if line.boundary is None:
            continue
        poly = np.array(line.boundary, dtype=np.float32)  # sub‑pixel precision
        if poly.shape[0] < 3:
            continue
        # Scale back to original coordinates if we resized
        if scale != 1.0:
            poly *= scale
            poly = np.round(poly).astype(np.int32)
        else:
            poly = poly.astype(np.int32)

        lines.append(TextLine(index=i, polygon=poly))

    return _sort_reading_order(lines)


def _sort_reading_order(lines: List[TextLine]) -> List[TextLine]:
    if not lines:
        return lines
    y_centers = [np.mean(l.polygon[:, 1]) for l in lines]
    heights   = [np.ptp(l.polygon[:, 1]) for l in lines]
    median_h  = float(np.median(heights)) if heights else 30.0
    tol = median_h * CFG.row_tolerance

    paired = sorted(zip(y_centers, lines), key=lambda p: p[0])
    rows: List[List[Tuple[float, TextLine]]] = []
    for y_c, line in paired:
        placed = False
        for row in rows:
            row_y = np.mean([r[0] for r in row])
            if abs(y_c - row_y) < tol:
                row.append((y_c, line))
                placed = True
                break
        if not placed:
            rows.append([(y_c, line)])

    ordered: List[TextLine] = []
    for row in rows:
        row_sorted = sorted(row, key=lambda p: np.mean(p[1].polygon[:, 0]))
        ordered.extend(l for _, l in row_sorted)
    return ordered


# ════════════════════════════════════════════════════════════════════
#  STEP 2 — CROP (polygon mask, white bg, NO extra preprocessing)
# ════════════════════════════════════════════════════════════════════
def _crop_single_worker(args: Tuple[np.ndarray, np.ndarray]) -> Optional[np.ndarray]:
    """
    Worker function for ProcessPoolExecutor.
    Takes (full‑page BGR, polygon XY) → clean RGB line crop.
    The polygon is already at original image scale.
    """
    image_np, polygon = args

    x_min = max(0, int(polygon[:, 0].min()) - CFG.crop_padding_x)
    x_max = min(image_np.shape[1], int(polygon[:, 0].max()) + CFG.crop_padding_x)
    y_min = max(0, int(polygon[:, 1].min()) - CFG.crop_padding_y)
    y_max = min(image_np.shape[0], int(polygon[:, 1].max()) + CFG.crop_padding_y)

    crop_h = y_max - y_min
    crop_w = x_max - x_min
    if crop_h < CFG.min_crop_h or crop_w < CFG.min_crop_w:
        return None

    crop = image_np[y_min:y_max, x_min:x_max].copy()

    local_poly = polygon.copy()
    local_poly[:, 0] -= x_min
    local_poly[:, 1] -= y_min
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.fillPoly(mask, [local_poly.astype(np.int32)], 255)
    crop[mask == 0] = 255   # white background

    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def preprocess_parallel(
    image_bgr: np.ndarray,
    lines: List[TextLine],
) -> List[Tuple[int, Image.Image]]:
    """
    Submit all crops to process pool simultaneously.
    Returns list of (original_index, PIL_image) for successful crops.
    """
    args = [(image_bgr, line.polygon) for line in lines]
    futures = {G.preproc_pool.submit(_crop_single_worker, a): i for i, a in enumerate(args)}

    results: List[Tuple[int, Image.Image]] = []
    for future in as_completed(futures):
        idx = futures[future]
        try:
            rgb = future.result()
            if rgb is not None:
                pil = Image.fromarray(rgb)
                results.append((idx, pil))
        except Exception as e:
            log.warning(f"Crop failed for line {idx}: {e}")

    results.sort(key=lambda x: x[0])
    return results


# ════════════════════════════════════════════════════════════════════
#  STEP 3 — RECOGNITION  (TrOCR, GPU batched)
# ════════════════════════════════════════════════════════════════════
def recognize_batch(pil_images: List[Image.Image]) -> List[str]:
    batch_size = CFG.batch_size_cuda if DEVICE.type == "cuda" else CFG.batch_size_cpu
    all_texts: List[str] = []

    for i in range(0, len(pil_images), batch_size):
        batch = pil_images[i : i + batch_size]
        inputs = G.trocr_processor(images=batch, return_tensors="pt", padding=True)
        pixel_values = inputs.pixel_values

        if DEVICE.type == "cuda":
            pixel_values = pixel_values.pin_memory()
            with torch.cuda.stream(G.cuda_stream):
                pixel_values = pixel_values.to(DEVICE, non_blocking=True)
                if USE_FP16:
                    pixel_values = pixel_values.half()
                with torch.inference_mode():
                    generated_ids = G.trocr_model.generate(
                        pixel_values,
                        max_new_tokens=CFG.max_new_tokens,
                        num_beams=CFG.beam_size,
                        early_stopping=True,
                    )
            G.cuda_stream.synchronize()
        else:
            pixel_values = pixel_values.to(DEVICE)
            with torch.inference_mode():
                generated_ids = G.trocr_model.generate(
                    pixel_values,
                    max_new_tokens=CFG.max_new_tokens,
                    num_beams=CFG.beam_size,
                    early_stopping=True,
                )

        texts = G.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)
        all_texts.extend(texts)

    return all_texts


# ════════════════════════════════════════════════════════════════════
#  FULL PIPELINE
# ════════════════════════════════════════════════════════════════════
def run_pipeline(image_bgr: np.ndarray) -> OCRResult:
    timings: dict = {}
    t_start = time.perf_counter()

    # PIL for Kraken
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    # 1. Detection (with optional downscale)
    t0 = time.perf_counter()
    lines = detect_lines(image_pil)
    timings["detection_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    log.info(f"[1] Detected {len(lines)} lines ({timings['detection_ms']} ms)")

    if not lines:
        return OCRResult(lines_detected=0, lines_recognized=0, text="", timings_ms=timings)

    # 2. Parallel cropping (original BGR, original‑scale polygons)
    t0 = time.perf_counter()
    indexed_crops = preprocess_parallel(image_bgr, lines)
    timings["preprocess_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    log.info(f"[2] Cropped {len(indexed_crops)}/{len(lines)} lines ({timings['preprocess_ms']} ms)")

    if not indexed_crops:
        return OCRResult(lines_detected=len(lines), lines_recognized=0, text="", timings_ms=timings)

    pil_images = [pil for _, pil in indexed_crops]
    valid_indices = [idx for idx, _ in indexed_crops]

    # 3. Recognition
    t0 = time.perf_counter()
    raw_texts = recognize_batch(pil_images)
    timings["recognition_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    log.info(f"[3] Recognized {len(raw_texts)} lines ({timings['recognition_ms']} ms)")

    # 4. Assemble final text in reading order
    slot_to_text = {valid_indices[i]: raw_texts[i] for i in range(len(raw_texts))}
    final_lines = [slot_to_text[i].strip() for i in sorted(slot_to_text) if slot_to_text[i].strip()]
    full_text = "\n".join(final_lines)

    timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
    log.info(f"Pipeline done — {timings['total_ms']} ms total")

    return OCRResult(
        lines_detected=len(lines),
        lines_recognized=len(raw_texts),
        text=full_text,
        timings_ms=timings,
    )


# ════════════════════════════════════════════════════════════════════
#  FASTAPI APPLICATION
# ════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="Handwritten OCR — Kraken + TrOCR",
    description="High‑accuracy pipeline: Kraken downscale + original‑res polygon crops",
    version="3.0.0",
    lifespan=lifespan,
)


@app.post("/ocr", summary="Extract text from a handwritten image")
async def ocr_endpoint(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    data = await file.read()
    np_arr = np.frombuffer(data, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=422, detail="Could not decode image.")

    # Run pipeline in thread pool to keep event loop free
    loop = asyncio.get_event_loop()
    result: OCRResult = await loop.run_in_executor(None, run_pipeline, img_bgr)

    # Optionally save to disk
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(result.text)

    return JSONResponse(content={
        "lines_detected":    result.lines_detected,
        "lines_recognized":  result.lines_recognized,
        "text":              result.text,
        "timings_ms":        result.timings_ms,
    })


@app.get("/health", summary="GPU/CPU health check")
async def health():
    info = {
        "status": "ok",
        "device": str(DEVICE),
        "trocr_model": CFG.trocr_model,
    }
    if DEVICE.type == "cuda":
        info["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
        info["vram_reserved_mb"]  = round(torch.cuda.memory_reserved() / 1e6, 1)
    return info


@app.get("/config", summary="Current pipeline configuration")
async def get_config():
    return {k: str(v) for k, v in CFG.__dict__.items()}


# ════════════════════════════════════════════════════════════════════
#  STANDALONE USAGE (without FastAPI)
# ════════════════════════════════════════════════════════════════════
def process_image_file(path: str) -> OCRResult:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return run_pipeline(img)


if __name__ == "__main__":
    import sys
    import uvicorn

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        print(f"\nProcessing: {img_path}")

        # Load models manually for CLI mode
        from kraken.lib.vgsl import TorchVGSLModel as _KrakenModel
        from transformers import TrOCRProcessor as _P, VisionEncoderDecoderModel as _M

        G.kraken_model    = _KrakenModel.load_model("blla.mlmodel")
        G.trocr_processor = _P.from_pretrained(CFG.trocr_model)
        G.trocr_model     = _M.from_pretrained(CFG.trocr_model)
        G.trocr_model.to(DEVICE).eval()
        if USE_FP16:
            G.trocr_model = G.trocr_model.half()
        G.preproc_pool = ProcessPoolExecutor(max_workers=CFG.preproc_workers)
        _warmup()

        result = process_image_file(img_path)
        print(f"\n{'─'*60}")
        print(f"Lines detected   : {result.lines_detected}")
        print(f"Lines recognized : {result.lines_recognized}")
        print(f"Timings          : {result.timings_ms}")
        print(f"{'─'*60}")
        print(result.text)

        G.preproc_pool.shutdown()

    else:
        uvicorn.run(
            "ocr_pipeline:app",
            host="0.0.0.0",
            port=8000,
            workers=1,          # must be 1 because GPU models are not fork-safe
            log_level="info",
        )