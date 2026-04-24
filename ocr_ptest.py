"""
Handwritten OCR Pipeline — Kraken blla + TrOCR-base
Optimised for 50 MP images (<10s per page) with dual‑resolution detection.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
cv2.setNumThreads(1)          # prevent CPU oversubscription

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from kraken import blla as kraken_blla
from kraken.lib.vgsl import TorchVGSLModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("ocr_pipeline")


# ═══════════════ CONFIG ═══════════════
@dataclass
class Config:
    trocr_model: str = "microsoft/trocr-base-handwritten"
    max_new_tokens: int = 128
    beam_size: int = 4
    batch_size_cuda: int = 20          # bumped for speed
    batch_size_cpu: int = 4
    preproc_workers: int = max(1, (os.cpu_count() or 4) - 1)
    row_tolerance: float = 0.55
    max_detection_dim: int = 1000      # smaller = faster detection on CPU


CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16 = DEVICE.type == "cuda"


# ═══════════════ DATA ═══════════════
@dataclass
class TextLine:
    index: int
    polygon: np.ndarray     # full‑res coords
    crop_pil: Optional[Image.Image] = None
    text: str = ""
    confidence: float = 0.0


@dataclass
class OCRResult:
    lines_detected: int
    lines_recognized: int
    text: str
    confidences: List[float] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)


# ═══════════════ GLOBAL MODELS ═══════════════
class _Models:
    kraken_model: TorchVGSLModel = None
    trocr_processor: TrOCRProcessor = None
    trocr_model: VisionEncoderDecoderModel = None
    cuda_stream: Optional[torch.cuda.Stream] = None
    preproc_pool: ThreadPoolExecutor = None


G = _Models()


# ═══════════════ LIFESPAN ═══════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()

    # Kraken – use your working loading method
    G.kraken_model = TorchVGSLModel.load_model("blla.mlmodel")
    G.kraken_model.eval()
    if DEVICE.type == "cuda":
        G.kraken_model.nn.to(DEVICE)

    # TrOCR
    G.trocr_processor = TrOCRProcessor.from_pretrained(CFG.trocr_model)
    
    G.trocr_model = VisionEncoderDecoderModel.from_pretrained(CFG.trocr_model)
    G.trocr_model.to(DEVICE)
    G.trocr_model.eval()

    if USE_FP16:
        G.trocr_model = G.trocr_model.half()
        torch.backends.cudnn.benchmark = True

    # torch.compile removed – it caused recompilation on first real batch

    if DEVICE.type == "cuda":
        G.cuda_stream = torch.cuda.Stream()

    G.preproc_pool = ThreadPoolExecutor(max_workers=CFG.preproc_workers)
    _warmup()
    log.info(f"Kraken is on {next(G.kraken_model.nn.parameters()).device}")
    log.info(f"TrOCR is on {next(G.trocr_model.parameters()).device}")
    log.info("Pipeline ready — Kraken + TrOCR on %s (%.0f ms)", DEVICE, (time.perf_counter() - t0) * 1000)
    yield
    G.preproc_pool.shutdown(wait=False)
    log.info("Shutdown complete.")


def _warmup():
    dummy = Image.fromarray(np.full((384, 512, 3), 200, dtype=np.uint8))
    inputs = G.trocr_processor(images=[dummy], return_tensors="pt")
    pv = inputs.pixel_values.to(DEVICE)
    if USE_FP16:
        pv = pv.half()
    with torch.inference_mode():
        G.trocr_model.generate(pv, max_new_tokens=4)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


# ═══════════════ STEP 1 — DETECTION (on downscaled image) ═══════════════
def detect_lines(image_pil: Image.Image) -> List[TextLine]:
    """
    Kraken segmentation on the (small) image.
    Returns TextLine objects with polygon coords in the small image's system.
    """
    try:
        seg = kraken_blla.segment(image_pil, model=G.kraken_model)
    except Exception as e:
        log.error("Detection failed: %s", e)
        return []

    lines = []
    for i, line in enumerate(seg.lines):
        if line.boundary is None:
            continue
        poly = np.array(line.boundary, dtype=np.int32)
        if poly.shape[0] < 3:
            continue
        lines.append(TextLine(index=i, polygon=poly))
    return _sort_reading_order(lines)


def _sort_reading_order(lines: List[TextLine]) -> List[TextLine]:
    if not lines:
        return lines
    y_centers = [np.mean(l.polygon[:, 1]) for l in lines]
    heights = [np.ptp(l.polygon[:, 1]) for l in lines]
    median_h = float(np.median(heights)) if heights else 30.0
    tol = median_h * CFG.row_tolerance

    paired = sorted(zip(y_centers, lines), key=lambda p: p[0])
    rows = []
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

    ordered = []
    for row in rows:
        row_sorted = sorted(row, key=lambda p: np.mean(p[1].polygon[:, 0]))
        ordered.extend(l for _, l in row_sorted)
    return ordered


# ═══════════════ STEP 2 — CROP + PREPROCESS (full‑res image) ═══════════════
def _preprocess_single(args: Tuple[np.ndarray, np.ndarray]) -> Optional[np.ndarray]:
    image_np, polygon = args
    x_min = max(0, int(polygon[:, 0].min()))
    x_max = min(image_np.shape[1], int(polygon[:, 0].max()))
    y_min = max(0, int(polygon[:, 1].min()))
    y_max = min(image_np.shape[0], int(polygon[:, 1].max()))
    h, w = y_max - y_min, x_max - x_min

    if h < 12 or w < 24:
        return None

    crop = image_np[y_min:y_max, x_min:x_max].copy()

    local_poly = polygon.copy()
    local_poly[:, 0] -= x_min
    local_poly[:, 1] -= y_min
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [local_poly.astype(np.int32)], 255)
    crop[mask == 0] = 255

    target_h = 384
    if h > target_h:
        scale = target_h / h
        new_w = max(1, int(w * scale))
        crop = cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = _deskew(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def _deskew(gray: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(gray < 128))
    if len(coords) < 20:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess_parallel(full_res_bgr: np.ndarray, lines: List[TextLine]) -> List[Tuple[int, Image.Image]]:
    args = [(full_res_bgr, line.polygon) for line in lines]
    futures = {G.preproc_pool.submit(_preprocess_single, a): i for i, a in enumerate(args)}
    results = []
    for f in as_completed(futures):
        idx = futures[f]
        rgb = f.result()
        if rgb is not None:
            results.append((idx, Image.fromarray(rgb)))
    results.sort(key=lambda x: x[0])
    return results


# ═══════════════ STEP 3 — RECOGNITION + CONFIDENCE ═══════════════
def recognize_batch(pil_images: List[Image.Image]) -> Tuple[List[str], List[float]]:
    batch_size = CFG.batch_size_cuda if DEVICE.type == "cuda" else CFG.batch_size_cpu
    all_texts = []
    all_confidences = []

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
                    output = G.trocr_model.generate(
                        pixel_values,
                        max_new_tokens=CFG.max_new_tokens,
                        num_beams=CFG.beam_size,
                        early_stopping=True,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
            G.cuda_stream.synchronize()
        else:
            pixel_values = pixel_values.to(DEVICE)
            with torch.inference_mode():
                output = G.trocr_model.generate(
                    pixel_values,
                    max_new_tokens=CFG.max_new_tokens,
                    num_beams=CFG.beam_size,
                    early_stopping=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

        sequences = output.sequences
        scores = output.scores
        token_logprobs = torch.stack([torch.nn.functional.log_softmax(s, dim=-1) for s in scores], dim=1)
        chosen_logprobs = torch.gather(token_logprobs, -1, sequences[:, 1:].unsqueeze(-1)).squeeze(-1)
        avg_logprob = chosen_logprobs.sum(dim=1) / chosen_logprobs.size(1)
        confidences = torch.exp(avg_logprob).cpu().tolist()

        texts = G.trocr_processor.batch_decode(sequences, skip_special_tokens=True)
        all_texts.extend(texts)
        all_confidences.extend(confidences)

    return all_texts, all_confidences


# ═══════════════ FULL PIPELINE ═══════════════
def run_pipeline(image_bgr: np.ndarray) -> OCRResult:
    timings = {}
    t_start = time.perf_counter()

    full_res = image_bgr
    h, w = full_res.shape[:2]
    max_dim = CFG.max_detection_dim
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)

    # Downscaled copy for detection
    if scale < 1.0:
        small_img = cv2.resize(full_res, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small_img = full_res

    image_rgb = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    # Detection
    t0 = time.perf_counter()
    lines = detect_lines(image_pil)
    timings["detection_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    log.info("Detection: %d lines in %s ms", len(lines), timings["detection_ms"])

    if not lines:
        return OCRResult(lines_detected=0, lines_recognized=0, text="", timings_ms=timings)

    # Scale polygons back to full‑res
    if scale != 1.0:
        inv_scale = 1.0 / scale
        for line in lines:
            line.polygon = (line.polygon * inv_scale).astype(np.int32)

    # Preprocessing on full‑res
    t0 = time.perf_counter()
    indexed_crops = preprocess_parallel(full_res, lines)
    timings["preprocess_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    log.info("Preprocessing: %d/%d lines in %s ms", len(indexed_crops), len(lines), timings["preprocess_ms"])

    if not indexed_crops:
        return OCRResult(lines_detected=len(lines), lines_recognized=0, text="", timings_ms=timings)

    pil_images = [pil for _, pil in indexed_crops]
    valid_indices = [idx for idx, _ in indexed_crops]

    # Recognition
    t0 = time.perf_counter()
    raw_texts, confidences = recognize_batch(pil_images)
    timings["recognition_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    log.info("Recognition: %d lines in %s ms", len(raw_texts), timings["recognition_ms"])

    # Assemble final text (reading order preserved)
    slot_to_text = {}
    slot_to_conf = {}
    for idx, text, conf in zip(valid_indices, raw_texts, confidences):
        slot_to_text[idx] = text.strip()
        slot_to_conf[idx] = round(conf, 3)

    final_lines = []
    confs = []
    for idx in sorted(slot_to_text):
        if slot_to_text[idx]:
            final_lines.append(slot_to_text[idx])
            confs.append(slot_to_conf[idx])

    full_text = "\n".join(final_lines)
    timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
    log.info("Total: %s ms", timings["total_ms"])

    return OCRResult(
        lines_detected=len(lines),
        lines_recognized=len(final_lines),
        text=full_text,
        confidences=confs,
        timings_ms=timings,
    )


# ═══════════════ FASTAPI ═══════════════
app = FastAPI(
    title="Handwritten OCR — Kraken + TrOCR",
    description="High-accuracy handwritten text recognition optimised for 50 MP images",
    version="2.2.1",
    lifespan=lifespan,
)


@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    data = await file.read()
    np_arr = np.frombuffer(data, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=422, detail="Could not decode image.")

    loop = asyncio.get_event_loop()
    result: OCRResult = await loop.run_in_executor(None, run_pipeline, img_bgr)
    full_text = result.text

    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(full_text)
    return JSONResponse({
        "lines_detected": result.lines_detected,
        "lines_recognized": result.lines_recognized,
        "text": result.text,
        "confidences": result.confidences,
        "timings_ms": result.timings_ms,
    })


@app.get("/health")
async def health():
    info = {"status": "ok", "device": str(DEVICE), "trocr_model": CFG.trocr_model}
    if DEVICE.type == "cuda":
        info["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
        info["vram_reserved_mb"] = round(torch.cuda.memory_reserved() / 1e6, 1)
    return info


@app.get("/config")
async def get_config():
    return {k: str(v) for k, v in CFG.__dict__.items()}


# ═══════════════ CLI (standalone) ═══════════════
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
        G.kraken_model = TorchVGSLModel.load_model("blla.mlmodel")
        G.kraken_model.eval()
        if DEVICE.type == "cuda":
            G.kraken_model.nn.to(DEVICE)
        G.trocr_processor = TrOCRProcessor.from_pretrained(CFG.trocr_model)
        G.trocr_model = VisionEncoderDecoderModel.from_pretrained(CFG.trocr_model)
        G.trocr_model.to(DEVICE).eval()
        if USE_FP16:
            G.trocr_model = G.trocr_model.half()
            torch.backends.cudnn.benchmark = True
        if DEVICE.type == "cuda":
            G.cuda_stream = torch.cuda.Stream()
        G.preproc_pool = ThreadPoolExecutor(max_workers=CFG.preproc_workers)
        _warmup()

        result = process_image_file(img_path)
        print(f"\nLines detected: {result.lines_detected}")
        print(f"Lines recognized: {result.lines_recognized}")
        print(f"Timings (ms): {result.timings_ms}")
        print(f"Confidences: {result.confidences}")
        print(f"\n{result.text}")

        G.preproc_pool.shutdown()
    else:
        uvicorn.run("ocr_pipeline:app", host="0.0.0.0", port=8000, workers=1, log_level="info")