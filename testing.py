"""
╔══════════════════════════════════════════════════════════════════════╗
║     HANDWRITTEN OCR PIPELINE — Kraken blla + TrOCR-base             ║
║     Target  : RTX 2050 (4 GB VRAM) + CPU parallel preprocessing     ║
║     Goal    : >85% accuracy, <15s per page                           ║
║     Stack   : Kraken (detection) → TrOCR-base (recognition)         ║
╚══════════════════════════════════════════════════════════════════════╝

Architecture
─────────────────────────────────────────────────────────────────────
  ┌──────────────┐     ┌─────────────────────────────────────────┐
  │  FastAPI     │────▶│  Pipeline                               │
  │  /ocr  POST  │     │  ① Kraken blla  →  polygon lines        │
  └──────────────┘     │  ② CPU Pool     →  parallel crop+preproc│
                       │  ③ TrOCR-base   →  GPU batched inference│
                       │  ④ SpellCorrect →  post-process text    │
                       └─────────────────────────────────────────┘

Parallelism used
─────────────────────────────────────────────────────────────────────
  • Kraken detection    → GPU (single pass, whole page)
  • Image preprocessing → ProcessPoolExecutor (true multicore, no GIL)
  • TrOCR inference     → GPU batched with pinned memory + CUDA stream
  • FastAPI request     → asyncio.run_in_executor (non-blocking event loop)
  • Spell correction    → ThreadPoolExecutor (I/O-light, GIL-safe)
"""

from __future__ import annotations

# ── stdlib ──────────────────────────────────────────────────────────
import asyncio
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ── third-party ─────────────────────────────────────────────────────
import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Kraken
from kraken import blla
from kraken.lib import vgsl
from kraken.lib.vgsl import TorchVGSLModel


# Spell correction (pip install symspellpy)
try:
    from symspellpy import SymSpell, Verbosity
    SPELL_AVAILABLE = True
except ImportError:
    SPELL_AVAILABLE = False

# ── logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("ocr_pipeline")


# ════════════════════════════════════════════════════════════════════
#  CONFIGURATION  ← tweak everything here, nowhere else
# ════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    # ── Model ────────────────────────────────────────────────────────
    trocr_model: str = "microsoft/trocr-base-handwritten"

    # ── Inference ────────────────────────────────────────────────────
    max_new_tokens: int = 128
    beam_size: int = 4          # ↑ beams = ↑ accuracy, ↓ speed  (4 is sweet spot)
    batch_size_cuda: int = 12   # safe for 4 GB VRAM with base model
    batch_size_cpu: int = 4

    # ── Preprocessing ────────────────────────────────────────────────
    min_crop_h: int = 12        # discard noise boxes below this height
    min_crop_w: int = 24
    target_crop_h: int = 384    # TrOCR canonical input height
    clahe_clip: float = 2.5
    clahe_tile: Tuple[int, int] = (8, 8)
    denoise_h: int = 10

    # ── Reading order ────────────────────────────────────────────────
    row_tolerance: float = 0.55  # fraction of median line height for row grouping

    # ── Parallelism ──────────────────────────────────────────────────
    preproc_workers: int =2
    # max(1, (os.cpu_count() or 4) - 1)
    spell_workers: int = 2

    # ── Spell correction ─────────────────────────────────────────────
    spell_max_edit: int = 2
    spell_dict_path: str = ""   # leave empty to skip if dict not found


CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16 = DEVICE.type == "cuda"

log.info(f"Device: {DEVICE}  |  FP16: {USE_FP16}  |  Preproc workers: {CFG.preproc_workers}")


# ════════════════════════════════════════════════════════════════════
#  DATA TYPES
# ════════════════════════════════════════════════════════════════════
@dataclass
class TextLine:
    """One segmented handwritten line with its polygon and crop."""
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
    kraken_model: vgsl.TorchVGSLModel = None
    trocr_processor: TrOCRProcessor = None
    trocr_model: VisionEncoderDecoderModel = None
    cuda_stream: Optional[torch.cuda.Stream] = None
    preproc_pool: ProcessPoolExecutor = None
    spell_pool: ThreadPoolExecutor = None
    sym_spell: Optional[object] = None          # SymSpell instance


G = _Models()


# ════════════════════════════════════════════════════════════════════
#  STARTUP / SHUTDOWN
# ════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup, clean up on shutdown."""
    t0 = time.perf_counter()
    log.info("Loading models …")

    # ── Kraken segmentation model ──────────────────────────────────
    # blla.mlmodel is the default baseline segmentation model
    # shipped with kraken — trained specifically on handwritten docs.
    # G.kraken_model = vgsl.TorchVGSLModel.load_model(blla.mlmodel)
    G.kraken_model = TorchVGSLModel.load_model("blla.mlmodel")
    G.kraken_model.eval()
    if DEVICE.type == "cuda":
        G.kraken_model.nn.to(DEVICE)
    log.info("Kraken blla loaded ✓")

    # ── TrOCR ─────────────────────────────────────────────────────
    G.trocr_processor = TrOCRProcessor.from_pretrained(CFG.trocr_model)
    G.trocr_model = VisionEncoderDecoderModel.from_pretrained(CFG.trocr_model)
    G.trocr_model.to(DEVICE)
    G.trocr_model.eval()

    if USE_FP16:
        G.trocr_model = G.trocr_model.half()
        torch.backends.cudnn.benchmark = True

    # torch.compile gives ~15-25% throughput boost on PyTorch ≥ 2.0
    try:
        G.trocr_model = torch.compile(G.trocr_model, mode="reduce-overhead")
        log.info("torch.compile() enabled ✓")
    except Exception as e:
        log.warning(f"torch.compile() skipped: {e}")

    if DEVICE.type == "cuda":
        G.cuda_stream = torch.cuda.Stream()

    log.info("TrOCR loaded ✓")

    # ── Worker pools ───────────────────────────────────────────────
    G.preproc_pool = ProcessPoolExecutor(max_workers=CFG.preproc_workers)
    G.spell_pool   = ThreadPoolExecutor(max_workers=CFG.spell_workers)

    # ── SymSpell (optional spell corrector) ────────────────────────
    if SPELL_AVAILABLE:
        _load_symspell()

    # ── Warm-up ────────────────────────────────────────────────────
    _warmup()

    log.info(f"Pipeline ready in {(time.perf_counter()-t0)*1000:.0f} ms")
    yield

    # ── Cleanup ────────────────────────────────────────────────────
    G.preproc_pool.shutdown(wait=False)
    G.spell_pool.shutdown(wait=False)
    log.info("Shutdown complete.")


def _load_symspell():
    """Load SymSpell dictionary for post-OCR spell correction."""
    import pkg_resources
    dict_path = CFG.spell_dict_path or pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt"
    )
    if not os.path.exists(dict_path):
        log.warning(f"SymSpell dict not found at {dict_path} — skipping spell correction")
        return
    G.sym_spell = SymSpell(max_dictionary_edit_distance=CFG.spell_max_edit)
    G.sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
    log.info("SymSpell loaded ✓")


def _warmup():
    """
    Force CUDA kernel compilation before first real request.
    Without this, the first request takes 2-4x longer due to JIT.
    """
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
#  STEP 1 — DETECTION  (Kraken blla, GPU)
# ════════════════════════════════════════════════════════════════════
def detect_lines(image_pil: Image.Image) -> List[TextLine]:
    """
    Run Kraken's baseline segmentation on the full page.

    Returns TextLine objects sorted in reading order
    (top-to-bottom, left-to-right within each row).

    Kraken blla advantages over CRAFT for handwriting:
      • Predicts actual baseline curves → polygon crops follow real lines
      • Trained on handwritten Latin documents (not just scene text)
      • Handles wavy, slanted, and connected cursive correctly
    """
    seg = blla.segment(image_pil, model=G.kraken_model)

    lines: List[TextLine] = []
    for i, line in enumerate(seg.lines):
        if line.boundary is None:
            continue
        poly = np.array(line.boundary, dtype=np.int32)  # (N, 2)
        if poly.shape[0] < 3:
            continue
        lines.append(TextLine(index=i, polygon=poly))

    return _sort_reading_order(lines)


def _sort_reading_order(lines: List[TextLine]) -> List[TextLine]:
    """
    Group lines into rows by y-centroid proximity, sort x within rows.
    Produces correct top→bottom, left→right reading order.
    """
    if not lines:
        return lines

    y_centers = [np.mean(l.polygon[:, 1]) for l in lines]
    heights   = [np.ptp(l.polygon[:, 1]) for l in lines]   # y range per line

    median_h = float(np.median(heights)) if heights else 30.0
    tol = median_h * CFG.row_tolerance

    # Sort by y_center first
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
        # Sort by x-center of polygon within each row
        row_sorted = sorted(row, key=lambda p: np.mean(p[1].polygon[:, 0]))
        ordered.extend(l for _, l in row_sorted)

    return ordered


# ════════════════════════════════════════════════════════════════════
#  STEP 2 — CROP + PREPROCESS  (CPU, parallel via ProcessPoolExecutor)
# ════════════════════════════════════════════════════════════════════
def _preprocess_single(args: Tuple[np.ndarray, np.ndarray]) -> Optional[np.ndarray]:
    """
    Worker function — runs in a subprocess (no GIL, true parallel).

    Pipeline per crop:
      1. Polygon mask → tight bounding crop
      2. Upscale to target height (TrOCR trained at 384 px)
      3. Denoise with fast Non-Local Means
      4. CLAHE contrast enhancement
      5. Deskew
      6. Convert back to RGB for TrOCR

    Returns a uint8 RGB numpy array, or None to skip.
    """
    image_np, polygon = args

    # ── 1. Tight bounding rect of polygon ───────────────────────────
    x_min = max(0, int(polygon[:, 0].min()))
    x_max = min(image_np.shape[1], int(polygon[:, 0].max()))
    y_min = max(0, int(polygon[:, 1].min()))
    y_max = min(image_np.shape[0], int(polygon[:, 1].max()))

    h, w = y_max - y_min, x_max - x_min

    # Import cv2 inside worker (subprocess doesn't inherit parent imports cleanly)
    import cv2
    import numpy as np

    if h < 12 or w < 24:   # too small — noise
        return None

    crop = image_np[y_min:y_max, x_min:x_max].copy()

    # ── 2. Polygon mask — zero out pixels outside the actual line ───
    local_poly = polygon.copy()
    local_poly[:, 0] -= x_min
    local_poly[:, 1] -= y_min
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [local_poly.astype(np.int32)], 255)
    crop[mask == 0] = 255   # fill background white

    # ── 3. Upscale to target height ─────────────────────────────────
    target_h = 384
    if h < target_h:
        scale = target_h / h
        new_w  = max(1, int(w * scale))
        crop = cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    # ── 4. Grayscale + denoise ───────────────────────────────────────
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # ── 5. CLAHE ─────────────────────────────────────────────────────
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    # ── 6. Deskew ────────────────────────────────────────────────────
    gray = _deskew(gray)

    # ── 7. Back to RGB (TrOCR expects colour) ────────────────────────
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Correct minor rotation using minimum-area rect on dark pixels."""
    import cv2
    import numpy as np

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
    return cv2.warpAffine(gray, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def preprocess_parallel(
    image_np: np.ndarray,
    lines: List[TextLine],
) -> List[Tuple[int, Image.Image]]:
    """
    Submit all crops to the process pool simultaneously.
    Returns list of (original_index, PIL_image) for successful crops.

    Using ProcessPoolExecutor here (not Thread) because cv2 and numpy
    operations release the GIL inconsistently — processes guarantee
    true CPU parallelism.
    """
    args = [(image_np, line.polygon) for line in lines]

    # Submit all at once — zero waiting between submissions
    futures = {
        G.preproc_pool.submit(_preprocess_single, a): i
        for i, a in enumerate(args)
    }

    results: List[Tuple[int, Image.Image]] = []
    for future in as_completed(futures):
        idx = futures[future]
        try:
            rgb = future.result()
            if rgb is not None:
                pil = Image.fromarray(rgb)
                results.append((idx, pil))
        except Exception as e:
            log.warning(f"Preprocess failed for line {idx}: {e}")

    # Restore reading order (as_completed does not preserve submission order)
    results.sort(key=lambda x: x[0])
    return results


# ════════════════════════════════════════════════════════════════════
#  STEP 3 — RECOGNITION  (TrOCR, GPU batched)
# ════════════════════════════════════════════════════════════════════
def recognize_batch(pil_images: List[Image.Image]) -> List[str]:
    """
    Batched TrOCR inference with:
      • Pinned memory for fast host→GPU transfer
      • Dedicated CUDA stream (overlaps data transfer with compute)
      • FP16 to halve VRAM usage and speed up tensor cores
      • Beam search for accuracy (num_beams=4)

    Each batch is processed end-to-end before the next starts,
    keeping peak VRAM predictable on the 4 GB card.
    """
    batch_size = CFG.batch_size_cuda if DEVICE.type == "cuda" else CFG.batch_size_cpu
    all_texts: List[str] = []

    for i in range(0, len(pil_images), batch_size):
        batch = pil_images[i : i + batch_size]

        # Processor handles resizing + normalization
        inputs = G.trocr_processor(
            images=batch,
            return_tensors="pt",
            padding=True,
        )
        pixel_values = inputs.pixel_values  # (B, 3, H, W)  float32 on CPU

        if DEVICE.type == "cuda":
            # Pinned memory → async H2D transfer (overlaps with CPU work)
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

            # Must sync before decoding on CPU
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

        texts = G.trocr_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        all_texts.extend(texts)

    return all_texts


# ════════════════════════════════════════════════════════════════════
#  STEP 4 — SPELL CORRECTION  (optional, CPU threaded)
# ════════════════════════════════════════════════════════════════════
def _correct_word(word: str) -> str:
    """Correct a single word using SymSpell lookup."""
    if G.sym_spell is None or not word.strip():
        return word
    suggestions = G.sym_spell.lookup(
        word.lower(),
        Verbosity.CLOSEST,
        max_edit_distance=CFG.spell_max_edit,
    )
    if suggestions:
        # Preserve original capitalisation
        corrected = suggestions[0].term
        if word[0].isupper():
            corrected = corrected.capitalize()
        return corrected
    return word


def spell_correct_lines(lines: List[str]) -> List[str]:
    """
    Correct each line in parallel using a thread pool.
    Spell correction is I/O-light dict lookup — threads are fine here.
    """
    if G.sym_spell is None or not SPELL_AVAILABLE:
        return lines

    corrected: List[str] = []
    for line in lines:
        words = line.split()
        futures = [G.spell_pool.submit(_correct_word, w) for w in words]
        corrected_words = [f.result() for f in futures]
        corrected.append(" ".join(corrected_words))
    return corrected


# ════════════════════════════════════════════════════════════════════
#  FULL PIPELINE  (orchestrates all steps)
# ════════════════════════════════════════════════════════════════════
def run_pipeline(image_bgr: np.ndarray) -> OCRResult:
    """
    Full end-to-end pipeline:

       image_bgr  →  detect_lines  →  preprocess_parallel  →
       recognize_batch  →  spell_correct_lines  →  OCRResult

    Timing data is captured at each stage for diagnostics.
    """
    timings: dict = {}
    t_start = time.perf_counter()

    # ── Convert BGR → PIL RGB (Kraken requires PIL) ─────────────────
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    # ── STEP 1: Detection ───────────────────────────────────────────
    t0 = time.perf_counter()
    lines = detect_lines(image_pil)
    timings["detection_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    log.info(f"[1] Detected {len(lines)} lines  ({timings['detection_ms']} ms)")

    if not lines:
        return OCRResult(lines_detected=0, lines_recognized=0, text="", timings_ms=timings)

    # ── STEP 2: Parallel preprocessing ─────────────────────────────
    # Submit all crops to process pool simultaneously.
    # While processes run, the main thread waits — but crucially,
    # if detection and preprocessing overlap in future versions,
    # we gain additional concurrency here.
    t0 = time.perf_counter()
    indexed_crops = preprocess_parallel(image_bgr, lines)
    timings["preprocess_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    log.info(f"[2] Preprocessed {len(indexed_crops)}/{len(lines)} crops  ({timings['preprocess_ms']} ms)")

    if not indexed_crops:
        return OCRResult(lines_detected=len(lines), lines_recognized=0, text="", timings_ms=timings)

    pil_images = [pil for _, pil in indexed_crops]
    valid_indices = [idx for idx, _ in indexed_crops]

    # ── STEP 3: GPU Recognition ─────────────────────────────────────
    t0 = time.perf_counter()
    raw_texts = recognize_batch(pil_images)
    timings["recognition_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    log.info(f"[3] Recognized {len(raw_texts)} lines  ({timings['recognition_ms']} ms)")

    # ── STEP 4: Spell correction ────────────────────────────────────
    t0 = time.perf_counter()
    corrected_texts = spell_correct_lines(raw_texts)
    timings["spell_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    log.info(f"[4] Spell corrected  ({timings['spell_ms']} ms)")

    # ── Assemble output in reading order ────────────────────────────
    # valid_indices maps each recognized text back to its original line slot
    slot_to_text = {valid_indices[i]: corrected_texts[i] for i in range(len(corrected_texts))}
    final_lines  = [slot_to_text[i].strip() for i in sorted(slot_to_text) if slot_to_text[i].strip()]
    full_text    = "\n".join(final_lines)

    timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
    log.info(f"Pipeline done — {timings['total_ms']} ms total")

    return OCRResult(
        lines_detected=len(lines),
        lines_recognized=len(corrected_texts),
        text=full_text,
        timings_ms=timings,
    )


# ════════════════════════════════════════════════════════════════════
#  FASTAPI APPLICATION
# ════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="Handwritten OCR — Kraken + TrOCR",
    description="High-accuracy handwritten text recognition optimised for RTX 2050",
    version="2.0.0",
    lifespan=lifespan,
)


@app.post("/ocr", summary="Extract text from a handwritten image")
async def ocr_endpoint(file: UploadFile = File(...)):
    """
    Upload a handwritten image (JPG / PNG / BMP / TIFF).
    Returns detected lines, full text, and per-stage timings.
    """
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    data = await file.read()
    np_arr = np.frombuffer(data, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=422, detail="Could not decode image.")

    # Run blocking pipeline in a thread — keeps FastAPI event loop free
    # for concurrent requests while the GPU/CPU pipeline is running.
    loop = asyncio.get_event_loop()
    result: OCRResult = await loop.run_in_executor(None, run_pipeline, img_bgr)
    
    full_text = result.text

    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(full_text)
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
        "spell_correction": G.sym_spell is not None,
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
    """
    Convenience function for using the pipeline without FastAPI.

    Example:
        from ocr_pipeline import process_image_file
        result = process_image_file("answer_sheet.jpg")
        print(result.text)
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return run_pipeline(img)


if __name__ == "__main__":
    import sys
    import uvicorn

    if len(sys.argv) > 1:
        # CLI usage: python ocr_pipeline.py <image_path>
        img_path = sys.argv[1]
        print(f"\nProcessing: {img_path}")

        # Load models manually for CLI mode
        from kraken.lib import vgsl as _vgsl
        from transformers import TrOCRProcessor as _P, VisionEncoderDecoderModel as _M

        G.kraken_model    = _vgsl.TorchVGSLModel.load_model(blla.mlmodel)
        G.trocr_processor = _P.from_pretrained(CFG.trocr_model)
        G.trocr_model     = _M.from_pretrained(CFG.trocr_model)
        G.trocr_model.to(DEVICE).eval()
        if USE_FP16:
            G.trocr_model = G.trocr_model.half()
        G.preproc_pool = ProcessPoolExecutor(max_workers=CFG.preproc_workers)
        G.spell_pool   = ThreadPoolExecutor(max_workers=CFG.spell_workers)
        if SPELL_AVAILABLE:
            _load_symspell()
        _warmup()

        result = process_image_file(img_path)
        print(f"\n{'─'*60}")
        print(f"Lines detected   : {result.lines_detected}")
        print(f"Lines recognized : {result.lines_recognized}")
        print(f"Timings          : {result.timings_ms}")
        print(f"{'─'*60}")
        print(result.text)

        G.preproc_pool.shutdown()
        G.spell_pool.shutdown()

    else:
        # Server mode
        uvicorn.run(
            "ocr_pipeline:app",
            host="0.0.0.0",
            port=8000,
            workers=1,          # must be 1 — GPU models are not fork-safe
            log_level="info",
        )