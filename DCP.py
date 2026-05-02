import os
import time
import torch
import warnings
import numpy as np
import pandas as pd
import cv2

from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
from pathlib import Path

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════

INPUT_FOLDER    = "./full_pages"
LINE_OUTPUT_DIR = "./extracted_lines"
CSV_OUTPUT      = "./labels_to_review.csv"

KRAKEN_MAX_SIDE   = 1200
MIN_LINE_HEIGHT   = 18
MIN_LINE_WIDTH    = 80
CROP_PADDING_X    = 12
CROP_PADDING_Y    = 6
TROCR_LINE_HEIGHT = 64
TROCR_BATCH_SIZE  = 8      # raise to 16 if you have 8GB+ VRAM

os.makedirs(LINE_OUTPUT_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════
# INIT — GPU + models (all loaded before Stage 1 begins)
# ════════════════════════════════════════════════════════════════

def init_gpu() -> torch.device:
    """Crash immediately with a helpful message if CUDA is missing."""
    if not torch.cuda.is_available():
        raise EnvironmentError(
            "\n  ERROR: No CUDA GPU found.\n"
            "  Fix: pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu121\n"
        )
    device  = torch.device("cuda")
    name    = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  ✓ GPU   : {name}  ({vram_gb:.1f} GB VRAM)")
    return device


def init_kraken() -> tuple:
    """
    Load kraken blla segmenter + its default bundled model.

    Returns (segment_fn, seg_model) where seg_model may be None
    if the version doesn't require an explicit model argument.

    Version compatibility:
      v4+  : from kraken.blla import segment   (blla is a module)
      v3   : from kraken import blla; blla.segment(...)
    Both versions accept an optional model= kwarg; if the bundled
    default model path exists we load it explicitly so Kraken doesn't
    re-download it on every call.
    """
    # ── 1. Resolve the segment function ─────────────────────
    segment_fn = None
    for attempt in ["module", "submodule"]:
        try:
            if attempt == "module":
                from kraken.blla import segment
                segment_fn = segment
            else:
                from kraken import blla
                segment_fn = blla.segment
            break
        except (ImportError, AttributeError):
            continue

    if segment_fn is None:
        raise ImportError(
            "\n  ERROR: Could not import kraken.blla.segment.\n"
            "  Fix:\n"
            "    pip uninstall kraken -y\n"
            "    pip install --no-cache-dir kraken\n"
            "    python -c \"from kraken.blla import segment; print('OK')\"\n"
        )

    # ── 2. Load the bundled blla.mlmodel explicitly ──────────
    # Avoids repeated downloads and makes the model GPU-resident.
    seg_model = None
    try:
        from kraken.lib.vgsl import TorchVGSLModel
        import importlib.resources as pkg_resources
        # Kraken bundles blla.mlmodel inside its package directory
        try:
            model_path = str(pkg_resources.files("kraken").joinpath("blla.mlmodel"))
        except (TypeError, AttributeError):
            # older importlib.resources fallback
            import kraken as _kraken_pkg
            model_path = os.path.join(os.path.dirname(_kraken_pkg.__file__), "blla.mlmodel")

        if os.path.exists(model_path):
            seg_model = TorchVGSLModel.load_model(model_path)
            seg_model.eval()
            print(f"  ✓ Kraken model  : {model_path}")
        else:
            print("  ⚠ blla.mlmodel not found on disk — Kraken will use its default")
    except Exception as e:
        print(f"  ⚠ Could not pre-load blla model ({e}) — Kraken will auto-load")

    print(f"  ✓ Kraken blla   : {segment_fn}")
    return segment_fn, seg_model


def init_trocr(device: torch.device) -> tuple:
    """
    Load TrOCR with safetensors (bypasses torch.load CVE-2025-32434).
    Applies FP16 on GPU for ~2x faster inference.
    """
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    processor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-base-handwritten",
        use_fast=True,
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-base-handwritten",
        use_safetensors=True,       # avoids torch.load entirely
    )
    model = model.to(device).eval().half()  # FP16 on GPU

    print(f"  ✓ TrOCR (FP16)  : ready on {torch.cuda.get_device_name(0)}")
    return processor, model


# ════════════════════════════════════════════════════════════════
# IMAGE UTILITIES
# ════════════════════════════════════════════════════════════════

def load_original(page_path: str) -> Image.Image:
    return Image.open(page_path).convert("RGB")


def make_kraken_copy(original: Image.Image) -> tuple[Image.Image, float]:
    w, h = original.size
    longest = max(w, h)
    if longest <= KRAKEN_MAX_SIDE:
        return original.copy(), 1.0
    scale = longest / KRAKEN_MAX_SIDE
    return original.resize((int(w / scale), int(h / scale)), Image.LANCZOS), scale


def preprocess_for_kraken(image: Image.Image) -> Image.Image:
    img = image.convert("L")
    img = ImageEnhance.Contrast(img).enhance(1.8)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    return img.convert("RGB")


def scale_boundary_to_original(boundary: list, scale: float) -> list:
    return [(int(x * scale), int(y * scale)) for x, y in boundary]


def crop_line_from_original(original: Image.Image, boundary: list):
    if not boundary or len(boundary) < 3:
        return None
    xs = [p[0] for p in boundary]
    ys = [p[1] for p in boundary]
    x1 = max(0, min(xs) - CROP_PADDING_X)
    y1 = max(0, min(ys) - CROP_PADDING_Y)
    x2 = min(original.width,  max(xs) + CROP_PADDING_X)
    y2 = min(original.height, max(ys) + CROP_PADDING_Y)
    if (y2 - y1) < MIN_LINE_HEIGHT or (x2 - x1) < MIN_LINE_WIDTH:
        return None
    return original.crop((x1, y1, x2, y2))


# ════════════════════════════════════════════════════════════════
# LINE PROCESSING FOR TROCR
# ════════════════════════════════════════════════════════════════

def deskew_line(line_img: Image.Image) -> Image.Image:
    img_np = np.array(line_img.convert("L"))
    _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 50:
        return line_img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5 or abs(angle) > 15:
        return line_img
    return line_img.rotate(angle, expand=True, fillcolor=(255, 255, 255))


def remove_noise(line_img: Image.Image) -> Image.Image:
    img_np  = np.array(line_img.convert("L"))
    binary  = cv2.adaptiveThreshold(img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, blockSize=15, C=8)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return Image.fromarray(cleaned).convert("RGB")


def normalize_brightness(line_img: Image.Image) -> Image.Image:
    img_np  = np.array(line_img)
    lab     = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab     = cv2.merge((clahe.apply(l), a, b))
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def resize_for_trocr(line_img: Image.Image) -> Image.Image:
    w, h = line_img.size
    if h == 0 or w == 0:
        return line_img
    return line_img.resize((max(int(w * TROCR_LINE_HEIGHT / h), 32), TROCR_LINE_HEIGHT), Image.LANCZOS)


def process_line_for_trocr(raw_crop: Image.Image) -> Image.Image:
    img = raw_crop.convert("RGB")
    img = normalize_brightness(img)
    img = deskew_line(img)
    img = remove_noise(img)
    img = resize_for_trocr(img)
    img = ImageEnhance.Contrast(img).enhance(1.6)
    img = ImageEnhance.Sharpness(img).enhance(1.3)
    return img


# ════════════════════════════════════════════════════════════════
# STAGE 1 — KRAKEN LINE SEGMENTATION
# ════════════════════════════════════════════════════════════════

def segment_page(
    page_path: str,
    page_name: str,
    segment_fn,
    seg_model,          # TorchVGSLModel or None
) -> list[tuple[str, str]]:

    original            = load_original(page_path)
    kraken_input, scale = make_kraken_copy(original)
    kraken_input        = preprocess_for_kraken(kraken_input)

    # Pass model= only if we have one; otherwise let Kraken use its default
    kwargs = {"model": seg_model} if seg_model is not None else {}
    seg    = segment_fn(kraken_input, **kwargs)

    # Handles both modern Segmentation object and legacy dict
    lines     = seg.lines if hasattr(seg, "lines") else seg.get("lines", [])
    extracted = []

    for line_idx, line in enumerate(lines):
        try:
            boundary_scaled = (
                line.boundary if hasattr(line, "boundary") else line.get("boundary")
            )
            if not boundary_scaled or len(boundary_scaled) < 3:
                continue

            boundary_orig = scale_boundary_to_original(boundary_scaled, scale)
            raw_crop      = crop_line_from_original(original, boundary_orig)
            if raw_crop is None:
                continue

            processed     = process_line_for_trocr(raw_crop)
            line_filename = f"{page_name}_line{line_idx:03d}.png"
            line_path     = os.path.join(LINE_OUTPUT_DIR, line_filename)
            processed.save(line_path, format="PNG", optimize=True)
            extracted.append((line_path, line_filename))
        except Exception:
            continue

    return extracted


# ════════════════════════════════════════════════════════════════
# STAGE 2 — TROCR AUTO-LABELING (GPU batched)
# ════════════════════════════════════════════════════════════════

def predict_batch(
    image_paths: list[str],
    processor,
    model,
    device: torch.device,
) -> list[tuple[str, float]]:

    images = []
    for p in image_paths:
        try:
            images.append(Image.open(p).convert("RGB"))
        except Exception:
            images.append(Image.new("RGB", (256, 64), color=(255, 255, 255)))

    inputs       = processor(images, return_tensors="pt", padding=True)
    pixel_values = inputs.pixel_values.to(device=device, dtype=model.dtype)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            num_beams=4,
            max_length=128,
            output_scores=True,
            return_dict_in_generate=True,
        )

    texts = processor.batch_decode(outputs.sequences, skip_special_tokens=True)

    if outputs.scores:
        scores = torch.stack(outputs.scores, dim=1)
        probs  = torch.softmax(scores.float(), dim=-1)
        conf   = (probs.max(dim=-1).values.mean(dim=-1) * 100).tolist()
    else:
        conf = [0.0] * len(texts)

    return [(t.strip(), round(c, 1)) for t, c in zip(texts, conf)]


# ════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════

def run_pipeline():
    valid_ext   = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    page_images = sorted([
        f for f in Path(INPUT_FOLDER).iterdir()
        if f.suffix.lower() in valid_ext
    ])

    if not page_images:
        print(f"ERROR: No images found in '{INPUT_FOLDER}'")
        return

    print("=" * 60)
    print("  OCR PIPELINE — Kraken + TrOCR (GPU)")
    print("=" * 60)
    print(f"  Pages found      : {len(page_images)}")
    print(f"  Kraken max side  : {KRAKEN_MAX_SIDE}px")
    print(f"  TrOCR line height: {TROCR_LINE_HEIGHT}px")
    print(f"  TrOCR batch size : {TROCR_BATCH_SIZE}")
    print(f"  Line output dir  : {LINE_OUTPUT_DIR}")
    print(f"  CSV output       : {CSV_OUTPUT}")
    print("=" * 60)

    # ── INIT: all resources loaded before any page is touched ─
    print("\n[ INIT ]")
    device                 = init_gpu()
    segment_fn, seg_model  = init_kraken()
    processor, trocr_model = init_trocr(device)
    print()

    # ── STAGE 1 ───────────────────────────────────────────────
    print("[ STAGE 1 ] Kraken Line Segmentation\n")

    all_line_paths = []
    stage1_start   = time.time()
    failed_pages   = []

    for page_path in tqdm(page_images, desc="Pages"):
        page_name = page_path.stem
        try:
            lines = segment_page(str(page_path), page_name, segment_fn, seg_model)
            all_line_paths.extend(lines)
            tqdm.write(f"  ✓ {page_name}: {len(lines)} lines")
        except Exception as e:
            tqdm.write(f"  ✗ {page_name}: FAILED — {e}")
            failed_pages.append(page_name)

    stage1_time = time.time() - stage1_start
    total_lines = len(all_line_paths)

    print(f"\n  Stage 1 done in {stage1_time/60:.1f} min")
    print(f"  Lines extracted  : {total_lines}")
    print(f"  Pages failed     : {len(failed_pages)}")
    if failed_pages:
        print(f"  Failed pages     : {', '.join(failed_pages)}")

    if total_lines == 0:
        print("\nNo lines extracted. Check input images & Kraken install.")
        return

    # ── STAGE 2 ───────────────────────────────────────────────
    print("\n[ STAGE 2 ] TrOCR Auto-Labeling (GPU batched)\n")

    rows         = []
    stage2_start = time.time()
    all_paths    = [lp for lp, _ in all_line_paths]
    all_names    = [ln for _, ln in all_line_paths]

    for i in tqdm(range(0, total_lines, TROCR_BATCH_SIZE), desc="Batches", unit="batch"):
        batch_paths = all_paths[i : i + TROCR_BATCH_SIZE]
        batch_names = all_names[i : i + TROCR_BATCH_SIZE]
        try:
            results = predict_batch(batch_paths, processor, trocr_model, device)
        except Exception as e:
            results = [(f"[ERROR: {e}]", 0.0)] * len(batch_paths)

        for line_path, line_filename, (text, confidence) in zip(batch_paths, batch_names, results):
            rows.append({
                "image_path"    : line_path,
                "image_filename": line_filename,
                "page_name"     : "_".join(line_filename.split("_")[:-1]),
                "auto_label"    : text,
                "confidence_pct": confidence,
                "human_label"   : "",
                "reviewed"      : "No",
            })

    stage2_time = time.time() - stage2_start

    # ── STAGE 3 — Save CSV ────────────────────────────────────
    print("\n[ STAGE 3 ] Saving CSV\n")
    df = pd.DataFrame(rows)
    df = df.sort_values("confidence_pct", ascending=True).reset_index(drop=True)
    df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")

    # ── Summary ───────────────────────────────────────────────
    total_time = stage1_time + stage2_time
    avg_conf   = df["confidence_pct"].mean()
    high_conf  = len(df[df["confidence_pct"] >= 75])
    low_conf   = len(df[df["confidence_pct"] <  75])

    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Pages processed  : {len(page_images) - len(failed_pages)}")
    print(f"  Lines extracted  : {total_lines}")
    print(f"  Avg lines/page   : {total_lines / (len(page_images) or 1):.1f}")
    print(f"  Stage 1 time     : {stage1_time/60:.1f} min  (Kraken)")
    print(f"  Stage 2 time     : {stage2_time/60:.1f} min  (TrOCR GPU)")
    print(f"  Total time       : {total_time/60:.1f} min")
    print(f"\n  Avg confidence   : {avg_conf:.1f}%")
    print(f"  High conf (≥75%) : {high_conf} lines  ← likely correct")
    print(f"  Low conf  (<75%) : {low_conf}  lines  ← needs human review")
    print(f"\n  CSV saved to     : {CSV_OUTPUT}")
    print("=" * 60)
    print("\nNEXT STEPS:")
    print("  1. Open labels_to_review.csv in Excel / Google Sheets")
    print("  2. Low confidence rows are at the TOP — fix those first")
    print("  3. Fill 'human_label' column where auto_label is wrong")
    print("  4. Mark 'reviewed' = Yes when done with each row")
    print("  5. Run prepare_training_csv.py to build training dataset")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()