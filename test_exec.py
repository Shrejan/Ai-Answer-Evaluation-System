# ==============================================================================
# CONFIGURATION CELL: All benchmark settings in one central place
# ==============================================================================
import os
import random
import numpy as np

# API Service Configuration
API_URL = "http://localhost:8001"
ENDPOINT_PATH = "/ocr"
HEALTH_PATH = "/"
REQUEST_MODE = "multipart"  # Options: "multipart" or "base64_json"
FILE_FIELD = "file"
TIMEOUT_S = 60
MAX_RETRIES = 3
NUM_WORKERS = 4

# Execution Flags
QUICK_TEST = True       # Set to True for quick sanity testing (N=50 per set)
INCLUDE_PAGE_SET = False # Exclude page dataset; run single and line benchmarks only
RUN_ABLATION = True      # Run ablation comparison (raw vs clean)

# Benchmark Dataset Sample Sizes
N_SINGLE_SAMPLES = 50 if QUICK_TEST else 10000
N_LINE_SAMPLES = 50 if QUICK_TEST else 2000
N_PAGE_SAMPLES = 20 if QUICK_TEST else 300

# Synthetic Image Generation Settings
SEED = 42
MIN_DIGITS = 3
MAX_DIGITS = 10
DIGIT_SCALE = 1.5
SPACING_RANGE = (2, 12)
VERTICAL_JITTER = 4
NOISE_LEVEL = 0.0

# Output Paths
OUTPUT_DIR = "./results"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set global deterministic random seeds
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
except ImportError:
    pass

print(f"Configuration initialized. Outputs will be saved to: {OUTPUT_DIR}")
print(f"Sample Sizes -> Single: {N_SINGLE_SAMPLES}, Line: {N_LINE_SAMPLES}, Page: {N_PAGE_SAMPLES}")


# --- CELL ---

# %pip install requests torchvision numpy pandas pillow scikit-learn matplotlib seaborn tqdm jiwer rapidfuzz

from typing import Union, List, Dict, Tuple, Optional, Any
import requests
import torchvision
import numpy as np
import pandas as pd
import PIL
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import json
import time
import re
import io
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

try:
    import jiwer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False

print("=== Library Versions ===")
print(f"Requests     : {requests.__version__}")
print(f"Torchvision  : {torchvision.__version__}")
print(f"NumPy        : {np.__version__}")
print(f"Pandas       : {pd.__version__}")
print(f"Pillow (PIL) : {PIL.__version__}")
print(f"Scikit-Learn : {sklearn.__version__}")
print(f"Matplotlib   : {matplotlib.__version__}")
print(f"Seaborn      : {sns.__version__}")
print(f"Tqdm         : {tqdm.__version__}")
print(f"JiWER        : {'Available' if HAS_JIWER else 'Not Installed (Using Dynamic Programming Fallback)'}")


# --- CELL ---

from PIL import Image, ImageFilter
from torchvision.datasets import MNIST

# Load MNIST test dataset only
mnist_test = MNIST(root='./data', train=False, download=True)
print(f"Successfully loaded MNIST Test Set: {len(mnist_test)} samples.")

# Verification: Ensure no training split overlap
assert len(mnist_test) == 10000, "Unexpected MNIST test set size!"


def make_single_digit(pil_img: Image.Image, scale: float = 1.5, pad: int = 10) -> Image.Image:
    """Invert MNIST digit (dark on white bg), upscale with bicubic, add white margin."""
    np_img = 255 - np.array(pil_img)
    inv_pil = Image.fromarray(np_img)
    w, h = inv_pil.size
    new_w, new_h = int(w * scale), int(h * scale)
    scaled = inv_pil.resize((new_w, new_h), Image.Resampling.BICUBIC)
    
    padded = Image.new('L', (new_w + 2 * pad, new_h + 2 * pad), 255)
    padded.paste(scaled, (pad, pad))
    return padded


def make_line(
    dataset,
    min_digits: int = MIN_DIGITS,
    max_digits: int = MAX_DIGITS,
    scale: float = DIGIT_SCALE,
    spacing_range: Tuple[int, int] = SPACING_RANGE,
    jitter: int = VERTICAL_JITTER,
    pad: int = 15
) -> Tuple[Image.Image, str, int]:
    """Generate single-line synthetic multi-digit image with random gaps and vertical jitter."""
    n_digits = random.randint(min_digits, max_digits)
    indices = [random.randint(0, len(dataset) - 1) for _ in range(n_digits)]
    digits_imgs = []
    labels = []
    
    for idx in indices:
        img, lbl = dataset[idx]
        np_img = 255 - np.array(img)
        inv_pil = Image.fromarray(np_img)
        w, h = inv_pil.size
        scaled = inv_pil.resize((int(w * scale), int(h * scale)), Image.Resampling.BICUBIC)
        digits_imgs.append(scaled)
        labels.append(str(lbl))
        
    spacings = [random.randint(spacing_range[0], spacing_range[1]) for _ in range(n_digits - 1)]
    total_digit_w = sum(img.width for img in digits_imgs)
    total_spacing_w = sum(spacings)
    max_h = max(img.height for img in digits_imgs)
    
    canvas_w = total_digit_w + total_spacing_w + 2 * pad
    canvas_h = max_h + 2 * jitter + 2 * pad
    
    canvas = Image.new('L', (canvas_w, canvas_h), 255)
    
    curr_x = pad
    for i, d_img in enumerate(digits_imgs):
        j_y = random.randint(-jitter, jitter)
        curr_y = pad + jitter + j_y
        canvas.paste(d_img, (curr_x, curr_y))
        curr_x += d_img.width
        if i < len(spacings):
            curr_x += spacings[i]
            
    gt_label = ''.join(labels)
    assert len(gt_label) == n_digits, f"Label length mismatch! Got {len(gt_label)}, expected {n_digits}"
    return canvas, gt_label, n_digits


def make_page(
    dataset,
    min_lines: int = 3,
    max_lines: int = 5,
    line_spacing_range: Tuple[int, int] = (15, 30),
    pad: int = 20
) -> Tuple[Image.Image, List[str], List[List[int]]]:
    """Generate multi-line document image with line ground truth text and bounding boxes [x1, y1, x2, y2]."""
    n_lines = random.randint(min_lines, max_lines)
    lines_info = [make_line(dataset) for _ in range(n_lines)]
    
    max_w = max(line_img.width for line_img, _, _ in lines_info)
    line_spacings = [random.randint(line_spacing_range[0], line_spacing_range[1]) for _ in range(n_lines - 1)]
    total_h = sum(line_img.height for line_img, _, _ in lines_info) + sum(line_spacings) + 2 * pad
    page_w = max_w + 2 * pad
    
    page = Image.new('L', (page_w, total_h), 255)
    
    gt_texts = []
    boxes = []
    
    curr_y = pad
    for i, (line_img, gt_text, _) in enumerate(lines_info):
        curr_x = pad
        page.paste(line_img, (curr_x, curr_y))
        box = [curr_x, curr_y, curr_x + line_img.width, curr_y + line_img.height]
        boxes.append(box)
        gt_texts.append(gt_text)
        curr_y += line_img.height
        if i < len(line_spacings):
            curr_y += line_spacings[i]
            
    return page, gt_texts, boxes

print("Data generation functions defined successfully.")


# --- CELL ---

# Generate Single, Line, and Page Benchmark Sets
def build_benchmark_datasets():
    # 1. Single Set
    single_dir = os.path.join(IMAGES_DIR, "single")
    os.makedirs(single_dir, exist_ok=True)
    single_records = []
    
    print(f"Generating SINGLE digit dataset ({N_SINGLE_SAMPLES} samples)...")
    for i in tqdm.tqdm(range(N_SINGLE_SAMPLES)):
        img, lbl = mnist_test[i % len(mnist_test)]
        single_img = make_single_digit(img)
        fname = f"single_{i:05d}.png"
        fpath = os.path.join(single_dir, fname)
        single_img.save(fpath)
        single_records.append({
            "file_name": fname,
            "text": str(lbl),
            "num_digits": 1,
            "boxes": json.dumps([[0, 0, single_img.width, single_img.height]])
        })
    df_single_labels = pd.DataFrame(single_records)
    df_single_labels.to_csv(os.path.join(OUTPUT_DIR, "single_labels.csv"), index=False)
    
    # 2. Line Set
    line_dir = os.path.join(IMAGES_DIR, "line")
    os.makedirs(line_dir, exist_ok=True)
    line_records = []
    
    print(f"Generating LINE dataset ({N_LINE_SAMPLES} samples)...")
    for i in tqdm.tqdm(range(N_LINE_SAMPLES)):
        line_img, gt_text, n_digits = make_line(mnist_test)
        fname = f"line_{i:05d}.png"
        fpath = os.path.join(line_dir, fname)
        line_img.save(fpath)
        line_records.append({
            "file_name": fname,
            "text": gt_text,
            "num_digits": n_digits,
            "boxes": json.dumps([[0, 0, line_img.width, line_img.height]])
        })
    df_line_labels = pd.DataFrame(line_records)
    df_line_labels.to_csv(os.path.join(OUTPUT_DIR, "line_labels.csv"), index=False)
    
    # 3. Page Set
    df_page_labels = pd.DataFrame()
    if INCLUDE_PAGE_SET:
        page_dir = os.path.join(IMAGES_DIR, "page")
        os.makedirs(page_dir, exist_ok=True)
        page_records = []
        
        print(f"Generating PAGE dataset ({N_PAGE_SAMPLES} samples)...")
        for i in tqdm.tqdm(range(N_PAGE_SAMPLES)):
            page_img, gt_texts, gt_boxes = make_page(mnist_test)
            fname = f"page_{i:05d}.png"
            fpath = os.path.join(page_dir, fname)
            page_img.save(fpath)
            total_digits = sum(len(t) for t in gt_texts)
            page_records.append({
                "file_name": fname,
                "text": "\n".join(gt_texts),
                "num_digits": total_digits,
                "boxes": json.dumps(gt_boxes)
            })
        df_page_labels = pd.DataFrame(page_records)
        df_page_labels.to_csv(os.path.join(OUTPUT_DIR, "page_labels.csv"), index=False)

    return df_single_labels, df_line_labels, df_page_labels

df_single_labels, df_line_labels, df_page_labels = build_benchmark_datasets()

# Assert Quality Checks
assert len(df_single_labels) == N_SINGLE_SAMPLES, "Single set size mismatch!"
assert len(df_line_labels) == N_LINE_SAMPLES, "Line set size mismatch!"
if INCLUDE_PAGE_SET:
    assert len(df_page_labels) == N_PAGE_SAMPLES, "Page set size mismatch!"

print("[OK] All synthetic benchmark datasets created and verified successfully.")


# --- CELL ---

# Visualize sample grid of generated benchmark images
fig, axes = plt.subplots(3, 4, figsize=(14, 8))

for idx in range(4):
    # Single sample
    s_row = df_single_labels.iloc[idx]
    s_img = Image.open(os.path.join(IMAGES_DIR, "single", s_row["file_name"]))
    axes[0, idx].imshow(s_img, cmap="gray")
    axes[0, idx].set_title(f"Single GT: '{s_row['text']}'", fontsize=10, fontweight="bold")
    axes[0, idx].axis("off")

    # Line sample
    l_row = df_line_labels.iloc[idx]
    l_img = Image.open(os.path.join(IMAGES_DIR, "line", l_row["file_name"]))
    axes[1, idx].imshow(l_img, cmap="gray")
    axes[1, idx].set_title(f"Line GT: '{l_row['text']}'", fontsize=10, fontweight="bold")
    axes[1, idx].axis("off")

    # Page sample
    if INCLUDE_PAGE_SET and not df_page_labels.empty:
        p_row = df_page_labels.iloc[idx]
        p_img = Image.open(os.path.join(IMAGES_DIR, "page", p_row["file_name"]))
        axes[2, idx].imshow(p_img, cmap="gray")
        p_label_short = p_row['text'].replace('\n', ' | ')
        axes[2, idx].set_title(f"Page GT: '{p_label_short}'", fontsize=9, fontweight="bold")
        axes[2, idx].axis("off")

plt.tight_layout()
plt.suptitle("Synthetic Benchmark Sample Grid (MNIST Test Derived)", y=1.02, fontsize=14, fontweight="bold")
plt.savefig(os.path.join(FIGURES_DIR, "sample_grid.png"), dpi=200, bbox_inches="tight")
# plt.show()


# --- CELL ---

def health_check() -> bool:
    """Pings the target OCR service health endpoint."""
    url = f"{API_URL.rstrip('/')}{HEALTH_PATH}"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            print(f"[OK] Health Check Passed: OCR Service at '{API_URL}' is reachable (HTTP 200).")
            return True
        else:
            print(f"[WARNING] Health Check Warning: Server returned HTTP {resp.status_code} at '{url}'.")
            return False
    except Exception as e:
        print(f"[ERROR] Health Check Failed: Cannot connect to OCR Service at '{url}'. Details: {e}")
        print("Note: Benchmark execution will record server connection errors gracefully without crashing.")
        return False

SERVER_ONLINE = health_check()


def parse_response(resp_json: Union[dict, list, str]) -> dict:
    """
    Central Response Parser: Extracts text, bounding boxes, and confidence.
    Customize this function to fit your specific OCR server response schema!
    
    Supported JSON formats:
    1. {"extracted_text": "123\n456", "lines_detected": 2}
    2. {"text": "123"} or {"result": "123"}
    3. {"lines": [{"text": "123", "bbox": [x1, y1, x2, y2], "confidence": 0.98}, ...]}
    """
    raw_text = ""
    boxes = []
    confidence = None

    if isinstance(resp_json, str):
        raw_text = resp_json
    elif isinstance(resp_json, dict):
        # Format 1 & 2: Direct text key search
        for key in ["text", "extracted_text", "reconstructed", "content", "result", "prediction"]:
            if key in resp_json and resp_json[key] is not None:
                raw_text = str(resp_json[key])
                break

        # Format 3: Detailed line structures with bounding boxes & confidence
        if "lines" in resp_json and isinstance(resp_json["lines"], list):
            line_texts = []
            line_confs = []
            for item in resp_json["lines"]:
                if isinstance(item, dict):
                    if "text" in item:
                        line_texts.append(str(item["text"]))
                    if "bbox" in item or "box" in item:
                        boxes.append(item.get("bbox") or item.get("box"))
                    if "confidence" in item:
                        line_confs.append(float(item["confidence"]))
            if line_texts and not raw_text:
                raw_text = "\n".join(line_texts)
            if line_confs:
                confidence = float(np.mean(line_confs))

        if confidence is None and "confidence" in resp_json:
            try:
                confidence = float(resp_json["confidence"])
            except (ValueError, TypeError):
                confidence = None
    elif isinstance(resp_json, list):
        raw_text = "\n".join(str(item) for item in resp_json)

    # Post-process: Clean digits-only string
    pred_clean = re.sub(r"[^0-9]", "", raw_text)

    return {
        "pred_raw": raw_text,
        "pred_clean": pred_clean,
        "boxes": boxes,
        "confidence": confidence,
        "raw_json": resp_json
    }


def call_ocr_api(image_path: str) -> dict:
    """Sends image file to OCR API with retries, backoff, and timing."""
    url = f"{API_URL.rstrip('/')}{ENDPOINT_PATH}"
    
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    last_error = None
    start_t = time.perf_counter()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if REQUEST_MODE == "multipart":
                files = {FILE_FIELD: (os.path.basename(image_path), img_bytes, "image/png")}
                response = requests.post(url, files=files, timeout=TIMEOUT_S)
            else:
                b64_str = base64.b64encode(img_bytes).decode("utf-8")
                payload = {"image": b64_str}
                response = requests.post(url, json=payload, timeout=TIMEOUT_S)

            latency_ms = (time.perf_counter() - start_t) * 1000.0

            if response.status_code == 200:
                try:
                    resp_json = response.json()
                    parsed = parse_response(resp_json)
                    parsed["latency_ms"] = latency_ms
                    parsed["error"] = None
                    return parsed
                except Exception as parse_err:
                    return {
                        "pred_raw": "", "pred_clean": "", "boxes": [],
                        "confidence": None, "latency_ms": latency_ms,
                        "error": f"JSON Parse Error: {parse_err}"
                    }
            else:
                last_error = f"HTTP {response.status_code}: {response.text[:100]}"
        except requests.exceptions.Timeout:
            last_error = f"Timeout after {TIMEOUT_S}s"
        except requests.exceptions.ConnectionError:
            last_error = "Connection Error (Server Unreachable)"
        except Exception as err:
            last_error = f"Request Error: {err}"

        time.sleep(0.5 * attempt)

    latency_ms = (time.perf_counter() - start_t) * 1000.0
    return {
        "pred_raw": "", "pred_clean": "", "boxes": [],
        "confidence": None, "latency_ms": latency_ms,
        "error": last_error or "Unknown Failure"
    }

print("API Gateway caller and response parser initialized.")


# --- CELL ---

# Warm-up request & JSON structure verification
print("Running warm-up OCR request...")
sample_single_path = os.path.join(IMAGES_DIR, "single", df_single_labels.iloc[0]["file_name"])
warmup_result = call_ocr_api(sample_single_path)

print("\n--- Warm-up Request Result ---")
print(f"Latency   : {warmup_result['latency_ms']:.2f} ms")
print(f"Error     : {warmup_result['error']}")
print(f"Pred Raw  : '{warmup_result['pred_raw']}'")
print(f"Pred Clean: '{warmup_result['pred_clean']}'")
print(f"Raw JSON Response: {json.dumps(warmup_result['raw_json']) if 'raw_json' in warmup_result else 'N/A'}")


# --- CELL ---

def run_benchmark_on_dataset(dataset_name: str, df_labels: pd.DataFrame) -> pd.DataFrame:
    if df_labels.empty:
        return pd.DataFrame()

    csv_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_results.csv")
    img_folder = os.path.join(IMAGES_DIR, dataset_name)

    # Resume capability: Check if existing CSV contains processed records
    existing_df = pd.DataFrame()
    processed_files = set()
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            processed_files = set(existing_df["file"].dropna().unique())
            print(f"Resuming {dataset_name}: {len(processed_files)}/{len(df_labels)} samples already completed.")
        except Exception:
            existing_df = pd.DataFrame()

    records_to_run = df_labels[~df_labels["file_name"].isin(processed_files)]
    if records_to_run.empty and not existing_df.empty:
        print(f"[OK] All {len(df_labels)} samples for {dataset_name} are already processed!")
        return existing_df

    cols = ["file", "ground_truth", "num_digits", "boxes_gt", "pred_raw", "pred_clean", "confidence", "boxes_pred", "latency_ms", "error"]
    csv_lock = threading.Lock()
    
    def process_item(row):
        fname = row["file_name"]
        fpath = os.path.join(img_folder, fname)
        api_res = call_ocr_api(fpath)
        return {
            "file": fname,
            "ground_truth": row["text"],
            "num_digits": row["num_digits"],
            "boxes_gt": row["boxes"],
            "pred_raw": api_res["pred_raw"],
            "pred_clean": api_res["pred_clean"],
            "confidence": api_res["confidence"],
            "boxes_pred": json.dumps(api_res["boxes"]),
            "latency_ms": api_res["latency_ms"],
            "error": api_res["error"]
        }

    print(f"Executing concurrent inference on '{dataset_name}' set with {NUM_WORKERS} workers...")
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_item, row): row["file_name"] for _, row in records_to_run.iterrows()}
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc=f"Inference [{dataset_name}]"):
            try:
                res = future.result()
            except Exception as e:
                fname = futures[future]
                res = {
                    "file": fname, "ground_truth": "", "num_digits": 0, "boxes_gt": "[]",
                    "pred_raw": "", "pred_clean": "", "confidence": None, "boxes_pred": "[]",
                    "latency_ms": 0.0, "error": str(e)
                }
            
            # Real-time CSV update as each request completes
            res_df = pd.DataFrame([res], columns=cols)
            with csv_lock:
                write_header = not (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0)
                res_df.to_csv(csv_path, mode='a', header=write_header, index=False)

    final_df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
    print(f"Saved {len(final_df)} records to '{csv_path}'.")
    return final_df


df_res_single = run_benchmark_on_dataset("single", df_single_labels)
df_res_line = run_benchmark_on_dataset("line", df_line_labels)
df_res_page = run_benchmark_on_dataset("page", df_page_labels) if INCLUDE_PAGE_SET else pd.DataFrame()


# --- CELL ---

def compute_edit_distance_and_alignment(gt: str, pred: str):
    """Dynamic Programming Edit Distance and Character-Level Alignment."""
    m, n = len(gt), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if gt[i-1] == pred[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # Deletion
                dp[i][j-1] + 1,      # Insertion
                dp[i-1][j-1] + cost  # Substitution / Match
            )
            
    i, j = m, n
    aligned_pairs = []
    substitutions, deletions, insertions = 0, 0, 0
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (0 if gt[i-1] == pred[j-1] else 1):
            if gt[i-1] != pred[j-1]:
                substitutions += 1
            aligned_pairs.append((gt[i-1], pred[j-1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            deletions += 1
            aligned_pairs.append((gt[i-1], '<DEL>'))
            i -= 1
        else:
            insertions += 1
            aligned_pairs.append(('<INS>', pred[j-1]))
            j -= 1
            
    aligned_pairs.reverse()
    edit_dist = dp[m][n]
    cer = edit_dist / max(1, m)
    return cer, edit_dist, substitutions, deletions, insertions, aligned_pairs


def evaluate_recognition_dataset(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}

    total_gt_chars = 0
    total_edit_dist = 0
    total_sub, total_del, total_ins = 0, 0, 0
    exact_matches = 0
    aligned_all = []
    
    cer_list = []
    seq_acc_list = []
    non_digit_count = 0
    empty_count = 0
    failure_count = 0

    for _, row in df.iterrows():
        gt = str(row["ground_truth"]) if pd.notna(row["ground_truth"]) else ""
        pred_clean = str(row["pred_clean"]) if pd.notna(row["pred_clean"]) else ""
        pred_raw = str(row["pred_raw"]) if pd.notna(row["pred_raw"]) else ""
        err = row.get("error")

        if pd.notna(err) and err:
            failure_count += 1

        if not pred_clean:
            empty_count += 1

        if re.search(r"[^0-9]", pred_raw):
            non_digit_count += 1

        cer, ed, s, d, i_cnt, pairs = compute_edit_distance_and_alignment(gt, pred_clean)
        
        total_gt_chars += len(gt)
        total_edit_dist += ed
        total_sub += s
        total_del += d
        total_ins += i_cnt
        aligned_all.extend(pairs)
        
        is_exact = (gt == pred_clean)
        if is_exact:
            exact_matches += 1
            
        cer_list.append(cer)
        seq_acc_list.append(1.0 if is_exact else 0.0)

    n_samples = len(df)
    overall_cer = total_edit_dist / max(1, total_gt_chars)
    seq_acc = exact_matches / max(1, n_samples)
    char_acc = max(0.0, 1.0 - overall_cer)
    
    # Latency stats
    valid_latencies = df["latency_ms"].dropna()
    mean_lat = float(valid_latencies.mean()) if not valid_latencies.empty else 0.0
    median_lat = float(valid_latencies.median()) if not valid_latencies.empty else 0.0
    p95_lat = float(np.percentile(valid_latencies, 95)) if not valid_latencies.empty else 0.0
    throughput = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    return {
        "n_samples": n_samples,
        "overall_cer": overall_cer,
        "seq_accuracy": seq_acc,
        "char_accuracy": char_acc,
        "substitutions": total_sub,
        "deletions": total_del,
        "insertions": total_ins,
        "aligned_pairs": aligned_all,
        "cer_list": cer_list,
        "seq_acc_list": seq_acc_list,
        "pct_non_digit": (non_digit_count / max(1, n_samples)) * 100.0,
        "pct_empty": (empty_count / max(1, n_samples)) * 100.0,
        "pct_failures": (failure_count / max(1, n_samples)) * 100.0,
        "mean_latency_ms": mean_lat,
        "median_latency_ms": median_lat,
        "p95_latency_ms": p95_lat,
        "throughput_fps": throughput
    }


metrics_single = evaluate_recognition_dataset(df_res_single)
metrics_line = evaluate_recognition_dataset(df_res_line)
metrics_page = evaluate_recognition_dataset(df_res_page) if not df_res_page.empty else {}

print("Recognition metrics evaluated successfully.")


# --- CELL ---

# Detection & End-to-End Metrics (PAGE set)
def compute_box_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def evaluate_page_detection(df_page: pd.DataFrame, iou_thresh: float = 0.5):
    if df_page.empty:
        return {}

    total_tp, total_fp, total_fn = 0, 0, 0
    all_matched_ious = []
    line_count_errors = []

    for _, row in df_page.iterrows():
        try:
            gt_boxes = json.loads(row["boxes_gt"]) if pd.notna(row["boxes_gt"]) else []
            pred_boxes = json.loads(row["boxes_pred"]) if pd.notna(row["boxes_pred"]) else []
        except Exception:
            gt_boxes, pred_boxes = [], []

        line_count_errors.append(len(pred_boxes) - len(gt_boxes))

        if not gt_boxes and not pred_boxes:
            continue

        iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
        for i, g in enumerate(gt_boxes):
            for j, p in enumerate(pred_boxes):
                iou_matrix[i, j] = compute_box_iou(g, p)

        matched_gt, matched_pred = set(), set()
        if iou_matrix.size > 0:
            indices = np.unravel_index(np.argsort(-iou_matrix, axis=None), iou_matrix.shape)
            for i, j in zip(*indices):
                if iou_matrix[i, j] < iou_thresh:
                    break
                if i not in matched_gt and j not in matched_pred:
                    matched_gt.add(i)
                    matched_pred.add(j)
                    all_matched_ious.append(iou_matrix[i, j])

        tp = len(matched_gt)
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn

    prec = total_tp / max(1, total_tp + total_fp)
    rec = total_tp / max(1, total_tp + total_fn)
    f1 = 2 * prec * rec / max(1e-6, prec + rec)
    mean_iou = float(np.mean(all_matched_ious)) if all_matched_ious else 0.0

    return {
        "precision": prec, "recall": rec, "f1": f1,
        "mean_iou": mean_iou,
        "mean_line_count_error": float(np.mean(line_count_errors)) if line_count_errors else 0.0
    }

detection_metrics = evaluate_page_detection(df_res_page) if not df_res_page.empty else {}
print(f"Page Line Detection Metrics @ IoU 0.5: {detection_metrics}")


# --- CELL ---

# Figure 1: Confusion Matrices (Single & Line Sets)
from sklearn.metrics import confusion_matrix

def plot_confusion_matrices():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Single digit confusion matrix
    if not df_res_single.empty:
        gt_single = df_res_single["ground_truth"].astype(str).tolist()
        pred_single = df_res_single["pred_clean"].astype(str).tolist()
        labels_digits = [str(i) for i in range(10)]
        cm_single = confusion_matrix(gt_single, pred_single, labels=labels_digits, normalize="true")
        sns.heatmap(cm_single, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels_digits, yticklabels=labels_digits, ax=axes[0])
        axes[0].set_title("Single Digit Confusion Matrix (Normalized)", fontweight="bold")
        axes[0].set_xlabel("Predicted Digit")
        axes[0].set_ylabel("True Digit")

    # Line set aligned confusion matrix
    if metrics_line.get("aligned_pairs"):
        pairs = [(g, p) for g, p in metrics_line["aligned_pairs"] if g in "0123456789" and p in "0123456789"]
        if pairs:
            gt_line, pred_line = zip(*pairs)
            cm_line = confusion_matrix(gt_line, pred_line, labels=labels_digits, normalize="true")
            sns.heatmap(cm_line, annot=True, fmt=".2f", cmap="Greens", xticklabels=labels_digits, yticklabels=labels_digits, ax=axes[1])
            axes[1].set_title("Line Aligned Digit Confusion Matrix (Normalized)", fontweight="bold")
            axes[1].set_xlabel("Predicted Digit")
            axes[1].set_ylabel("True Digit")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "01_confusion_matrix.png"), dpi=200)
    # plt.show()

plot_confusion_matrices()


# --- CELL ---

# Figure 2: CER Histogram / KDE
plt.figure(figsize=(8, 5))
if metrics_line.get("cer_list"):
    sns.histplot(metrics_line["cer_list"], kde=True, color="#2563EB", bins=20)
    plt.axvline(metrics_line["overall_cer"], color="red", linestyle="--", label=f"Mean CER: {metrics_line['overall_cer']:.3f}")
    plt.title("Character Error Rate (CER) Distribution - Line Set", fontweight="bold")
    plt.xlabel("CER")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(os.path.join(FIGURES_DIR, "02_cer_histogram.png"), dpi=200)
    # plt.show()


# --- CELL ---

# Figure 3: Accuracy and CER vs Line Length
if not df_res_line.empty:
    df_res_line["cer"] = metrics_line["cer_list"]
    df_res_line["exact_match"] = metrics_line["seq_acc_list"]
    grouped = df_res_line.groupby("num_digits").agg(
        mean_cer=("cer", "mean"),
        acc=("exact_match", "mean")
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    sns.barplot(data=grouped, x="num_digits", y="acc", ax=ax1, color="#3B82F6", alpha=0.7)
    sns.lineplot(data=grouped, x=ax1.get_xticks(), y="mean_cer", ax=ax2, color="#EF4444", marker="o", linewidth=2.5)

    ax1.set_xlabel("Number of Digits in Line", fontweight="bold")
    ax1.set_ylabel("Exact Match Accuracy", color="#3B82F6", fontweight="bold")
    ax2.set_ylabel("Mean CER", color="#EF4444", fontweight="bold")
    plt.title("Accuracy and CER vs. Line Length", fontweight="bold")
    plt.savefig(os.path.join(FIGURES_DIR, "03_acc_cer_vs_length.png"), dpi=200)
    # plt.show()


# --- CELL ---

# Figure 4: Per-Position Accuracy Bar Chart
def plot_per_position_accuracy():
    pos_matches = {}
    pos_counts = {}

    for _, row in df_res_line.iterrows():
        gt = str(row["ground_truth"])
        pred = str(row["pred_clean"])
        for idx in range(len(gt)):
            pos = idx + 1
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
            if idx < len(pred) and gt[idx] == pred[idx]:
                pos_matches[pos] = pos_matches.get(pos, 0) + 1

    positions = sorted(pos_counts.keys())
    accs = [pos_matches.get(p, 0) / pos_counts[p] for p in positions]

    plt.figure(figsize=(9, 5))
    sns.barplot(x=positions, y=accs, palette="viridis")
    plt.title("Per-Position Character Accuracy (1 to N)", fontweight="bold")
    plt.xlabel("Digit Position in Line")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(os.path.join(FIGURES_DIR, "04_per_position_acc.png"), dpi=200)
    # plt.show()

plot_per_position_accuracy()


# --- CELL ---

# Figure 5: Per-Class Precision/Recall/F1 Bar Chart
from sklearn.metrics import classification_report

def plot_per_class_metrics():
    if df_res_single.empty:
        return
    gt = df_res_single["ground_truth"].astype(str)
    pred = df_res_single["pred_clean"].astype(str)
    report = classification_report(gt, pred, labels=[str(i) for i in range(10)], output_dict=True, zero_division=0)
    
    classes = [str(i) for i in range(10)]
    prec = [report[c]["precision"] for c in classes]
    rec = [report[c]["recall"] for c in classes]
    f1 = [report[c]["f1-score"] for c in classes]

    df_cls = pd.DataFrame({"Class": classes, "Precision": prec, "Recall": rec, "F1-Score": f1})
    df_melt = df_cls.melt(id_vars="Class", var_name="Metric", value_name="Score")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_melt, x="Class", y="Score", hue="Metric", palette="Set2")
    plt.title("Per-Class (0-9) Classification Performance (SINGLE Set)", fontweight="bold")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(os.path.join(FIGURES_DIR, "05_per_class_metrics.png"), dpi=200)
    # plt.show()

plot_per_class_metrics()


# --- CELL ---

# Figure 6: Error Type Breakdown Stacked Bar Chart
labels_sets = ["SINGLE", "LINE"]
sub_counts = [metrics_single.get("substitutions", 0), metrics_line.get("substitutions", 0)]
del_counts = [metrics_single.get("deletions", 0), metrics_line.get("deletions", 0)]
ins_counts = [metrics_single.get("insertions", 0), metrics_line.get("insertions", 0)]

plt.figure(figsize=(8, 5))
p1 = plt.bar(labels_sets, sub_counts, label="Substitutions", color="#EAB308")
p2 = plt.bar(labels_sets, del_counts, bottom=sub_counts, label="Deletions", color="#EF4444")
bottom_ins = [s + d for s, d in zip(sub_counts, del_counts)]
p3 = plt.bar(labels_sets, ins_counts, bottom=bottom_ins, label="Insertions", color="#3B82F6")

plt.title("Error Type Breakdown Across Evaluation Sets", fontweight="bold")
plt.ylabel("Total Error Count")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.savefig(os.path.join(FIGURES_DIR, "06_error_breakdown.png"), dpi=200)
# plt.show()


# --- CELL ---

# Figure 7: Detection Precision-Recall vs IoU Threshold (PAGE Set)
if INCLUDE_PAGE_SET and not df_res_page.empty:
    iou_thresholds = np.arange(0.3, 0.95, 0.1)
    prec_list, rec_list, f1_list = [], [], []

    for thresh in iou_thresholds:
        res = evaluate_page_detection(df_res_page, iou_thresh=thresh)
        prec_list.append(res.get("precision", 0.0))
        rec_list.append(res.get("recall", 0.0))
        f1_list.append(res.get("f1", 0.0))

    plt.figure(figsize=(8, 5))
    plt.plot(iou_thresholds, prec_list, marker="o", label="Precision", color="#2563EB", linewidth=2)
    plt.plot(iou_thresholds, rec_list, marker="s", label="Recall", color="#16A34A", linewidth=2)
    plt.plot(iou_thresholds, f1_list, marker="^", label="F1-Score", color="#DC2626", linewidth=2)
    plt.title("Line Detection Metrics vs. IoU Threshold (PAGE Set)", fontweight="bold")
    plt.xlabel("IoU Threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIGURES_DIR, "07_detection_pr_vs_iou.png"), dpi=200)
    # plt.show()
else:
    print("PAGE set box detection metrics skipped (boxes not provided by API or set disabled).")


# --- CELL ---

# Figure 8: Confidence Distribution & Reliability Diagram
confs = df_res_line["confidence"].dropna() if not df_res_line.empty else pd.Series()
if not confs.empty:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    correct_confs = df_res_line[df_res_line["ground_truth"] == df_res_line["pred_clean"]]["confidence"].dropna()
    wrong_confs = df_res_line[df_res_line["ground_truth"] != df_res_line["pred_clean"]]["confidence"].dropna()

    sns.kdeplot(correct_confs, ax=axes[0], color="green", label="Correct Predictions", fill=True)
    sns.kdeplot(wrong_confs, ax=axes[0], color="red", label="Wrong Predictions", fill=True)
    axes[0].set_title("Confidence Distribution (Correct vs Wrong)", fontweight="bold")
    axes[0].set_xlabel("Confidence Score")
    axes[0].legend()

    # Reliability diagram
    bins = np.linspace(0, 1, 11)
    df_res_line["conf_bin"] = pd.cut(df_res_line["confidence"], bins=bins)
    rel = df_res_line.groupby("conf_bin").agg(
        acc=("exact_match", "mean"),
        conf=("confidence", "mean")
    ).dropna()
    
    axes[1].plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    axes[1].bar(rel["conf"], rel["acc"], width=0.08, alpha=0.7, color="#2563EB", label="Model Calibration")
    axes[1].set_title("Reliability Diagram (ECE)", fontweight="bold")
    axes[1].set_xlabel("Mean Confidence")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "08_confidence_calibration.png"), dpi=200)
    # plt.show()
else:
    print("Confidence calibration plot skipped (API did not return confidence scores).")


# --- CELL ---

# Figure 9: Latency Distribution Histogram
all_latencies = pd.concat([
    df_res_single.get("latency_ms", pd.Series()),
    df_res_line.get("latency_ms", pd.Series())
]).dropna()

plt.figure(figsize=(8, 5))
sns.histplot(all_latencies, kde=True, color="#8B5CF6", bins=30)
mean_l = all_latencies.mean() if not all_latencies.empty else 0
median_l = all_latencies.median() if not all_latencies.empty else 0
p95_l = np.percentile(all_latencies, 95) if not all_latencies.empty else 0

plt.axvline(mean_l, color="red", linestyle="--", label=f"Mean: {mean_l:.1f} ms")
plt.axvline(median_l, color="green", linestyle="-.", label=f"Median: {median_l:.1f} ms")
plt.axvline(p95_l, color="orange", linestyle=":", label=f"P95: {p95_l:.1f} ms")

plt.title("Round-Trip API Latency Distribution", fontweight="bold")
plt.xlabel("Latency (ms)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.savefig(os.path.join(FIGURES_DIR, "09_latency_histogram.png"), dpi=200)
# plt.show()


# --- CELL ---

# Figure 10: Grid of the 20 Worst Predictions
df_errors = df_res_line[df_res_line["ground_truth"] != df_res_line["pred_clean"]].head(20) if not df_res_line.empty else pd.DataFrame()

if not df_errors.empty:
    n_show = len(df_errors)
    cols = 4
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 2.5))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, (_, row) in enumerate(df_errors.iterrows()):
        img_p = os.path.join(IMAGES_DIR, "line", row["file"])
        img = Image.open(img_p)
        axes[idx].imshow(img, cmap="gray")
        axes[idx].set_title(f"GT: '{row['ground_truth']}'\nPred: '{row['pred_clean']}'", fontsize=9, color="red")
        axes[idx].axis("off")

    for idx in range(n_show, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Top 20 Worst Prediction Failures (LINE Set)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "10_worst_predictions.png"), dpi=200, bbox_inches="tight")
    # plt.show()


# --- CELL ---

# Figure 11: Top 10 Most Common Digit Confusions
if metrics_line.get("aligned_pairs"):
    confusions = {}
    for g, p in metrics_line["aligned_pairs"]:
        if g != p:
            key = f"'{g}' -> '{p}'"
            confusions[key] = confusions.get(key, 0) + 1

    sorted_conf = sorted(confusions.items(), key=lambda x: x[1], reverse=True)[:10]
    if sorted_conf:
        keys, counts = zip(*sorted_conf)
        plt.figure(figsize=(9, 5))
        sns.barplot(x=list(counts), y=list(keys), palette="Reds_r")
        plt.title("Top 10 Digit Confusions (True -> Predicted)", fontweight="bold")
        plt.xlabel("Frequency")
        plt.ylabel("Confusion Pair")
        plt.grid(axis="x", alpha=0.3)
        plt.savefig(os.path.join(FIGURES_DIR, "11_top_confusions.png"), dpi=200)
        # plt.show()


# --- CELL ---

# Build Styled Summary Table
summary_data = [
    {
        "Set": "SINGLE",
        "Samples": metrics_single.get("n_samples", 0),
        "CER": f"{metrics_single.get('overall_cer', 0.0):.4f}",
        "Seq Accuracy": f"{metrics_single.get('seq_accuracy', 0.0) * 100:.2f}%",
        "Char Accuracy": f"{metrics_single.get('char_accuracy', 0.0) * 100:.2f}%",
        "Mean Latency (ms)": f"{metrics_single.get('mean_latency_ms', 0.0):.1f}",
        "P95 Latency (ms)": f"{metrics_single.get('p95_latency_ms', 0.0):.1f}",
        "Throughput (img/s)": f"{metrics_single.get('throughput_fps', 0.0):.1f}",
        "Failures": f"{metrics_single.get('pct_failures', 0.0):.1f}%"
    },
    {
        "Set": "LINE",
        "Samples": metrics_line.get("n_samples", 0),
        "CER": f"{metrics_line.get('overall_cer', 0.0):.4f}",
        "Seq Accuracy": f"{metrics_line.get('seq_accuracy', 0.0) * 100:.2f}%",
        "Char Accuracy": f"{metrics_line.get('char_accuracy', 0.0) * 100:.2f}%",
        "Mean Latency (ms)": f"{metrics_line.get('mean_latency_ms', 0.0):.1f}",
        "P95 Latency (ms)": f"{metrics_line.get('p95_latency_ms', 0.0):.1f}",
        "Throughput (img/s)": f"{metrics_line.get('throughput_fps', 0.0):.1f}",
        "Failures": f"{metrics_line.get('pct_failures', 0.0):.1f}%"
    }
]

if INCLUDE_PAGE_SET and metrics_page:
    summary_data.append({
        "Set": "PAGE",
        "Samples": metrics_page.get("n_samples", 0),
        "CER": f"{metrics_page.get('overall_cer', 0.0):.4f}",
        "Seq Accuracy": f"{metrics_page.get('seq_accuracy', 0.0) * 100:.2f}%",
        "Char Accuracy": f"{metrics_page.get('char_accuracy', 0.0) * 100:.2f}%",
        "Mean Latency (ms)": f"{metrics_page.get('mean_latency_ms', 0.0):.1f}",
        "P95 Latency (ms)": f"{metrics_page.get('p95_latency_ms', 0.0):.1f}",
        "Throughput (img/s)": f"{metrics_page.get('throughput_fps', 0.0):.1f}",
        "Failures": f"{metrics_page.get('pct_failures', 0.0):.1f}%"
    })

df_summary = pd.DataFrame(summary_data)
df_summary.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
    json.dump(summary_data, f, indent=2)

print("=== BENCHMARK SUMMARY TABLE ===")
print(df_summary)


# --- CELL ---

if RUN_ABLATION and not df_res_line.empty:
    print("Running Ablation Comparison: Raw Output vs Digit-Cleaned Output...")
    
    raw_cers = [compute_edit_distance_and_alignment(str(row["ground_truth"]), str(row["pred_raw"]))[0] for _, row in df_res_line.iterrows()]
    clean_cers = metrics_line["cer_list"]

    df_ablation = pd.DataFrame([
        {"Decoding Strategy": "Raw Output", "Mean CER": float(np.mean(raw_cers)), "Exact Match": float(np.mean([g == r for g, r in zip(df_res_line["ground_truth"], df_res_line["pred_raw"])]))},
        {"Decoding Strategy": "Digit-Cleaned", "Mean CER": float(np.mean(clean_cers)), "Exact Match": float(metrics_line["seq_accuracy"])}
    ])

    df_ablation.to_csv(os.path.join(OUTPUT_DIR, "ablation_results.csv"), index=False)
    
    plt.figure(figsize=(7, 4))
    sns.barplot(data=df_ablation, x="Decoding Strategy", y="Mean CER", palette="Set1")
    plt.title("Ablation Analysis: Raw vs Digit-Filtered Decoding", fontweight="bold")
    plt.ylabel("Mean Character Error Rate (CER)")
    plt.savefig(os.path.join(FIGURES_DIR, "09_ablation_comparison.png"), dpi=200)
    # plt.show()

    print(df_ablation)
