import json
import numpy as np
import cv2
import random
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# ── load TrOCR model once ─────────────────────────────────────────────────────
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model     = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
device    = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print(f"Model loaded on {device}")

# ── load detections ───────────────────────────────────────────────────────────
with open("response_1773894419405.json", "r", encoding="utf-8") as f:
    data = json.load(f)

boxes = data["detections"][0]

# ── step 1: convert quads to bboxes ──────────────────────────────────────────
def quad_to_bbox(quad):
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return {
        "quad"    : quad,
        "x_min"   : min(xs),
        "y_min"   : min(ys),
        "x_max"   : max(xs),
        "y_max"   : max(ys),
        "y_center": (min(ys) + max(ys)) / 2,
    }

bboxes = [quad_to_bbox(q) for q in boxes]

# ── step 2: sort top → bottom ─────────────────────────────────────────────────
bboxes.sort(key=lambda b: b["y_center"])

# ── step 3: group into lines ──────────────────────────────────────────────────
def group_into_lines(bboxes, y_threshold=15):
    lines = []
    for bbox in bboxes:
        if not lines:
            lines.append([bbox])
            continue
        # use average y of current line to avoid drift on long lines
        avg_y = sum(b["y_center"] for b in lines[-1]) / len(lines[-1])
        if abs(bbox["y_center"] - avg_y) <= y_threshold:
            lines[-1].append(bbox)
        else:
            lines.append([bbox])
    # sort each line left → right
    for line in lines:
        line.sort(key=lambda b: b["x_min"])
    return lines

lines = group_into_lines(bboxes, y_threshold=15)

# ── step 4: merge line boxes + crop ──────────────────────────────────────────
def merge_line_bbox(line, pad=4):
    x_min = min(b["x_min"] for b in line)
    y_min = min(b["y_min"] for b in line)
    x_max = max(b["x_max"] for b in line)
    y_max = max(b["y_max"] for b in line)
    return x_min - pad, y_min - pad, x_max + pad, y_max + pad

def crop_line(image, x1, y1, x2, y2):
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return image[y1:y2, x1:x2]


image     = cv2.imread("processed.jpg")
debug_img = image.copy()
line_crops = []

for i, line in enumerate(lines):
    

    # merge + crop
    x1, y1, x2, y2 = merge_line_bbox(line, pad=4)
    
    crop = crop_line(image, x1, y1, x2, y2)
    if crop.size > 0:
        line_crops.append(crop)

cv2.imwrite("debug_lines.jpg", debug_img)
print(f"Lines detected : {len(lines)}")
print(f"Crops ready    : {len(line_crops)}")

# ── step 5: TrOCR recognition ─────────────────────────────────────────────────
def recognize_lines(line_crops: list, batch_size: int = 8) -> list[str]:
    """
    Input  : list of OpenCV BGR crop images (one per line)
    Output : list of recognized strings (one per line)
    """
    all_texts = []

    for i in range(0, len(line_crops), batch_size):
        batch = line_crops[i : i + batch_size]

        # OpenCV BGR → PIL RGB
        pil_images = [
            Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            for crop in batch
        ]

        pixel_values = processor(
            images=pil_images,
            return_tensors="pt"
        ).pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(pixel_values)

        texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        all_texts.extend(texts)

    return all_texts

# ── step 6: run + print results ───────────────────────────────────────────────
texts = recognize_lines(line_crops, batch_size=8)

print("\n── recognized text ──────────────────────────────────────────────────")
for i, text in enumerate(texts):
    print(f"Line {i:02d}: {text}")

full_text = "\n".join(texts)
print("\n── full extracted text ──────────────────────────────────────────────")
print(full_text)

# ── step 7: save result to file ───────────────────────────────────────────────
with open("extracted_text.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

print("\nSaved to extracted_text.txt")
print("Debug image saved to debug_lines.jpg")