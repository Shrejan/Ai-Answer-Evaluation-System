from fastapi import FastAPI, UploadFile, File
import torch
from paddleocr import PaddleOCR
import cv2
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel



app = FastAPI()

ocr = PaddleOCR(use_angle_cls=False, lang="en")

# ── load TrOCR model once ─────────────────────────────────────────────────────
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model     = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
device    = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print(f"Model loaded on {device}")


@app.post("/ocr")                                         # FIX 1: put → post
async def run_ocr(file: UploadFile = File(...)):
    data = await file.read()
    
    np_arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    cv2.imwrite("processed.jpg", thresh)

    result = ocr.ocr(thresh, det=True, rec=False)

    boxes = result[0]

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
    bboxes.sort(key=lambda b: b["y_center"])

    def group_into_lines(bboxes, y_threshold=15):
        lines = []
        for bbox in bboxes:
            if not lines:
                lines.append([bbox])
                continue
            avg_y = sum(b["y_center"] for b in lines[-1]) / len(lines[-1])
            if abs(bbox["y_center"] - avg_y) <= y_threshold:
                lines[-1].append(bbox)
            else:
                lines.append([bbox])
        for line in lines:
            line.sort(key=lambda b: b["x_min"])
        return lines

    lines = group_into_lines(bboxes, y_threshold=15)

    def merge_line_bbox(line, pad=4):
        x_min = min(b["x_min"] for b in line)
        y_min = min(b["y_min"] for b in line)
        x_max = max(b["x_max"] for b in line)
        y_max = max(b["y_max"] for b in line)
        return x_min - pad, y_min - pad, x_max + pad, y_max + pad

    def crop_line(image, x1, y1, x2, y2):
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return image[y1:y2, x1:x2]

    # FIX 2: use thresh directly (converted to BGR) instead of re-reading from disk
    image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    line_crops = []

    for i, line in enumerate(lines):
        x1, y1, x2, y2 = merge_line_bbox(line, pad=4)
        crop = crop_line(image, x1, y1, x2, y2)
        if crop.size > 0:
            line_crops.append(crop)

    print(f"Lines detected : {len(lines)}")
    print(f"Crops ready    : {len(line_crops)}")

    def recognize_lines(line_crops: list, batch_size: int = 8) -> list[str]:
        all_texts = []
        for i in range(0, len(line_crops), batch_size):
            batch = line_crops[i : i + batch_size]
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

    texts = recognize_lines(line_crops, batch_size=8)
    full_text = "\n".join(texts)

    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(full_text)

    # FIX 3: return the actual extracted text
    return {
        "lines_detected": len(lines),
        "extracted_text": full_text
    }   