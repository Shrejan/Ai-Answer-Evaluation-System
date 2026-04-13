from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
import torch
from PIL import Image
from kraken import binarization, blla
from kraken.lib.vgsl import TorchVGSLModel
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

app = FastAPI()

# -----------------------------
# LOAD MODELS ONCE (IMPORTANT)
# -----------------------------
kraken_model = TorchVGSLModel.load_model("blla.mlmodel")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

device = "cuda" if torch.cuda.is_available() else "cpu"
trocr_model.to(device)
trocr_model.eval()

print(f"Models loaded on {device}")

# -----------------------------
# HELPERS
# -----------------------------
def deskew_crop(crop, pts):
    rect = cv2.minAreaRect(pts)
    angle = rect[-1]

    if angle < -45:
        angle += 90

    (h, w) = crop.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        crop, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

def extract_line_crops(image_np, segmentation, padding=15):
    """
    Extract line crops with padding to improve OCR accuracy.
    padding: pixels to add around detected line boundaries (helps with wavy lines)
    """
    h, w = image_np.shape[:2]
    crops = []

    # sort lines top → bottom
    lines = sorted(segmentation.lines,
                   key=lambda l: np.mean([p[1] for p in l.boundary]))

    for line in lines:
        if not hasattr(line, "boundary") or line.boundary is None:
            continue

        pts = np.array(line.boundary, dtype=np.int32)

        # mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        # apply mask
        masked = cv2.bitwise_and(image_np, image_np, mask=mask)

        # get bounding box
        coords = np.column_stack(np.where(mask > 0))
        if coords.size == 0:
            continue

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # add padding around the detected line
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)

        crop = image_np[y_min:y_max, x_min:x_max]

        if crop.size > 0:
            crops.append(crop)

    return crops

def recognize_batch(crops, batch_size=8):
    texts = []

    for i in range(0, len(crops), batch_size):
        batch = crops[i:i + batch_size]

        pil_images = [
            Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
            for c in batch
        ]

        pixel_values = processor(images=pil_images, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            ids = trocr_model.generate(pixel_values)

        texts.extend(processor.batch_decode(ids, skip_special_tokens=True))

    return texts

# -----------------------------
# API ENDPOINT
# -----------------------------
@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):

    # read image
    data = await file.read()
    np_arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # grayscale only for kraken
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pil_gray = Image.fromarray(gray)

    # binarization
    bin_img = binarization.nlbin(pil_gray)

    # segmentation
    segmentation = blla.segment(bin_img, model=kraken_model)

    # extract crops with padding to improve accuracy on wavy lines
    crops = extract_line_crops(img, segmentation, padding=15)

    if not crops:
        return {"lines_detected": 0, "text": ""}

    # higher batch size for faster processing
    batch_size = 32 if device == "cuda" else 8

    texts = recognize_batch(crops, batch_size=batch_size)
    full_text = "\n".join(texts)

    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(full_text)

    return {
        "lines_detected": len(crops),
        "text": "\n".join(texts)
    }