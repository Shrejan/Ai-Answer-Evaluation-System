from kraken import binarization, blla
from kraken.lib.vgsl import TorchVGSLModel
from PIL import Image
import numpy as np
import cv2
import os

# -----------------------------
# Setup
# -----------------------------
os.makedirs("lines", exist_ok=True)

# 1. Load ORIGINAL image (color)
orig_image = Image.open("test2.jpg").convert("RGB")
orig_np = np.array(orig_image)

# 2. Create grayscale ONLY for Kraken
gray_image = orig_image.convert("L")
bin_img = binarization.nlbin(gray_image)

# 3. Load model
model = TorchVGSLModel.load_model("blla.mlmodel")

# 4. Segment (on grayscale)
segmentation = blla.segment(bin_img, model=model)

# 5. For visualization
cv_img = orig_np.copy()
h, w = orig_np.shape[:2]

# -----------------------------
# Process lines
# -----------------------------
for i, line in enumerate(segmentation.lines):

    if not hasattr(line, "boundary") or line.boundary is None:
        continue

    pts = np.array(line.boundary, dtype=np.int32)

    # Draw polygon (for debug)
    cv2.polylines(cv_img, [pts], True, (0, 255, 0), 2)

    # -----------------------------
    # Create mask on ORIGINAL image
    # -----------------------------
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    # Smooth mask (important)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Apply mask to ORIGINAL image
    masked = cv2.bitwise_and(orig_np, orig_np, mask=mask)

    # -----------------------------
    # Get bounding box
    # -----------------------------
    x, y, bw, bh = cv2.boundingRect(pts)

    PAD_X = int(bw * 0.05)
    PAD_Y = int(bh * 0.15)

    x1 = max(0, x - PAD_X)
    y1 = max(0, y - PAD_Y)
    x2 = min(w, x + bw + PAD_X)
    y2 = min(h, y + bh + PAD_Y)

    # -----------------------------
    # Crop from ORIGINAL masked image
    # -----------------------------
    crop = masked[y1:y2, x1:x2]

    # Convert to PIL (KEEP COLOR)
    crop_pil = Image.fromarray(crop)

    # Save
    crop_pil.save(f"lines/line_{i}.png")

# -----------------------------
# Save visualization
# -----------------------------
cv2.imwrite("detected_kraken_processed.png", cv_img)

print("✅ Done — using ORIGINAL image (no grayscale loss)")