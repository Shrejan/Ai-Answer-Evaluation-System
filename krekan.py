from kraken import binarization
from kraken import pageseg
from PIL import Image
import numpy as np
import cv2

# 1. Load image
image = Image.open("processed.jpg").convert("L")

# 2. Binarize
bin_img = binarization.nlbin(image)

# 3. Segment
segmentation = pageseg.segment(bin_img)

# 4. Convert to OpenCV
cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2BGR)

# 5. Draw lines (FIXED)
for line in segmentation.lines:
    
    # CASE 1: Polygon exists
    if hasattr(line, "boundary") and line.boundary is not None:
        pts = np.array(line.boundary, dtype=np.int32)
        cv2.polylines(cv_img, [pts], True, (0, 255, 0), 2)

    # CASE 2: Only bounding box
    elif hasattr(line, "bbox") and line.bbox is not None:
        x1, y1, x2, y2 = map(int, line.bbox)
        cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 6. Save output
cv2.imwrite("detected_kraken.png", cv_img)

print("✅ Done! Saved as detected_kraken.png")