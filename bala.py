from kraken import binarization
from kraken import blla
from kraken.lib.vgsl import TorchVGSLModel   # 👈 IMPORTANT
from PIL import Image
import numpy as np
import cv2

# 1. Load image
image = Image.open("processed.jpg").convert("L")

# 2. Binarize
bin_img = binarization.nlbin(image)

# 3. Load segmentation model CORRECTLY
seg_model = TorchVGSLModel.load_model("blla.mlmodel")

# 4. Run BLLA segmentation
segmentation = blla.segment(bin_img, model=seg_model)

# 5. Convert to OpenCV
cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2BGR)

# 6. Draw polygon lines
for line in segmentation.lines:
    pts = np.array(line.boundary, dtype=np.int32)
    cv2.polylines(cv_img, [pts], True, (0, 255, 0), 2)

# 7. Save output
cv2.imwrite("detected_kraken_processed.png", cv_img)

print("✅ BLLA segmentation SUCCESS!")