from paddleocr import PaddleOCR
import cv2
import numpy as np

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=False, lang='en')

# Image path
img_path = "test5.jpg"

# Read image
img = cv2.imread(img_path)
#cleaning image

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
# thresholding
thresh = cv2.adaptiveThreshold(
    blur,255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,2
)

# save processed image
cv2.imwrite("processed.jpg", thresh)
# Detect text boxes only
result = ocr.ocr("processed.jpg", det=True, rec=False)
path = img = cv2.imread("processed.jpg")
# Draw boxes
for box in result[0]:
    pts = np.array(box).astype(int)
    cv2.polylines(path, [pts], True, (0,255,0), 2)

# Save output image
cv2.imwrite("detected_with_gs_text.jpg", path)

print("Detection complete. Check detected_with_gs_text.jpg")