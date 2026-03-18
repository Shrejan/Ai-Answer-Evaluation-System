from fastapi import FastAPI, UploadFile, File
from paddleocr import PaddleOCR
import cv2
import numpy as np


app = FastAPI()

ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False)

@app.put("/ocr")
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

    path =cv2.imread("processed.jpg")
  
    for box in result[0]:
        pts = np.array(box).astype(int)
        cv2.polylines(path, [pts], True, (0, 255, 0), 1)

    return {"detections": result}
