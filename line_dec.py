import json
import numpy as np
import cv2,random

# Load your JSON response (example: from a file like response_*.json)
with open("response_1773894419405.json", "r", encoding="utf-8") as f:
    data = json.load(f)

boxes = data["detections"][0]  # list of quad boxes

def quad_to_bbox(quad):
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return {
        "quad": quad,
        "x_min": min(xs),
        "y_min": min(ys),
        "x_max": max(xs),
        "y_max": max(ys),
        "y_center": (min(ys) + max(ys)) / 2,
    }

bboxes = [quad_to_bbox(q) for q in boxes]

# 1) Sort by vertical position (top → bottom)
bboxes.sort(key=lambda b: b["y_center"])

# 2) Group into lines by y proximity
lines = []
line_threshold = 15  # tweak this (pixels) for your image scale

for bbox in bboxes:
    if not lines:
        lines.append([bbox])
        continue

    last_line = lines[-1]
    # compare to last box in the last line (or use avg y)
    last_y = last_line[-1]["y_center"]
    if abs(bbox["y_center"] - last_y) <= line_threshold:
        last_line.append(bbox)
    else:
        lines.append([bbox])

# 3) Sort boxes in each line left→right
for line in lines:
    line.sort(key=lambda b: b["x_min"])
def random_color():
    return (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    
# lines is now a list of line-groups; each group is sorted left-to-right.
path =cv2.imread("processed.jpg")
for i, line in enumerate(lines):
        col=random_color()
        for bbox in line:
            pts = np.array(bbox["quad"]).astype(int)
            cv2.polylines(path, [pts], True, col, 4)
cv2.imwrite("img.jpg", path)