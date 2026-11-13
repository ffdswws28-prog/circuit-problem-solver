# Debug: check RapidOCR det format and YOLO bbox distances
import sys, time
sys.path.insert(0, r"D:\ai\circuit-recognition")
sys.path.insert(0, r"D:\ai\circuit-recognition\src")

import numpy as np
import fitz
from rapidocr_onnxruntime import RapidOCR
from src.pipeline.yolo_detector import YOLODetector

ocr = RapidOCR()
yolo = YOLODetector()
yolo.load_model()

doc = fitz.open(r"D:\桌面\ai\电路.pdf")
page = doc[0]
pix = page.get_pixmap(dpi=200)
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, ::-1]

# Get det boxes via use_rec=False
result, elapse = ocr(img, use_rec=False)
print(f"det_result type: {type(result)}")
print(f"det_result[0] type: {type(result[0])}")
print(f"det_result[0]: {result[0]}")
print(f"det_result len: {len(result)}")

# Convert to numpy boxes
boxes = [np.array(b, dtype=np.float32) for b in result]

# YOLO
dets = yolo.detect(img, 0.3)
print(f"\nYOLO: {len(dets)} components")
for d in dets:
    x, y, w, h = d["bbox"]
    print(f"  {d['class']}: bbox=[{x},{y},{w},{h}] center=({x+w//2},{y+h//2})")

# For each text box, find distance to nearest YOLO center
print(f"\nText boxes ({len(boxes)}):")
for i, box in enumerate(boxes):
    cx = int(box[:, 0].mean())
    cy = int(box[:, 1].mean())
    min_dist = 9999
    for d in dets:
        dx, dy, dw, dh = d["bbox"]
        dcx, dcy = dx + dw//2, dy + dh//2
        dist = ((cx - dcx)**2 + (cy - dcy)**2)**0.5
        min_dist = min(min_dist, dist)
    near = "NEAR" if min_dist < 500 else "FAR"
    print(f"  {i:2d}: center=({cx:4d},{cy:4d}) min_dist={min_dist:5.0f}px {near}")

doc.close()
