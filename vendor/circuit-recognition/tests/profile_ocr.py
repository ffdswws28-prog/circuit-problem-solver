# -*- coding: utf-8 -*-
"""Profile RapidOCR internals to find optimization targets"""
import sys, time
sys.path.insert(0, r"D:\ai\circuit-recognition")
sys.path.insert(0, r"D:\ai\circuit-recognition\src")

import cv2, numpy as np
from rapidocr_onnxruntime import RapidOCR
import fitz

ocr = RapidOCR()
doc = fitz.open(r"D:\桌面\ai\电路.pdf")
pix = doc[0].get_pixmap(dpi=200)
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
    pix.height, pix.width, pix.n
)[:, :, ::-1]  # RGB→BGR

print("=== Full RapidOCR call (warmup) ===")
result, elapse = ocr(img)
print(f"  det={elapse[0]:.3f}s, cls={elapse[1]:.3f}s, rec={elapse[2]:.3f}s")
print(f"  total={sum(elapse):.3f}s, boxes={len(result) if result else 0}")

print("\n=== Separate det + rec ===")
# Run detection only — preprocess returns (img_tensor, padding_info)
prep = ocr.preprocess(img)
img_tensor = prep[0]
print(f"  preprocess result: {type(prep)}, len={len(prep)}")
t0 = time.time()
dt_boxes, det_elapse = ocr.text_det(img_tensor)
print(f"  det only: {det_elapse:.3f}s, boxes={len(dt_boxes)}")

# Run recognition on all boxes
t0 = time.time()
rec_total = 0
for box in dt_boxes:
    crop = ocr.get_crop_img_list(img, [box])[0]
    text, conf = ocr.text_rec(crop)
rec_total = time.time() - t0
print(f"  rec only ({len(dt_boxes)} boxes): {rec_total:.3f}s")

# Filter: only boxes near components
from src.pipeline.yolo_detector import YOLODetector
yolo = YOLODetector()
yolo.load_model()
dets = yolo.detect(img, conf_threshold=0.3)
print(f"\n  YOLO detections: {len(dets)}")

# Compute component centers
comp_centers = []
for d in dets:
    x, y, w, h = d["bbox"]
    comp_centers.append((x + w // 2, y + h // 2))

# Filter text boxes near components
threshold_dist = 300  # pixels
nearby_boxes = []
far_boxes = []
for box in dt_boxes:
    # Compute box center
    cx = int(box[:, 0].mean())
    cy = int(box[:, 1].mean())
    min_dist = min(
        (abs(cx - px) + abs(cy - py)) for px, py in comp_centers
    ) if comp_centers else 99999
    if min_dist < threshold_dist:
        nearby_boxes.append(box)
    else:
        far_boxes.append(box)

print(f"  Nearby boxes: {len(nearby_boxes)}, Far boxes: {len(far_boxes)}")

# Rec only nearby
t0 = time.time()
for box in nearby_boxes:
    crop = ocr.get_crop_img_list(img, [box])[0]
    text, conf = ocr.text_rec(crop)
rec_nearby = time.time() - t0
print(f"  rec nearby only: {rec_nearby:.3f}s")

# Rec only far
t0 = time.time()
for box in far_boxes:
    crop = ocr.get_crop_img_list(img, [box])[0]
    text, conf = ocr.text_rec(crop)
rec_far = time.time() - t0
print(f"  rec far only: {rec_far:.3f}s")

print(f"\n=== 优化分析 ===")
det_time = det_elapse
orig_total = sum(elapse)
opt_total = det_time + rec_nearby
saved = orig_total - opt_total
print(f"  原始: det={det_time:.3f}s + rec={elapse[2]:.3f}s = {orig_total:.3f}s")
print(f"  优化: det={det_time:.3f}s + rec_nearby={rec_nearby:.3f}s = {opt_total:.3f}s")
print(f"  节省: {saved:.3f}s ({saved/orig_total*100:.0f}%)")
print(f"  加速: {orig_total/opt_total:.1f}x")
print(f"  过滤掉 {len(far_boxes)}/{len(dt_boxes)} 个非元件文字框")

doc.close()
