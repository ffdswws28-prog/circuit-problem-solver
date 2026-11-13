# -*- coding: utf-8 -*-
"""Compare synthetic-only vs mixed-trained model on E2E circuit diagram."""
import sys
sys.path.insert(0, r"D:\ai\circuit-recognition")
from ultralytics import YOLO
from pathlib import Path
import json

TEST_IMAGE = r"D:\ai\circuit-recognition\tests\_e2e_test_circuit.png"
OLD_MODEL = r"D:\ai\circuit-recognition\models\circuit_yolov8n_enhanced.pt"
NEW_MODEL = r"D:\ai\circuit-recognition\models\circuit_yolov8n_mixed\weights\best.pt"

print("=" * 60)
print("Model Comparison: Synthetic-Only vs Mixed (Synthetic+CGHD)")
print("=" * 60)

for label, model_path in [("Old (Synth Only)", OLD_MODEL), ("New (Mixed)", NEW_MODEL)]:
    print(f"\n--- {label} ---")
    print(f"  Model: {model_path}")
    model = YOLO(model_path)
    results = model(TEST_IMAGE, conf=0.25, verbose=False)

    r = results[0]
    boxes = r.boxes
    if boxes is None:
        print("  No detections!")
        continue

    classes = {}
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        if cls_name not in classes:
            classes[cls_name] = []
        classes[cls_name].append(round(conf, 3))

    total = len(boxes)
    print(f"  Total detections: {total}")
    for cls_name, confs in sorted(classes.items()):
        print(f"    {cls_name:12s}: {len(confs):2d}  (avg conf: {sum(confs)/len(confs):.3f})")

    # Save annotated image
    out_path = r"D:\ai\circuit-recognition\tests\_compare_" + ("old" if "Old" in label else "new") + ".jpg"
    annotated = r.plot(show=False, save=True, filename=out_path)
    print(f"  Annotated: {out_path}")

print(f"\n{'='*60}")
print("Comparison complete! Check D:\\ai\\circuit-recognition\\tests\\_compare_*.jpg")
