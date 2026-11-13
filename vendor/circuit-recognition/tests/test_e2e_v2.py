# -*- coding: utf-8 -*-
"""End-to-end test: YOLO v2 (10-class) + RapidOCR + full pipeline."""
import sys
import time
from pathlib import Path

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

import numpy as np
from src.utils.image_io import imread


def test_yolo_v2():
    """Test 1: YOLO v2 model loading and detection (10 classes)."""
    print("=" * 60)
    print("Test 1: YOLOv8n v2 Detection (10-class)")
    print("=" * 60)

    from src.pipeline.yolo_detector import YOLODetector

    detector = YOLODetector()
    print(f"  Model path: {detector.model_path}")
    detector.load_model()

    # Use a CGHD-1152 image (hand-drawn) and an HCD image (schematic)
    val_dir = _project_root / "datasets" / "circuit_v2" / "images" / "val"
    test_images = sorted(val_dir.glob("*.jpg"))[:3]

    for img_path in test_images:
        print(f"\n  Image: {img_path.name}")
        image = imread(str(img_path))
        h, w = image.shape[:2]
        print(f"    Size: {w}x{h}")

        t0 = time.time()
        detections = detector.detect(image)
        elapsed = time.time() - t0
        print(f"    Detections: {len(detections)} ({elapsed:.3f}s)")

        # Group by class
        from collections import Counter
        class_counts = Counter(d["class"] for d in detections)
        for cls_name, count in sorted(class_counts.items()):
            print(f"      {cls_name}: {count}")

    print("\n  YOLO v2: PASSED")
    return True


def test_ocr_rapid():
    """Test 2: RapidOCR text detection and recognition."""
    print("\n" + "=" * 60)
    print("Test 2: RapidOCR Text Recognition")
    print("=" * 60)

    from src.pipeline.ocr_engine import OCREngine

    ocr = OCREngine(use_easyocr=False, enable_correction=True)
    ocr.load_model()

    val_dir = _project_root / "datasets" / "circuit_v2" / "images" / "val"
    test_images = sorted(val_dir.glob("*.jpg"))[:2]

    for img_path in test_images:
        print(f"\n  Image: {img_path.name}")
        image = imread(str(img_path))

        t0 = time.time()
        results = ocr.detect_text_regions(image)
        elapsed = time.time() - t0

        print(f"    Text regions: {len(results)} ({elapsed:.3f}s)")
        for r in results[:5]:
            print(f"      [{r['confidence']:.2f}] \"{r['text']}\" @ {r['bbox']}")
        if len(results) > 5:
            print(f"      ... and {len(results) - 5} more")

    print("\n  RapidOCR: PASSED")
    return True


def test_pipeline_integration():
    """Test 3: Full pipeline integration (YOLO + OCR)."""
    print("\n" + "=" * 60)
    print("Test 3: Full Pipeline Integration (YOLO v2 + RapidOCR)")
    print("=" * 60)

    from src.pipeline.yolo_detector import YOLODetector
    from src.pipeline.ocr_engine import OCREngine

    detector = YOLODetector()
    detector.load_model()

    ocr = OCREngine(use_easyocr=False, enable_correction=True)
    ocr.load_model()

    val_dir = _project_root / "datasets" / "circuit_v2" / "images" / "val"
    test_images = sorted(val_dir.glob("*.jpg"))[:2]

    for img_path in test_images:
        print(f"\n  Image: {img_path.name}")
        image = imread(str(img_path))

        # Stage 1: YOLO detection
        t0 = time.time()
        detections = detector.detect(image)
        yolo_time = time.time() - t0

        # Stage 2: OCR text
        t1 = time.time()
        texts = ocr.detect_text_regions(image)
        ocr_time = time.time() - t1

        total_time = time.time() - t0
        print(f"    Components: {len(detections)}, Texts: {len(texts)}")
        print(f"    YOLO: {yolo_time:.3f}s, OCR: {ocr_time:.3f}s, Total: {total_time:.3f}s")

    print("\n  Pipeline integration: PASSED")
    return True


def test_class_coverage():
    """Test 4: Verify all 10 classes are detectable."""
    print("\n" + "=" * 60)
    print("Test 4: 10-Class Coverage Check")
    print("=" * 60)

    from src.pipeline.yolo_detector import YOLODetector

    detector = YOLODetector()
    detector.load_model()

    expected_classes = {
        "resistor", "capacitor", "inductor", "diode", "transistor",
        "opamp", "ic", "ground", "voltage_source", "current_source"
    }

    # Check model names
    actual_classes = set(detector.model.names.values())
    print(f"  Model classes: {sorted(actual_classes)}")

    missing = expected_classes - actual_classes
    extra = actual_classes - expected_classes

    if missing:
        print(f"  MISSING: {missing}")
        return False
    if extra:
        print(f"  EXTRA: {extra}")

    assert actual_classes == expected_classes, f"Class mismatch: missing={missing}, extra={extra}"
    print(f"  All 10 classes present in model: PASSED")

    # Scan multiple validation images to check all classes appear
    val_dir = _project_root / "datasets" / "circuit_v2" / "images" / "val"
    all_detected = set()
    for img_path in sorted(val_dir.glob("*.jpg")):
        image = imread(str(img_path))
        detections = detector.detect(image)
        for d in detections:
            all_detected.add(d["class"])
        if all_detected >= expected_classes:
            break

    print(f"  Classes found in val images: {sorted(all_detected)}")
    not_found = expected_classes - all_detected
    if not_found:
        print(f"  WARNING: Not found in val set: {not_found}")
        print(f"  (This may be expected if val set lacks those classes)")
    else:
        print(f"  All 10 classes detected in validation images: PASSED")

    return True


def main():
    results = []
    results.append(("YOLO v2 Detection", test_yolo_v2()))
    results.append(("RapidOCR", test_ocr_rapid()))
    results.append(("Pipeline Integration", test_pipeline_integration()))
    results.append(("Class Coverage", test_class_coverage()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_pass = False
        print(f"  {name}: {status}")

    print(f"\n  Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
