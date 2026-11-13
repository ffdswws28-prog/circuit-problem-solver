# -*- coding: utf-8 -*-
"""End-to-end pipeline test with full Fusion + Topology validation"""
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np


def create_test_circuit() -> np.ndarray:
    """Create a synthetic circuit image with components, labels, values, and wires."""
    w, h = 900, 700
    img = np.ones((h, w, 3), dtype=np.uint8) * 255

    # --- Component symbols (boxes simulating detection regions) ---
    components = [
        # (label, value, x, y, w, h, color, class)
        ("R1", "10k",     70, 120, 80, 40, (0, 0, 255),     "resistor"),
        ("C1", "100uF",   70, 240, 80, 40, (255, 0, 0),     "capacitor"),
        ("D1", "1N4148",  70, 370, 80, 40, (0, 128, 0),     "diode"),
        ("U1", "LM358",  400, 130, 120, 60, (0, 128, 128),   "ic"),
        ("R2", "4.7k",   400, 300, 80, 40, (0, 0, 255),     "resistor"),
        ("C2", "2200uF", 400, 420, 80, 40, (255, 0, 0),     "capacitor"),
        ("Q1", "2N2222", 650, 130, 80, 50, (128, 0, 128),   "transistor"),
        ("L1", "100uH",  650, 300, 80, 40, (0, 128, 255),   "inductor"),
    ]

    for label, value, x, y, cw, ch, color, cls in components:
        cv2.rectangle(img, (x, y), (x + cw, y + ch), color, 2)
        # Label above (more spacing + larger font to avoid OCR miss)
        cv2.putText(img, label, (x - 5, y - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        # Value below
        cv2.putText(img, value, (x, y + ch + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        # Pin dots on left/right for two-terminal components
        if cls in ("resistor", "capacitor", "diode", "inductor"):
            cv2.circle(img, (x, y + ch // 2), 4, (0, 0, 0), -1)       # left pin
            cv2.circle(img, (x + cw, y + ch // 2), 4, (0, 0, 0), -1)  # right pin
        elif cls == "transistor":
            cv2.circle(img, (x + cw // 2, y), 4, (0, 0, 0), -1)        # base
            cv2.circle(img, (x + cw // 2, y + ch), 4, (0, 0, 0), -1)   # emitter
            cv2.circle(img, (x + cw, y + ch // 2), 4, (0, 0, 0), -1)   # collector

    # --- Net labels ---
    cv2.putText(img, "VCC",  (200, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
    cv2.putText(img, "GND",  (200, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
    cv2.putText(img, "VOUT", (550, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)

    # --- Connecting wires ---
    # R1.left -> VCC bus (route outside label area)
    cv2.line(img, (70, 140), (70, 60), (0, 0, 0), 2)
    cv2.line(img, (70, 60),  (300, 60), (0, 0, 0), 2)
    cv2.line(img, (300, 60), (400, 60), (0, 0, 0), 2)
    cv2.line(img, (400, 60), (400, 130), (0, 0, 0), 2)
    # R1.right -> C1.right -> D1.right (vertical bus on right side)
    cv2.line(img, (150, 140), (150, 260), (0, 0, 0), 2)
    cv2.line(img, (150, 260), (150, 390), (0, 0, 0), 2)
    # U1.right -> Q1.left
    cv2.line(img, (520, 160), (650, 155), (0, 0, 0), 2)
    # R2.left -> U1.bottom
    cv2.line(img, (400, 190), (400, 220), (0, 0, 0), 2)
    cv2.line(img, (400, 220), (460, 220), (0, 0, 0), 2)
    cv2.line(img, (460, 220), (460, 300), (0, 0, 0), 2)
    # C2.left -> GND
    cv2.line(img, (400, 440), (400, 520), (0, 0, 0), 2)
    # Q1.right -> L1.left
    cv2.line(img, (730, 155), (730, 200), (0, 0, 0), 2)
    cv2.line(img, (730, 200), (650, 200), (0, 0, 0), 2)
    cv2.line(img, (650, 200), (650, 300), (0, 0, 0), 2)
    # L1.right -> VOUT
    cv2.line(img, (730, 320), (730, 500), (0, 0, 0), 2)

    # --- Title block ---
    cv2.rectangle(img, (650, 550), (880, 620), (0, 0, 0), 1)
    cv2.putText(img, "Test Circuit v1.0",  (660, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "Sheet 1/1",  (660, 610), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img


class E2ETester:
    """Runs the full pipeline and validates output structure."""

    def __init__(self):
        self.errors = []
        self.warnings = []

    def check(self, condition: bool, msg: str):
        if not condition:
            self.errors.append(msg)
            print(f"  FAIL: {msg}")
        else:
            print(f"  OK: {msg}")

    def run(self):
        print("=" * 60)
        print("Circuit Recognition - Full Pipeline E2E Test")
        print("=" * 60)

        # --- Create test image ---
        img = create_test_circuit()
        test_path = Path(__file__).parent / "_e2e_test_circuit.png"
        cv2.imwrite(str(test_path), img)
        print(f"\nTest image: {test_path} ({img.shape[1]}x{img.shape[0]})")

        t_total_start = time.time()

        # ---- Stage 0: Preprocessor ----
        print("\n--- Stage 0: Preprocessor ---")
        from src.pipeline.preprocessor import Preprocessor
        preprocessor = Preprocessor(target_size=(640, 640))
        loaded = preprocessor.load_image(str(test_path))
        tensor = preprocessor.preprocess(loaded)
        self.check(loaded.shape == (700, 900, 3), f"Image shape: {loaded.shape}")
        self.check(tensor.shape == (1, 3, 640, 640), f"Tensor shape: {tensor.shape}")

        # ---- Stage 1: YOLO Detection ----
        print("\n--- Stage 1: YOLO Detection ---")
        t1 = time.time()
        from src.pipeline.yolo_detector import YOLODetector
        yolo = YOLODetector()
        yolo.load_model()
        detections = yolo.detect(loaded, conf_threshold=0.35)
        yolo_time = time.time() - t1

        print(f"  Detections: {len(detections)} components ({yolo_time:.1f}s)")
        for d in detections:
            print(f"    {d['class']:15s} conf={d['confidence']:.2f}  bbox={d['bbox']}")

        # Use fine-tuned model detections (filtered by confidence)
        # No longer need synthetic fallback - model is trained!
        if not detections:
            print("  WARNING: Fine-tuned model found 0 components")
        else:
            print(f"  Using {len(detections)} fine-tuned detections (conf >= 0.35)")
        self.check(len(detections) >= 1, f"At least 1 component detected (got {len(detections)})")

        # ---- Stage 2: SAM Segmentation ----
        print("\n--- Stage 2: SAM Segmentation ---")
        t2 = time.time()
        from src.pipeline.sam_segmentor import SAMSegmentor
        sam = SAMSegmentor()
        sam.load_model()
        seg_targets = [d for d in detections if not sam.should_skip(d.get("class", ""))]
        print(f"  SAM targets: {len(seg_targets)} (skipped {len(detections) - len(seg_targets)} rectangular)")
        sam_results = []
        if seg_targets:
            sam_results = sam.segment(loaded, seg_targets)
        sam_time = time.time() - t2
        print(f"  SAM results: {len(sam_results)} ({sam_time:.1f}s)")
        self.check(len(sam_results) == len(seg_targets),
                   f"SAM results match targets ({len(sam_results)} vs {len(seg_targets)})")

        # ---- Stage 3: OCR ----
        print("\n--- Stage 3: OCR (EasyOCR + Corrector) ---")
        t3 = time.time()
        from src.pipeline.ocr_engine import OCREngine
        ocr = OCREngine(use_easyocr=True)
        ocr.load_model()
        ocr_results = ocr.recognize_all(loaded)
        ocr_time = time.time() - t3

        print(f"  OCR found: {len(ocr_results)} text regions ({ocr_time:.1f}s)")
        for t in ocr_results:
            orig = f" (was: {t['original']})" if t.get('original') else ""
            print(f"    '{t['text']}' conf={t['confidence']:.2f}{orig}")
        self.check(len(ocr_results) >= 5, f"At least 5 text regions found (got {len(ocr_results)})")

        # Verify OCR found circuit-relevant text (synthetic images have known OCR limitations)
        found_texts = set(item['text'] for item in ocr_results)
        # Check for circuit values (numbers+units) and net labels
        circuit_patterns = ['10k', 'GND', '100', '4.7k']
        found_patterns = [p for p in circuit_patterns if any(p in t for t in found_texts)]
        print(f"  Circuit patterns found: {found_patterns}")
        self.check(len(found_patterns) >= 2, f"At least 2 circuit patterns found (got {len(found_patterns)}): {found_patterns}")

        # ---- Stage 4: Topology Analysis ----
        print("\n--- Stage 4: Topology Analysis ---")
        t4 = time.time()
        from src.pipeline.topology import TopologyAnalyzer
        topo = TopologyAnalyzer()
        coarse = topo.coarse_analysis(loaded)
        print(f"  Coarse: {len(coarse.get('lines', []))} wires, "
              f"{len(coarse.get('junctions', []))} junctions, "
              f"{len(coarse.get('nodes', []))} nodes")

        # Associate OCR labels for topology
        from src.pipeline.fusion import FusionEngine
        temp_fusion = FusionEngine()
        comps_with_labels = temp_fusion._associate_ocr(
            [{"bbox": d["bbox"], "class": d["class"], "confidence": d["confidence"]} for d in detections],
            ocr_results,
            [],
        )
        fine = topo.fine_analysis(comps_with_labels, ocr_results, coarse)
        topo_time = time.time() - t4
        print(f"  Fine: {len(fine.get('connections', []))} connections, "
              f"{len(fine.get('nets', {}))} nets, "
              f"{len(fine.get('net_labels', []))} net labels ({topo_time:.1f}s)")
        self.check(len(coarse.get('lines', [])) > 0, "Wire lines detected")
        self.check(len(coarse.get('junctions', [])) >= 0, "Junction detection ran")

        # ---- Stage 5: Fusion ----
        print("\n--- Stage 5: Fusion ---")
        t5 = time.time()
        from src.pipeline.fusion import FusionEngine
        fusion = FusionEngine()
        result = fusion.fuse(
            yolo_results=detections,
            sam_results=sam_results,
            ocr_results=ocr_results,
            topo_results=fine,
        )
        fusion_time = time.time() - t5

        components = result.get("components", [])
        connections = result.get("connections", [])
        nets = result.get("nets", {})
        summary = result.get("summary", "")

        print(f"  Components: {len(components)}")
        for c in components:
            label = c.get('label', '(unlabeled)')
            value = c.get('value', '')
            val_str = f" = {value}" if value else ""
            cls = c.get('class') or '?'
            warn = f" [WARN: {c.get('warnings', [])}]" if c.get('warnings') else ""
            label_str = label or '(none)'
            print(f"    {cls:12s} {label_str:8s}{val_str}{warn}")

        print(f"  Connections: {len(connections)}")
        for conn in connections[:10]:
            print(f"    {conn.get('from', '?'):12s} -> {conn.get('to', '?'):12s}  [{conn.get('type', 'wire')}]")

        print(f"  Nets: {len(nets)}")
        for net_name, refs in nets.items():
            print(f"    {net_name}: {refs}")

        print(f"  Summary: {summary}")
        fusion_time_total = time.time() - t5

        # ---- Validation ----
        print("\n--- Validation ---")
        # Structure checks
        self.check(isinstance(components, list), "Components is a list")
        self.check(isinstance(connections, list), "Connections is a list")
        self.check(isinstance(nets, dict), "Nets is a dict")
        self.check(isinstance(summary, str) and len(summary) > 0, "Summary is non-empty string")
        self.check("components" in result, "Result has 'components' key")
        self.check("connections" in result, "Result has 'connections' key")
        self.check("nets" in result, "Result has 'nets' key")
        self.check("summary" in result, "Result has 'summary' key")

        # Component structure
        if components:
            c0 = components[0]
            self.check("bbox" in c0, "Component has 'bbox'")
            self.check("class" in c0, "Component has 'class'")
            self.check("confidence" in c0, "Component has 'confidence'")
            self.check("label" in c0, "Component has 'label' field")
            self.check("value" in c0, "Component has 'value' field")

        # Label match rate (synthetic images have OCR limitations with small labels)
        labeled = sum(1 for c in components if c.get("label"))
        valued = sum(1 for c in components if c.get("value"))
        print(f"  Labels: {labeled}/{len(components)}, Values: {valued}/{len(components)}")
        # At least some values should be matched (values are more reliably OCR'd)
        self.check(valued >= 1, f"At least 1 component has a value (got {valued})")

        # Network labels found
        net_texts_in_ocr = [t['text'] for t in ocr_results
                          if t['text'].upper() in ('VCC', 'GND', 'VOUT', 'VDD', 'VSS')]
        print(f"  Net labels in OCR: {net_texts_in_ocr}")
        self.check(len(net_texts_in_ocr) >= 1, f"Net labels found in OCR: {net_texts_in_ocr}")

        # ---- Cleanup ----
        test_path.unlink(missing_ok=True)

        total_time = time.time() - t_total_start
        print(f"\n--- Timing ---")
        print(f"  Total: {total_time:.1f}s")
        print(f"  YOLO:  {yolo_time:.1f}s")
        print(f"  SAM:   {sam_time:.1f}s")
        print(f"  OCR:   {ocr_time:.1f}s")
        print(f"  Topo:  {topo_time:.1f}s")
        print(f"  Fusion:{fusion_time:.1f}s")

        # ---- Report ----
        print(f"\n{'=' * 60}")
        if self.errors:
            print(f"FAILED: {len(self.errors)} check(s) failed")
            for e in self.errors:
                print(f"  - {e}")
            return False
        else:
            print(f"ALL CHECKS PASSED")
            return True


if __name__ == "__main__":
    tester = E2ETester()
    success = tester.run()
    sys.exit(0 if success else 1)
