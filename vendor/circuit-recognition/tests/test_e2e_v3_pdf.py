# -*- coding: utf-8 -*-
"""V3 端到端测试：YOLO v3 (10-class) + RapidOCR on 电路.pdf (textbook)"""
import sys
import time
from pathlib import Path
from collections import Counter

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

import fitz  # PyMuPDF
import numpy as np
from src.pipeline.yolo_detector import YOLODetector
from src.pipeline.ocr_engine import OCREngine

PDF_PATH = Path(r"D:\桌面\ai\电路.pdf")
OUTPUT_DIR = _project_root / "tests" / "output_v3_pdf"
CONF_THRESHOLD = 0.3
MAX_PAGES = 10  # Test first 10 pages


def main():
    print("=" * 70)
    print("V3 端到端测试 — 电路.pdf (教材风格)")
    print("=" * 70)

    # ── Load models ──
    print("\n[1/4] 加载模型...")
    detector = YOLODetector()
    print(f"  YOLO: {detector.model_path}")
    detector.load_model()

    ocr = OCREngine(use_easyocr=False, enable_correction=True)
    ocr.load_model()
    print("  模型加载完成")

    # ── Extract PDF pages ──
    print(f"\n[2/4] 提取 PDF 页面 ({MAX_PAGES} 页)...")
    doc = fitz.open(str(PDF_PATH))
    total_pages = doc.page_count
    print(f"  PDF 总页数: {total_pages}")

    test_pages = min(MAX_PAGES, total_pages)

    # ── Process each page ──
    print(f"\n[3/4] 处理前 {test_pages} 页...")
    print("-" * 70)

    all_class_counts = Counter()
    page_results = []

    for p in range(test_pages):
        page = doc[p]
        # Render at 200 DPI
        pix = page.get_pixmap(dpi=200)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 3:
            img = img[:, :, ::-1]  # RGB → BGR for OpenCV
        elif pix.n == 4:
            img = img[:, :, [2, 1, 0]]  # RGBA → BGR

        h, w = img.shape[:2]

        # Stage 1: YOLO detection
        t0 = time.time()
        detections = detector.detect(img, conf_threshold=CONF_THRESHOLD)
        yolo_time = time.time() - t0

        # Stage 2: OCR text
        t1 = time.time()
        texts = ocr.detect_text_regions(img)
        ocr_time = time.time() - t1

        # Count classes
        class_counts = Counter(d["class"] for d in detections)
        total_time = yolo_time + ocr_time
        all_class_counts.update(class_counts)

        print(f"  页码 {p+1:2d} | {w}x{h} | YOLO:{yolo_time:.2f}s OCR:{ocr_time:.2f}s | "
              f"元件:{len(detections):2d} 文字:{len(texts):2d}")
        if detections:
            print(f"        类别: {dict(class_counts)}")

        page_results.append({
            "page": p + 1,
            "size": f"{w}x{h}",
            "yolo_time": yolo_time,
            "ocr_time": ocr_time,
            "total_time": total_time,
            "components": len(detections),
            "texts": len(texts),
            "classes": dict(class_counts),
        })

    doc.close()

    # ── Summary ──
    print(f"\n[4/4] 汇总")
    print("=" * 70)
    total_comps = sum(r["components"] for r in page_results)
    total_texts = sum(r["texts"] for r in page_results)
    avg_yolo = np.mean([r["yolo_time"] for r in page_results])
    avg_ocr = np.mean([r["ocr_time"] for r in page_results])
    avg_total = np.mean([r["total_time"] for r in page_results])

    pages_with_detections = sum(1 for r in page_results if r["components"] > 0)
    print(f"  测试页数: {test_pages}/{total_pages}")
    print(f"  有检测结果的页: {pages_with_detections}/{test_pages}")
    print(f"  检测元件总数: {total_comps}")
    print(f"  识别文字总数: {total_texts}")
    print(f"  平均 YOLO: {avg_yolo:.3f}s, OCR: {avg_ocr:.3f}s, 合计: {avg_total:.3f}s/页")
    print(f"\n  全部检测类别:")
    for cls_name, count in all_class_counts.most_common():
        bar = "█" * (count // max(1, all_class_counts.most_common(1)[0][1] // 10))
        print(f"    {cls_name:16s}: {count:3d}  {bar}")

    # Verdict
    print(f"\n{'=' * 70}")
    if pages_with_detections > 0:
        print(f"  ✅ V3 成功在教材 PDF 上检测到 {total_comps} 个电路元件")
        print(f"     覆盖 {len(all_class_counts)} 个类别")
    else:
        print(f"  ❌ V3 在教材 PDF 上未检测到任何元件")
    print(f"{'=' * 70}")

    return 0 if pages_with_detections > 0 else 1


if __name__ == "__main__":
    exit(main())
