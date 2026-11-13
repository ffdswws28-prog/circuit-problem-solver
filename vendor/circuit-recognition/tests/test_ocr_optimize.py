# -*- coding: utf-8 -*-
"""OCR 优化对比测试 — 全图 OCR vs 区域裁剪 OCR"""
import sys
import time
from pathlib import Path

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

import numpy as np
from src.pipeline.ocr_engine import OCREngine
from src.pipeline.ocr_optimizer import compute_text_rois, ocr_in_regions, estimate_speedup
from src.pipeline.yolo_detector import YOLODetector


def test_ocr_optimization():
    print("=" * 70)
    print("OCR 优化对比测试 — 全图 vs 区域裁剪")
    print("=" * 70)

    # Load models
    print("\n[1] 加载模型...")
    yolo = YOLODetector()
    yolo.load_model()

    ocr = OCREngine(use_easyocr=False, enable_correction=True)
    ocr.load_model()

    # Test images
    import fitz
    pdf_path = Path(r"D:\桌面\ai\电路.pdf")
    doc = fitz.open(str(pdf_path))

    # Pick pages with circuits
    test_pages = list(range(min(10, doc.page_count)))

    print(f"\n[2] 测试 {len(test_pages)} 页...")
    print("-" * 70)

    total_original = 0
    total_optimized = 0
    total_full_detections = 0
    total_roi_detections = 0
    page_results = []

    for p_idx in test_pages:
        page = doc[test_pages[p_idx]]
        pix = page.get_pixmap(dpi=200)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n >= 3:
            img = img[:, :, [2, 1, 0]]  # RGBA/RGB → BGR
        h, w = img.shape[:2]

        # YOLO detection
        t_yolo = time.time()
        detections = yolo.detect(img, conf_threshold=0.3)
        yolo_time = time.time() - t_yolo

        if not detections:
            print(f"  页 {p_idx+1}: 无检测，跳过")
            continue

        # Compute ROIs
        rois = compute_text_rois(detections, (h, w))
        est = estimate_speedup(detections, (h, w))

        # --- Full-image OCR (慢) ---
        t1 = time.time()
        results_full = ocr.recognize_all(img)
        full_time = time.time() - t1

        # --- Region-based OCR (快) ---
        results_roi, roi_time = ocr_in_regions(ocr, img, rois)

        # --- Report ---
        total_original += full_time
        total_optimized += roi_time
        total_full_detections += len(results_full)
        total_roi_detections += len(results_roi)

        speedup = full_time / roi_time if roi_time > 0 else 999
        print(f"  页 {p_idx+1:2d} | {w}x{h} | {len(detections):2d}元件 | "
              f"ROI覆盖:{est['coverage']}% | {est['num_rois']}区域 | "
              f"全图:{full_time:.2f}s → 裁剪:{roi_time:.2f}s ({speedup:.1f}x) | "
              f"文字: {len(results_full)}→{len(results_roi)}")

        page_results.append({
            "page": p_idx + 1,
            "detections": len(detections),
            "rois": est['num_rois'],
            "coverage": est['coverage'],
            "full_time": full_time,
            "roi_time": roi_time,
            "speedup": speedup,
            "full_texts": len(results_full),
            "roi_texts": len(results_roi),
        })

    doc.close()

    # Summary
    print(f"\n[3] 汇总")
    print("=" * 70)
    avg_speedup = total_original / total_optimized if total_optimized > 0 else 0
    print(f"  全图 OCR 总计:   {total_original:.2f}s")
    print(f"  裁剪 OCR 总计:   {total_optimized:.2f}s")
    print(f"  总加速比:        {avg_speedup:.1f}x")
    print(f"  全图文字总数:    {total_full_detections}")
    print(f"  裁剪文字总数:    {total_roi_detections}")

    # Check text coverage
    text_recall = total_roi_detections / max(total_full_detections, 1) * 100
    print(f"  文字召回率:      {text_recall:.0f}%")

    print(f"\n  逐页详情:")
    for r in page_results:
        bar = "█" * min(int(r['speedup']), 30)
        print(f"    页{r['page']:2d}: {r['full_time']:.2f}s→{r['roi_time']:.2f}s "
              f"({r['speedup']:.1f}x) {bar} | 文字{r['full_texts']}→{r['roi_texts']}")

    print(f"\n{'=' * 70}")
    if avg_speedup >= 3:
        print(f"  ✅ OCR 加速 {avg_speedup:.1f}x，文字召回率 {text_recall:.0f}%")
    else:
        print(f"  ⚠️ 加速比 {avg_speedup:.1f}x 低于预期，检查 ROI 覆盖")

    return 0


if __name__ == "__main__":
    exit(test_ocr_optimization())
