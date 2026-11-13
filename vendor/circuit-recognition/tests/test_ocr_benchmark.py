# -*- coding: utf-8 -*-
"""OCR 加速方案对比: 原始 vs 降采样 vs 区域裁剪"""
import sys
import time
from pathlib import Path

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

import cv2
import numpy as np
from src.pipeline.ocr_engine import OCREngine
from src.pipeline.yolo_detector import YOLODetector
from src.pipeline.ocr_optimizer import compute_text_rois, ocr_in_regions


def scale_coords_back(results, scale_x, scale_y):
    """Scale OCR results back to original image coordinates"""
    for r in results:
        r["bbox"][0] = int(r["bbox"][0] * scale_x)
        r["bbox"][1] = int(r["bbox"][1] * scale_y)
        r["bbox"][2] = int(r["bbox"][2] * scale_x)
        r["bbox"][3] = int(r["bbox"][3] * scale_y)
    return results


def test_strategies():
    print("=" * 70)
    print("OCR 加速方案对比")
    print("=" * 70)

    ocr = OCREngine(use_easyocr=False, enable_correction=False)
    ocr.load_model()

    yolo = YOLODetector()
    yolo.load_model()

    import fitz
    doc = fitz.open(r"D:\桌面\ai\电路.pdf")
    test_pages = list(range(min(10, doc.page_count)))

    strategies = {
        "original": None,      # no change
        "downscale_2x": 2,     # 1/4 pixels
        "downscale_3x": 3,     # 1/9 pixels
        "max_1024": 1024,      # limit max dim
        "max_800": 800,        # more aggressive
    }

    total_times = {k: 0.0 for k in strategies}
    total_texts = {k: 0 for k in strategies}

    for p_idx in test_pages:
        page = doc[test_pages[p_idx]]
        pix = page.get_pixmap(dpi=200)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n >= 3:
            img = img[:, :, [2, 1, 0]]
        h, w = img.shape[:2]

        print(f"\n  页 {p_idx+1} ({w}x{h}):")

        for name, cfg in strategies.items():
            if cfg is None:
                # Original (no change)
                t0 = time.time()
                results = ocr.detect_text_regions(img)
                elapsed = time.time() - t0
            elif isinstance(cfg, int) and cfg > 1:
                # Downscale by factor
                new_w, new_h = w // cfg, h // cfg
                small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                t0 = time.time()
                results = ocr.detect_text_regions(small)
                scale_coords_back(results, w / new_w, h / new_h)
                elapsed = time.time() - t0
            else:
                # Max dimension limit
                max_dim = cfg
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    t0 = time.time()
                    results = ocr.detect_text_regions(small)
                    scale_coords_back(results, w / new_w, h / new_h)
                    elapsed = time.time() - t0
                else:
                    t0 = time.time()
                    results = ocr.detect_text_regions(img)
                    elapsed = time.time() - t0

            total_times[name] += elapsed
            total_texts[name] += len(results)
            print(f"    {name:16s}: {elapsed:.2f}s | {len(results):3d} 文字")

    doc.close()

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  汇总 (10页)")
    print(f"{'=' * 70}")
    print(f"  {'方案':20s} {'总时间':>8s} {'文字数':>8s} {'加速比':>8s}")
    print(f"  {'-' * 45}")

    orig_time = total_times["original"]
    orig_texts = total_texts["original"]

    for name in strategies:
        t = total_times[name]
        n = total_texts[name]
        speedup = orig_time / t if t > 0 else 999
        recall = n / max(orig_texts, 1) * 100
        mark = " ← 推荐" if 2 < speedup < 20 and recall > 70 else ""
        print(f"  {name:20s} {t:7.1f}s {n:7d}  {speedup:6.1f}x  (召回{recall:.0f}%){mark}")

    return 0


if __name__ == "__main__":
    exit(test_strategies())
