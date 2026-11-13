# -*- coding: utf-8 -*-
"""OCR 优化最终对比测试 — 原始 vs optimize_near_components"""
import sys, time
from pathlib import Path

sys.path.insert(0, r"D:\ai\circuit-recognition")
sys.path.insert(0, r"D:\ai\circuit-recognition\src")

import numpy as np
import fitz
from src.pipeline.ocr_engine import OCREngine
from src.pipeline.yolo_detector import YOLODetector

print("=" * 70)
print("OCR 优化对比 — 原始 vs 只识别元件附近文字")
print("=" * 70)

ocr = OCREngine(use_easyocr=False, enable_correction=True)
ocr.load_model()

yolo = YOLODetector()
yolo.load_model()

doc = fitz.open(r"D:\桌面\ai\电路.pdf")
n_pages = min(10, doc.page_count)

total_orig = 0
total_opt = 0
total_texts_orig = 0
total_texts_opt = 0

for p_idx in range(n_pages):
    page = doc[p_idx]
    pix = page.get_pixmap(dpi=200)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )[:, :, ::-1]

    # YOLO detection
    dets = yolo.detect(img, conf_threshold=0.3)

    if not dets:
        continue

    # Original OCR
    t0 = time.time()
    results_orig = ocr.recognize_all(img)
    t_orig = time.time() - t0

    # Optimized OCR
    t0 = time.time()
    results_opt = ocr.recognize_near_components(img, dets)
    t_opt = time.time() - t0

    total_orig += t_orig
    total_opt += t_opt
    total_texts_orig += len(results_orig)
    total_texts_opt += len(results_opt)

    speedup = t_orig / t_opt if t_opt > 0 else 999
    recall = len(results_opt) / max(len(results_orig), 1) * 100
    print(f"  页{p_idx+1:2d}: {t_orig:.2f}s→{t_opt:.2f}s ({speedup:.1f}x) | "
          f"文字{len(results_orig)}→{len(results_opt)} ({recall:.0f}%)")

doc.close()

print(f"\n{'=' * 70}")
print(f"  合计: {total_orig:.1f}s → {total_opt:.1f}s "
      f"({total_orig/total_opt:.1f}x)")
print(f"  文字: {total_texts_orig} → {total_texts_opt} "
      f"({total_texts_opt/max(total_texts_orig,1)*100:.0f}%)")
print(f"  等效加速: 全流程从 {total_orig/n_pages:.1f}s/页 → "
      f"{total_opt/n_pages:.1f}s/页")
