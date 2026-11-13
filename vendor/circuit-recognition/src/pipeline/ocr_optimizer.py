# -*- coding: utf-8 -*-
"""OCR 推理优化 — 基于 YOLO 检测结果的区域裁剪 OCR

核心思路: YOLO 先定位元件位置 → 在元件周围扩展搜索区 →
只在搜索区跑 OCR，跳过大量空白背景，实现 3-10x OCR 加速。

同时保留全局文字（标题、注释等）的检测能力。
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path


def compute_text_rois(
    detections: List[Dict],
    image_shape: Tuple[int, int],
    padding_ratio: float = 2.5,
    min_crop_size: int = 40,
    include_margins: bool = True,
) -> List[List[int]]:
    """根据 YOLO 检测结果计算 OCR 搜索区域。

    对每个元件的 bbox 向外扩展 padding_ratio 倍，
    然后合并重叠区域，避免重复 OCR。

    Args:
        detections: YOLO 检测结果 [{"bbox": [x,y,w,h], ...}, ...]
        image_shape: (h, w) 图像尺寸
        padding_ratio: 扩展比例（相对 bbox 尺寸）
        min_crop_size: 最小裁剪尺寸
        include_margins: 是否包含页眉/页脚区域（标题、注释）

    Returns:
        List of [x, y, w, h] ROIs in image coordinates
    """
    h, w = image_shape[:2]
    regions = []

    for det in detections:
        x, y, bw, bh = det["bbox"]

        # 扩展搜索区域：向上/左多扩展（标签通常在元件上方或左侧）
        pad_x = max(int(bw * padding_ratio), min_crop_size // 2)
        pad_y = max(int(bh * padding_ratio), min_crop_size // 2)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y // 2 - pad_y)  # 上方多留空间给标签
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        if x2 - x1 >= min_crop_size and y2 - y1 >= min_crop_size:
            regions.append([x1, y1, x2 - x1, y2 - y1])

    if not regions:
        # 没有检测结果，回退到全图
        return [[0, 0, w, h]]

    # 合并重叠区域
    regions = _merge_overlapping(regions)

    # 可选：添加页眉/页脚区域
    if include_margins and h > 400:
        top_strip_h = min(80, h // 15)
        bottom_strip_h = min(60, h // 20)
        margin_regions = [
            [0, 0, w, top_strip_h],                    # 顶部标题区
            [0, h - bottom_strip_h, w, bottom_strip_h],  # 底部注释区
        ]
        regions = _merge_overlapping(regions + margin_regions)

    return regions


def _merge_overlapping(regions: List[List[int]], iou_threshold: float = 0.1) -> List[List[int]]:
    """合并重叠/相邻的矩形区域（贪心算法）"""
    if len(regions) <= 1:
        return regions

    # 按 x 坐标排序
    regions = sorted(regions, key=lambda r: (r[0], r[1]))
    merged = []
    used = [False] * len(regions)

    for i, r1 in enumerate(regions):
        if used[i]:
            continue
        x1, y1, w1, h1 = r1
        x1b, y1b = x1 + w1, y1 + h1

        # 尝试合并后续区域
        for j in range(i + 1, len(regions)):
            if used[j]:
                continue
            x2, y2, w2, h2 = regions[j]
            x2b, y2b = x2 + w2, y2 + h2

            # 计算 IOU
            inter_x1 = max(x1, x2)
            inter_y1 = max(y1, y2)
            inter_x2 = min(x1b, x2b)
            inter_y2 = min(y1b, y2b)

            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                area1 = w1 * h1
                area2 = w2 * h2
                union_area = area1 + area2 - inter_area
                iou = inter_area / union_area if union_area > 0 else 0

                if iou > iou_threshold:
                    # 合并
                    x1 = min(x1, x2)
                    y1 = min(y1, y2)
                    x1b = max(x1b, x2b)
                    y1b = max(y1b, y2b)
                    w1 = x1b - x1
                    h1 = y1b - y1
                    used[j] = True

        merged.append([x1, y1, w1, h1])
        used[i] = True

    return merged


def ocr_in_regions(
    ocr_engine,
    image: np.ndarray,
    rois: List[List[int]],
    global_ocr: bool = False,
) -> Tuple[List[Dict], float]:
    """在指定区域运行 OCR，比全图 OCR 快 3-10x。

    Args:
        ocr_engine: OCREngine 实例
        image: 原始图像 (H, W, 3) BGR
        rois: 搜索区域列表 [[x, y, w, h], ...]
        global_ocr: 是否同时运行全图 OCR（用于对比/回退）

    Returns:
        (ocr_results, total_time)
    """
    import time
    t0 = time.time()

    all_results = []
    seen_texts = set()  # 去重

    for roi in rois:
        x, y, w, h = roi
        if w < 20 or h < 20:
            continue

        crop = image[y:y + h, x:x + w]
        if crop.size == 0:
            continue

        # 在当前区域运行 OCR
        region_results = ocr_engine.detect_text_regions(crop)

        # 坐标映射回全图
        for r in region_results:
            bx, by, bw, bh = r["bbox"]
            r["bbox"] = [bx + x, by + y, bw, bh]

            # 去重（相同文本 + 相近位置）
            key = (r["text"], round(r["bbox"][0] / 50), round(r["bbox"][1] / 50))
            if key not in seen_texts:
                seen_texts.add(key)
                all_results.append(r)

    elapsed = time.time() - t0
    return all_results, elapsed


def estimate_speedup(detections: List[Dict], image_shape: Tuple[int, int]) -> Dict:
    """估算 OCR 加速比"""
    h, w = image_shape[:2]
    total_area = w * h

    rois = compute_text_rois(detections, image_shape)
    roi_area = sum(r[2] * r[3] for r in rois)

    coverage = roi_area / total_area if total_area > 0 else 1.0
    # RapidOCR 时间大致与像素数成正比
    estimated_speedup = 1.0 / max(coverage, 0.05)

    return {
        "total_area": total_area,
        "roi_area": roi_area,
        "coverage": round(coverage * 100, 1),
        "num_rois": len(rois),
        "estimated_speedup": round(estimated_speedup, 1),
    }
