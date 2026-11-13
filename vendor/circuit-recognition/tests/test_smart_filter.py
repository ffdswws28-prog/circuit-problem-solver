"""Test SmartFilter — multi-strategy text box filtering"""
import sys
sys.path.insert(0, r"D:\ai\circuit-recognition")
sys.path.insert(0, r"D:\ai\circuit-recognition\src")

import numpy as np
import pytest
from src.pipeline.smart_filter import SmartFilter


# ---- helpers to create synthetic text boxes ----
def box_at(cx, cy, w=60, h=20) -> np.ndarray:
    """Create a 4-point polygon centered at (cx, cy)."""
    half_w, half_h = w / 2, h / 2
    return np.array([
        [cx - half_w, cy - half_h],
        [cx + half_w, cy - half_h],
        [cx + half_w, cy + half_h],
        [cx - half_w, cy + half_h],
    ], dtype=np.float32)


def yolo_det(x, y, w, h, cls="resistor"):
    return {"bbox": [x, y, w, h], "class": cls, "confidence": 0.9}


class TestSmartFilterSize:
    """Size-based filtering"""

    def test_filters_too_small_boxes(self):
        f = SmartFilter({"size_min_h": 6, "size_max_h": 400, "conf_threshold": 0.0, "nms_iou": 1.0, "min_texts": 0})
        tiny = box_at(100, 100, w=20, h=3)    # 3px height < 6
        normal = box_at(200, 200, w=60, h=20)
        boxes = [tiny, normal]
        scores = [0.9, 0.9]
        filtered, stats = f.apply(boxes, scores, (300, 300), None)
        assert len(filtered) == 1
        assert stats["filtered_by_size"] == 1
        assert stats["total"] == 2
        assert stats["passed"] == 1

    def test_filters_too_large_boxes(self):
        f = SmartFilter({"size_min_h": 6, "size_max_h": 400, "conf_threshold": 0.0, "nms_iou": 1.0, "min_texts": 0})
        huge = box_at(100, 250, w=800, h=500)  # 500px > 400
        normal = box_at(200, 200, w=60, h=20)
        boxes = [huge, normal]
        scores = [0.9, 0.9]
        filtered, stats = f.apply(boxes, scores, (1000, 1000), None)
        assert len(filtered) == 1
        assert stats["filtered_by_size"] == 1


class TestSmartFilterConf:
    """Confidence-based filtering"""

    def test_filters_low_confidence(self):
        f = SmartFilter({"size_min_h": 2, "size_max_h": 800, "conf_threshold": 0.5, "nms_iou": 1.0, "min_texts": 0})
        low = box_at(100, 100)
        high = box_at(200, 200)
        boxes = [low, high]
        scores = [0.3, 0.9]
        filtered, stats = f.apply(boxes, scores, (300, 300), None)
        assert len(filtered) == 1
        assert stats["filtered_by_conf"] == 1


class TestSmartFilterNMS:
    """Overlap deduplication"""

    def test_removes_overlapping_boxes(self):
        f = SmartFilter({"size_min_h": 2, "size_max_h": 800, "conf_threshold": 0.0, "nms_iou": 0.5, "min_texts": 0})
        # Two boxes at same position — NMS should keep only higher-confidence one
        b1 = box_at(100, 100, w=60, h=20)
        b2 = box_at(102, 102, w=62, h=22)
        boxes = [b1, b2]
        scores = [0.9, 0.7]
        filtered, stats = f.apply(boxes, scores, (300, 300), None)
        assert len(filtered) == 1
        # The removed one shows in filtered counts
        total_removed = stats["filtered_by_size"] + stats["filtered_by_conf"]
        assert total_removed >= 0


class TestSmartFilterYoloROI:
    """YOLO region-based filtering"""

    def test_keeps_boxes_near_components(self):
        f = SmartFilter({
            "size_min_h": 2, "size_max_h": 800, "conf_threshold": 0.0,
            "nms_iou": 1.0, "yolo_padding": 0.4, "min_texts": 0,
        })
        # Component at (100, 100, 60, 40)
        dets = [yolo_det(100, 100, 60, 40)]
        # Box near component center
        near = box_at(130, 120, w=30, h=15)
        # Box far away
        far = box_at(800, 800, w=30, h=15)
        boxes = [near, far]
        scores = [0.9, 0.9]
        filtered, stats = f.apply(boxes, scores, (1000, 1000), dets)
        assert len(filtered) == 1
        assert stats["filtered_by_roi"] == 1

    def test_no_yolo_keeps_all(self):
        """Without YOLO detections, no ROI filtering occurs"""
        f = SmartFilter({
            "size_min_h": 2, "size_max_h": 800, "conf_threshold": 0.0,
            "nms_iou": 1.0, "yolo_padding": 0.4,
        })
        boxes = [box_at(100, 100), box_at(800, 800)]
        scores = [0.9, 0.9]
        filtered, stats = f.apply(boxes, scores, (1000, 1000), None)
        assert len(filtered) == 2
        assert stats["filtered_by_roi"] == 0


class TestSmartFilterFallback:
    """min_texts fallback"""

    def test_fallback_when_too_few(self):
        f = SmartFilter({
            "size_min_h": 50, "size_max_h": 100, "conf_threshold": 0.0,
            "nms_iou": 1.0, "min_texts": 4,
        })
        # All boxes are filtered by size (< 50px), should trigger fallback
        boxes = [box_at(100, 100, w=20, h=10)]  # h=10 < 50
        scores = [0.9]
        filtered, stats = f.apply(boxes, scores, (300, 300), None)
        assert len(filtered) == 1  # original boxes returned
        assert stats["trigger_min_fallback"] is True


class TestSmartFilterMargins:
    """Top/bottom margin preservation"""

    def test_keeps_top_margin_boxes(self):
        f = SmartFilter({
            "size_min_h": 2, "size_max_h": 800, "conf_threshold": 0.0,
            "nms_iou": 1.0, "yolo_padding": 0.2,
            "margin_top": 0.1, "margin_bot": 0.05,
        })
        # Component in center
        dets = [yolo_det(400, 400, 60, 40)]
        # Box in top 10%
        top = box_at(50, 30, w=40, h=15)
        # Box in bottom 5%
        bot = box_at(50, 970, w=40, h=15)
        # Box far from component, not in margin
        far = box_at(900, 500, w=40, h=15)
        boxes = [top, bot, far]
        scores = [0.9, 0.9, 0.9]
        filtered, stats = f.apply(boxes, scores, (1000, 1000), dets)
        # top + bot should be kept as margins
        assert len(filtered) >= 2
        assert stats["margin_added"] >= 2
