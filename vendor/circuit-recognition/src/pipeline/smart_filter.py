"""Multi-strategy text box filter tailored for circuit diagrams."""
import numpy as np
from typing import List, Dict, Optional, Tuple


DEFAULT_CONFIG = {
    "size_min_h": 6,
    "size_max_h": 400,
    "conf_threshold": 0.3,
    "nms_iou": 0.5,
    "yolo_padding": 0.4,
    "margin_top": 0.1,
    "margin_bot": 0.05,
    "min_texts": 4,
}


class SmartFilter:
    """Filters text detection boxes through a chain of strategies."""

    def __init__(self, config: Dict = None):
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.size_min_h = cfg["size_min_h"]
        self.size_max_h = cfg["size_max_h"]
        self.conf_threshold = cfg["conf_threshold"]
        self.nms_iou = cfg["nms_iou"]
        self.yolo_padding = cfg["yolo_padding"]
        self.margin_top = cfg["margin_top"]
        self.margin_bot = cfg["margin_bot"]
        self.min_texts = cfg["min_texts"]

    # ----- public API -----

    def apply(
        self,
        dt_boxes: List[np.ndarray],
        dt_scores: List[float],
        img_hw: Tuple[int, int],
        yolo_detections: Optional[List[Dict]],
    ) -> Tuple[List[np.ndarray], Dict]:
        """Apply all filter stages in order.

        Returns:
            (filtered_boxes, stats)
        """
        H, W = img_hw
        n_total = len(dt_boxes)
        stats = {
            "total": n_total,
            "passed": 0,
            "filtered_by_size": 0,
            "filtered_by_conf": 0,
            "filtered_by_roi": 0,
            "margin_added": 0,
            "loss_ratio": 0.0,
            "trigger_min_fallback": False,
        }

        if n_total == 0:
            return [], stats

        # Stage 1: Size filter
        boxes, scores, n_size = self._filter_size(dt_boxes, dt_scores)
        stats["filtered_by_size"] = n_size

        # Stage 2: Confidence filter
        boxes, scores, n_conf = self._filter_conf(boxes, scores)
        stats["filtered_by_conf"] = n_conf

        # Stage 3: NMS dedup
        boxes, scores, n_nms_discard = self._nms_dedup(boxes, scores)
        # NMS discards counted in "by_conf" only, this is just structural dedup
        n_before_roi = len(boxes)

        # Stage 4: YOLO ROI filter
        if yolo_detections:
            boxes, scores, n_roi = self._filter_by_yolo_roi(boxes, scores, yolo_detections, W, H)
            stats["filtered_by_roi"] = n_roi

        # Stage 5: Add margin boxes (from the boxes that were filtered by ROI)
        if yolo_detections:
            boxes, scores, n_margin = self._add_margins(dt_boxes, dt_scores, boxes, W, H)
            stats["margin_added"] = n_margin

        # Fallback: too few boxes -> return all
        if len(boxes) < self.min_texts:
            stats["trigger_min_fallback"] = True
            boxes = dt_boxes

        stats["passed"] = len(boxes)
        stats["loss_ratio"] = round((n_total - stats["passed"]) / n_total, 4) if n_total > 0 else 0.0
        return boxes, stats

    # ----- internal stages -----

    def _box_height(self, box: np.ndarray) -> float:
        """Approximate height from polygon corners."""
        ys = box[:, 1]
        return float(np.max(ys) - np.min(ys))

    def _box_center(self, box: np.ndarray) -> Tuple[float, float]:
        return float(box[:, 0].mean()), float(box[:, 1].mean())

    def _filter_size(
        self, boxes: List[np.ndarray], scores: List[float]
    ) -> Tuple[List[np.ndarray], List[float], int]:
        kept_boxes, kept_scores = [], []
        removed = 0
        for box, score in zip(boxes, scores):
            h = self._box_height(box)
            if self.size_min_h <= h <= self.size_max_h:
                kept_boxes.append(box)
                kept_scores.append(score)
            else:
                removed += 1
        return kept_boxes, kept_scores, removed

    def _filter_conf(
        self, boxes: List[np.ndarray], scores: List[float]
    ) -> Tuple[List[np.ndarray], List[float], int]:
        kept_boxes, kept_scores = [], []
        removed = 0
        for box, score in zip(boxes, scores):
            if score >= self.conf_threshold:
                kept_boxes.append(box)
                kept_scores.append(score)
            else:
                removed += 1
        return kept_boxes, kept_scores, removed

    def _box_iou(self, a: np.ndarray, b: np.ndarray) -> float:
        """Approximate IOU from polygon bounding rectangles."""
        ax1, ay1 = a[:, 0].min(), a[:, 1].min()
        ax2, ay2 = a[:, 0].max(), a[:, 1].max()
        bx1, by1 = b[:, 0].min(), b[:, 1].min()
        bx2, by2 = b[:, 0].max(), b[:, 1].max()
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _nms_dedup(
        self, boxes: List[np.ndarray], scores: List[float]
    ) -> Tuple[List[np.ndarray], List[float], int]:
        """Greedy NMS — keep box with higher confidence."""
        if len(boxes) <= 1:
            return boxes, scores, 0
        # Sort by confidence descending
        idxs = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
        keep = []
        removed = 0
        while idxs:
            i = idxs.pop(0)
            keep.append(i)
            idxs = [j for j in idxs if self._box_iou(boxes[i], boxes[j]) < self.nms_iou]
        removed = len(boxes) - len(keep)
        return [boxes[i] for i in keep], [scores[i] for i in keep], removed

    def _filter_by_yolo_roi(
        self,
        boxes: List[np.ndarray],
        scores: List[float],
        yolo_dets: List[Dict],
        W: int, H: int,
    ) -> Tuple[List[np.ndarray], List[float], int]:
        """Keep only boxes whose center falls within expanded YOLO component region."""
        # Compute overall circuit region from all YOLO bboxes
        x_min, y_min = W, H
        x_max, y_max = 0, 0
        for det in yolo_dets:
            x, y, bw, bh = det["bbox"]
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + bw)
            y_max = max(y_max, y + bh)
        cw = x_max - x_min
        ch = y_max - y_min
        pad_x = int(cw * self.yolo_padding)
        pad_y = int(ch * self.yolo_padding)
        rx1 = max(0, x_min - pad_x)
        ry1 = max(0, y_min - pad_y)
        rx2 = min(W, x_max + pad_x)
        ry2 = min(H, y_max + pad_y)

        kept_boxes, kept_scores = [], []
        removed = 0
        for box, score in zip(boxes, scores):
            cx, cy = self._box_center(box)
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                kept_boxes.append(box)
                kept_scores.append(score)
            else:
                removed += 1
        return kept_boxes, kept_scores, removed

    def _add_margins(
        self,
        all_boxes: List[np.ndarray],
        all_scores: List[float],
        inside_boxes: List[np.ndarray],
        W: int, H: int,
    ) -> Tuple[List[np.ndarray], List[float], int]:
        """Add back boxes in top/bottom margin strips (titles, captions)."""
        top_h = int(H * self.margin_top)
        bot_h = int(H * self.margin_bot)
        inside_set = set(id(b) for b in inside_boxes)
        added = 0
        result_boxes = list(inside_boxes)

        for box, score in zip(all_boxes, all_scores):
            if id(box) in inside_set:
                continue
            cx, cy = self._box_center(box)
            if cy < top_h or cy > H - bot_h:
                result_boxes.append(box)
                added += 1

        return result_boxes, [], added
