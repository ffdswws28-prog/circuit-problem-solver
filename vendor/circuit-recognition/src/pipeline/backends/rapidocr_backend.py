"""RapidOCR (ONNX Runtime) backend — the primary fallback implementation."""
import time
import numpy as np
from typing import List, Dict
from pathlib import Path
import sys

# Ensure project root in path
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from .base import OcrBackendBase, BackendInitError


class RapidOcrBackend(OcrBackendBase):
    """Wraps rapidocr_onnxruntime.RapidOCR behind the OcrBackendBase contract."""

    def __init__(self):
        self._ocr = None
        self._available = False

    # ---- lifecycle ----

    def initialize(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._ocr = RapidOCR()
            self._available = True
            print("[RapidOcrBackend] Initialized (ONNX Runtime)")
        except Exception as e:
            raise BackendInitError(f"RapidOCR init failed: {e}")

    def release(self) -> None:
        self._ocr = None
        self._available = False

    @property
    def is_available(self) -> bool:
        return self._available and self._ocr is not None

    # ---- detect ----

    def detect(self, image: np.ndarray) -> Dict:
        if not self.is_available:
            return {"boxes": [], "scores": [], "cost_sec": 0.0, "img_hw": image.shape[:2]}

        h, w = image.shape[:2]
        t0 = time.perf_counter()

        # Run detection only (no recognition)
        det_result, _ = self._ocr(image, use_rec=False)

        cost = time.perf_counter() - t0

        if det_result is None or len(det_result) == 0:
            return {"boxes": [], "scores": [], "cost_sec": cost, "img_hw": (h, w)}

        boxes = []
        scores = []
        for item in det_result:
            if item is None:
                continue
            # RapidOCR use_rec=False still returns (poly, text, score_str)
            # where score_str is a string like "0.8339"
            if len(item) >= 3:
                poly = np.array(item[0], dtype=np.float32)
                try:
                    score = float(item[2])
                except (ValueError, TypeError):
                    score = 1.0
                boxes.append(poly)
                scores.append(score)

        return {"boxes": boxes, "scores": scores, "cost_sec": cost, "img_hw": (h, w)}

    # ---- recognize_batch ----

    def recognize_batch(
        self, image: np.ndarray, boxes: List[np.ndarray]
    ) -> List[Dict]:
        """Recognize text in each box. Uses text_recognizer batch API."""
        if not self.is_available:
            return []

        # Collect all crops first, then batch-recognize
        valid_indices = []
        crops = []
        for i, box in enumerate(boxes):
            try:
                crop = self._ocr.get_crop_img_list(image, [box])[0]
                crops.append(crop)
                valid_indices.append(i)
            except Exception:
                pass

        rec_results = []
        if crops and hasattr(self._ocr, "text_recognizer"):
            try:
                rec_results, _ = self._ocr.text_recognizer(crops)
            except Exception:
                rec_results = []
        elif crops and hasattr(self._ocr, "text_rec"):
            # Legacy serial fallback (some older versions)
            rec_results = []
            for crop in crops:
                try:
                    text, conf = self._ocr.text_rec(crop)
                    rec_results.append((str(text) if text else "", str(conf) if conf else "0.0"))
                except Exception:
                    rec_results.append(("", "0.0"))

        # Build results in original box order
        results = []
        result_map = {}
        for idx_in_group, original_idx in enumerate(valid_indices):
            if idx_in_group < len(rec_results) and rec_results[idx_in_group] is not None:
                text, conf = rec_results[idx_in_group]
                if not text or str(text).strip() == "":
                    text = ""
                conf = float(conf) if conf else 0.0
            else:
                text = ""
                conf = 0.0
            bbox = self._poly_to_bbox(boxes[original_idx])
            result_map[original_idx] = {"text": str(text) if text else "",
                                        "confidence": float(conf) if conf else 0.0,
                                        "bbox": bbox}

        for i in range(len(boxes)):
            if i in result_map:
                results.append(result_map[i])
            else:
                results.append({"text": "", "confidence": 0.0,
                                "bbox": self._poly_to_bbox(boxes[i])})

        return results

    # ---- helpers ----

    @staticmethod
    def _poly_to_bbox(poly: np.ndarray) -> list:
        xs = [int(p[0]) for p in poly]
        ys = [int(p[1]) for p in poly]
        x, y = min(xs), min(ys)
        w, h = max(xs) - x, max(ys) - y
        return [x, y, w, h]
