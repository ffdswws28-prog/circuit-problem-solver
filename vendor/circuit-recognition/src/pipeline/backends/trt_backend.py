"""TensorRT-accelerated OCR backend.

Aggregates:
  - TrtDetector (text detection, optional per config)
  - SmartFilter (multi-strategy box filter)
  - TrtRecognizer (batched text recognition)
  - TraceProfiler (structured timing)

When TRT is unavailable, initialize() raises BackendInitError,
triggering OCREngine's fallback to RapidOcrBackend.
"""
import time
import sys
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from .base import OcrBackendBase, BackendInitError


class TrtOcrBackend(OcrBackendBase):
    """TensorRT OCR backend -- primary acceleration path."""

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        trt_cfg = cfg.get("trt", {})
        self._det_enabled = trt_cfg.get("det_enabled", False)
        self._precision = trt_cfg.get("precision", "fp16")
        self._engine_dir = trt_cfg.get("engine_cache_dir", "models/trt/")
        self._available = False

        # Sub-modules (created in initialize)
        self._trt_detector = None
        self._trt_recognizer = None
        self._smart_filter = None
        self._profiler = None

        # Fallback components
        self._rapid_det = None   # RapidOCR instance for det when TRT det disabled
        self._config = config or {}

    # ---- lifecycle ----

    def initialize(self) -> None:
        try:
            # 1. Load SmartFilter
            from src.pipeline.smart_filter import SmartFilter
            self._smart_filter = SmartFilter(self._config.get("filter"))

            # 2. Load profiler
            from src.pipeline.trace_profiler import TraceProfiler
            self._profiler = TraceProfiler("trt")

            # 3. Load TrtDetector (optional)
            if self._det_enabled:
                from src.pipeline.trt_detector import TrtDetector
                det_path = str(Path(self._engine_dir) / "det.engine")
                self._trt_detector = TrtDetector(
                    engine_path=det_path, precision=self._precision
                )
                if not self._trt_detector.initialize():
                    print("[TrtOcrBackend] TRT det failed, will use ONNX det")

            # 4. Load TrtRecognizer
            from src.pipeline.trt_recognizer import TrtRecognizer
            rec_path = str(Path(self._engine_dir) / "rec.engine")
            rec_step = self._config.get("trt", {}).get("rec_batch_group_step", 64)
            self._trt_recognizer = TrtRecognizer(
                engine_path=rec_path,
                precision=self._precision,
                batch_group_step=rec_step,
            )
            rec_ok = self._trt_recognizer.initialize()
            if not rec_ok:
                raise BackendInitError("TRT rec engine failed to initialize")

            # 5. Load RapidOCR for detection (when TRT det is disabled)
            from rapidocr_onnxruntime import RapidOCR
            self._rapid_det = RapidOCR()
            self._available = True
            print("[TrtOcrBackend] Initialized (det=ONNX, rec=TRT)")

        except BackendInitError:
            raise
        except Exception as e:
            raise BackendInitError(f"TrtOcrBackend init failed: {e}")

    def release(self) -> None:
        self._trt_detector = None
        self._trt_recognizer = None
        self._rapid_det = None
        self._smart_filter = None
        self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    # ---- detect (NON-TRT BY DEFAULT) ----

    def detect(self, image: np.ndarray) -> Dict:
        h, w = image.shape[:2]
        t0 = time.perf_counter()

        if self._trt_detector and self._trt_detector.is_available:
            result = self._trt_detector.detect(image)
        elif self._rapid_det:
            det_result, _ = self._rapid_det(image, use_rec=False)
            boxes, scores = [], []
            if det_result:
                for item in det_result:
                    if item is None:
                        continue
                    poly = np.array(item[0], dtype=np.float32)
                    score = float(item[2]) if len(item) > 2 and item[2] is not None else 1.0
                    boxes.append(poly)
                    scores.append(score)
            result = {
                "boxes": boxes,
                "scores": scores,
                "cost_sec": time.perf_counter() - t0,
                "img_hw": (h, w),
            }
        else:
            result = {"boxes": [], "scores": [], "cost_sec": 0.0, "img_hw": (h, w)}

        return result

    # ---- recognize_batch ----

    def recognize_batch(
        self, image: np.ndarray, boxes: List[np.ndarray]
    ) -> List[Dict]:
        if not self.is_available or not self._trt_recognizer:
            return self._fallback_rec_batch(image, boxes)

        try:
            return self._trt_recognizer.recognize_batch(image, boxes)
        except Exception as e:
            print(f"[TrtOcrBackend] Batch rec failed: {e}, falling back")
            return self._fallback_rec_batch(image, boxes)

    def _fallback_rec_batch(
        self, image: np.ndarray, boxes: List[np.ndarray]
    ) -> List[Dict]:
        """Fallback: serial recognition via RapidOCR."""
        results = []
        for box in boxes:
            try:
                if self._rapid_det:
                    crop = self._rapid_det.get_crop_img_list(image, [box])[0]
                    text, conf = self._rapid_det.text_rec(crop)
                    results.append({
                        "text": str(text) if text else "",
                        "confidence": float(conf) if conf else 0.0,
                        "bbox": TrtOcrBackend._poly_to_bbox(box),
                    })
                else:
                    results.append(self._empty_result(box))
            except Exception:
                results.append(self._empty_result(box))
        return results

    # ---- helpers ----

    @property
    def smart_filter(self):
        return self._smart_filter

    @property
    def profiler(self):
        return self._profiler

    @staticmethod
    def _poly_to_bbox(poly: np.ndarray) -> list:
        xs = [int(p[0]) for p in poly]
        ys = [int(p[1]) for p in poly]
        x, y = min(xs), min(ys)
        w, h = max(xs) - x, max(ys) - y
        return [x, y, w, h]

    @staticmethod
    def _empty_result(box: np.ndarray) -> Dict:
        return {"text": "", "confidence": 0.0, "bbox": TrtOcrBackend._poly_to_bbox(box)}
