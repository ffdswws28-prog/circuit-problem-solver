"""Stage 3: OCR text detection + recognition (backend-driven, multi-backend)"""
import sys
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import yaml

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.pipeline.backends.base import OcrBackendBase, BackendInitError
from src.pipeline.trace_profiler import TraceProfiler


def _load_config(config_path: Optional[str] = None) -> Dict:
    """Load ocr_config.yaml, return full config dict."""
    if config_path is None:
        config_path = str(_project_root / "config" / "ocr_config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


class OCREngine:
    """OCR engine for circuit text recognition.

    Public API (unchanged from original):
      - load_model()
      - detect_text_regions(image) -> List[Dict]
      - recognize_regions(image, roi_list) -> List[Dict]
      - recognize_all(image) -> List[Dict]
      - recognize_near_components(image, yolo_detections, ...) -> List[Dict]
    """

    CIRCUIT_CHARS = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.ΩkMμFn pHVAW/-+()")

    def __init__(
        self,
        use_easyocr: bool = False,
        enable_correction: bool = True,
        config_path: Optional[str] = None,
    ):
        self.use_easyocr = use_easyocr
        self.enable_correction = enable_correction
        self.reader = None         # EasyOCR fallback (preserved)
        self.corrector = None

        # New architecture
        self._config = _load_config(config_path)
        self._backend: Optional[OcrBackendBase] = None
        self._filter = None
        self._profiler: Optional[TraceProfiler] = None

    # ================ lifecycle ================

    def load_model(self):
        """Auto-select backend → initialize → load corrector."""
        if self.use_easyocr:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=True)
            print("[OCR] EasyOCR loaded (GPU)")
        else:
            self._backend = self._backend_factory()
            # Load SmartFilter (shared between backends)
            from src.pipeline.smart_filter import SmartFilter
            self._filter = SmartFilter(self._config.get("filter"))
            # Load profiler
            trace_cfg = self._config.get("trace", {})
            self._profiler = TraceProfiler(self._backend.__class__.__name__)

        # Load circuit text corrector (shared)
        if self.enable_correction:
            from src.circuit_text_corrector import CircuitTextCorrector
            self.corrector = CircuitTextCorrector()
            print("[OCR] Circuit text corrector loaded")

    def _backend_factory(self) -> OcrBackendBase:
        """Select backend based on config and environment availability."""
        backend_choice = self._config.get("backend", {}).get("default", "auto")

        if backend_choice == "rapidocr":
            return self._init_rapidocr()

        if backend_choice == "trt":
            trt = self._try_init_trt()
            if trt is not None:
                return trt
            print("[OCR] TRT backend requested but unavailable, falling back to RapidOCR")
            return self._init_rapidocr()

        # auto: try TRT first, fallback to RapidOCR
        trt = self._try_init_trt()
        if trt is not None:
            return trt
        return self._init_rapidocr()

    def _try_init_trt(self) -> Optional[OcrBackendBase]:
        try:
            from src.pipeline.backends.trt_backend import TrtOcrBackend
            backend = TrtOcrBackend(self._config)
            backend.initialize()
            if backend.is_available:
                print("[OCR] TrtOcrBackend selected")
                return backend
        except BackendInitError as e:
            print(f"[OCR] TRT init failed: {e}")
        except ImportError:
            print("[OCR] TRT modules not available")
        except Exception as e:
            print(f"[OCR] TRT unexpected error: {e}")
        return None

    def _init_rapidocr(self) -> OcrBackendBase:
        from src.pipeline.backends.rapidocr_backend import RapidOcrBackend
        backend = RapidOcrBackend()
        backend.initialize()
        print("[OCR] RapidOcrBackend selected (fallback)")
        return backend

    # ================ public API ================

    def detect_text_regions(self, image: np.ndarray) -> List[Dict]:
        """Full-image detection + recognition (backward compatible)."""
        if self.reader:
            return self._easyocr_detect(image)

        with self._profiler.trace("full_ocr"):
            det_res = self._backend.detect(image)
            if not det_res.get("boxes"):
                return []
            results = self._backend.recognize_batch(image, det_res["boxes"])

        if self.corrector:
            results = self.corrector.correct(results)
        return results

    def recognize_regions(self, image: np.ndarray, roi_list: List[List[int]]) -> List[Dict]:
        """Recognize text in specific ROIs (backward compatible)."""
        results = []
        for roi in roi_list:
            x, y, w, h = roi
            crop = image[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            if self.reader:
                ocr_results = self.reader.readtext(crop)
                if ocr_results:
                    text = " ".join(r[1] for r in ocr_results)
                    conf = max(r[2] for r in ocr_results)
                    results.append({"text": text, "confidence": conf, "bbox": roi})
            elif self._backend:
                box = np.array([
                    [x, y], [x + w, y], [x + w, y + h], [x, y + h]
                ], dtype=np.float32)
                recs = self._backend.recognize_batch(crop, [box])
                if recs:
                    results.append(recs[0])
        return results

    def recognize_all(self, image: np.ndarray) -> List[Dict]:
        """Full-image OCR (backward compatible)."""
        return self.detect_text_regions(image)

    def recognize_near_components(
        self,
        image: np.ndarray,
        yolo_detections: List[Dict],
        padding_ratio: float = 0.4,
        min_texts: int = 4,
    ) -> List[Dict]:
        """Optimized: detect → SmartFilter → batch_rec → correct."""
        if self.reader:
            return self.detect_text_regions(image)

        # Pass through explicit params to SmartFilter (override config defaults)
        if padding_ratio != 0.4 or min_texts != 4:
            from src.pipeline.smart_filter import SmartFilter
            if self._filter is None or not isinstance(self._filter, SmartFilter):
                self._filter = SmartFilter(self._config.get("filter"))
            self._filter.yolo_padding = padding_ratio
            self._filter.min_texts = min_texts

        h, w = image.shape[:2]

        with self._profiler.trace("recognize_near_components"):
            # 1. Detect
            with self._profiler.trace("det"):
                det_res = self._backend.detect(image)

            if not det_res.get("boxes"):
                return []

            # 2. Filter
            with self._profiler.trace("filter"):
                filtered_boxes, stats = self._filter.apply(
                    dt_boxes=det_res["boxes"],
                    dt_scores=det_res["scores"],
                    img_hw=det_res["img_hw"],
                    yolo_detections=yolo_detections,
                )
            self._profiler.record_filter_loss(stats["total"], stats["passed"])
            if stats.get("trigger_min_fallback"):
                self._profiler.record_fallback(1)

            # 3. Recognize
            with self._profiler.trace("rec_batch"):
                results = self._backend.recognize_batch(image, filtered_boxes)

        # 4. Correct
        if self.corrector and results:
            with self._profiler.trace("corrector"):
                try:
                    results = self.corrector.correct(results)
                except Exception:
                    pass

        return results

    # ================ compatibility helpers ================

    def _easyocr_detect(self, image: np.ndarray) -> List[Dict]:
        results = self.reader.readtext(image)
        items = [
            {"bbox": self._points_to_bbox(r[0]), "text": r[1], "confidence": r[2]}
            for r in results
        ]
        if self.corrector:
            items = self.corrector.correct(items)
        return items

    def release(self):
        """Release GPU resources held by the backend."""
        if self._backend:
            self._backend.release()
            self._backend = None
        self._filter = None
        self._profiler = None

    @property
    def backend_type(self) -> str:
        if self._backend:
            return type(self._backend).__name__
        return "easyocr" if self.reader else "none"

    @staticmethod
    def _points_to_bbox(points) -> List[int]:
        xs = [int(p[0]) for p in points]
        ys = [int(p[1]) for p in points]
        x, y = min(xs), min(ys)
        w, h = max(xs) - x, max(ys) - y
        return [x, y, w, h]
