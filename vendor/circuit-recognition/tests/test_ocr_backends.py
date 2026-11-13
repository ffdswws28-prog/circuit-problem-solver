"""Test OCR backend abstraction and implementations"""
import sys
sys.path.insert(0, r"D:\ai\circuit-recognition")
sys.path.insert(0, r"D:\ai\circuit-recognition\src")

import pytest
import numpy as np
import cv2
from src.pipeline.backends.base import OcrBackendBase, BackendInitError


class TestOcrBackendBase:
    """Base class contract tests"""

    def test_cannot_instantiate_abstract(self):
        """Abstract class should not be directly instantiated"""
        with pytest.raises(TypeError):
            OcrBackendBase()

    def test_concrete_subclass_must_implement_all_abstracts(self):
        """Missing abstract methods -> TypeError on instantiation"""
        class Incomplete(OcrBackendBase):
            pass
        with pytest.raises(TypeError):
            Incomplete()

    def test_minimal_concrete_subclass_instantiates(self):
        """Full implementation should instantiate"""
        class Minimal(OcrBackendBase):
            def initialize(self): pass
            def release(self): pass
            @property
            def is_available(self): return True
            def detect(self, image): return {"boxes": [], "scores": [], "cost_sec": 0.0, "img_hw": (0, 0)}
            def recognize_batch(self, image, boxes): return []
        backend = Minimal()
        assert backend.is_available is True


class TestBackendInitError:
    """Custom exception behavior"""

    def test_is_exception(self):
        err = BackendInitError("test failure")
        assert isinstance(err, Exception)

    def test_message_preserved(self):
        msg = "TRT engine not found"
        err = BackendInitError(msg)
        assert str(err) == msg


class TestRapidOcrBackend:
    """Test RapidOCR fallback backend"""

    def test_initialize_and_release(self):
        from src.pipeline.backends.rapidocr_backend import RapidOcrBackend
        backend = RapidOcrBackend()
        backend.initialize()
        assert backend.is_available is True
        backend.release()
        assert backend.is_available is False

    def test_detect_returns_structured_dict(self):
        """detect must return dict with required keys, even on empty image"""
        from src.pipeline.backends.rapidocr_backend import RapidOcrBackend
        backend = RapidOcrBackend()
        backend.initialize()
        # Use a small synthetic image with some text-like features
        import numpy as np
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(img, "R1 10k", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        result = backend.detect(img)
        assert "boxes" in result
        assert "scores" in result
        assert "cost_sec" in result
        assert "img_hw" in result
        assert result["img_hw"] == (300, 400)
        assert len(result["boxes"]) == len(result["scores"])
        assert result["cost_sec"] > 0
        backend.release()

    def test_recognize_batch_returns_same_length(self):
        """recognize_batch must return list same length as input boxes"""
        from src.pipeline.backends.rapidocr_backend import RapidOcrBackend
        backend = RapidOcrBackend()
        backend.initialize()
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.putText(img, "R1", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        # Create a synthetic box
        box = np.array([[80, 70], [140, 70], [140, 100], [80, 100]], dtype=np.float32)
        results = backend.recognize_batch(img, [box])
        assert len(results) == 1
        assert "text" in results[0]
        assert "confidence" in results[0]
        assert "bbox" in results[0]
        backend.release()

    def test_recognize_batch_empty_boxes(self):
        from src.pipeline.backends.rapidocr_backend import RapidOcrBackend
        backend = RapidOcrBackend()
        backend.initialize()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        results = backend.recognize_batch(img, [])
        assert results == []
        backend.release()

    def test_recognize_batch_failed_box_fills_empty(self):
        """Failed recognition should return empty text, not None"""
        from src.pipeline.backends.rapidocr_backend import RapidOcrBackend
        backend = RapidOcrBackend()
        backend.initialize()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Box outside image bounds — will fail to crop
        bad_box = np.array([[500, 500], [600, 500], [600, 600], [500, 600]], dtype=np.float32)
        results = backend.recognize_batch(img, [bad_box])
        assert len(results) == 1
        assert results[0] is not None
        assert results[0]["text"] == "" or isinstance(results[0]["text"], str)
        backend.release()


class TestTrtOcrBackendFallback:
    """When TRT is unavailable, TrtOcrBackend should report unavailable"""

    def test_unavailable_when_no_trt(self):
        from src.pipeline.backends.trt_backend import TrtOcrBackend
        backend = TrtOcrBackend()
        # Without TRT installed, initialize should not raise,
        # but is_available should be False
        try:
            backend.initialize()
        except Exception as e:
            # BackendInitError is acceptable too
            from src.pipeline.backends.base import BackendInitError
            if not isinstance(e, BackendInitError):
                raise
        assert backend.is_available is False

    def test_release_when_unavailable(self):
        from src.pipeline.backends.trt_backend import TrtOcrBackend
        backend = TrtOcrBackend()
        backend.release()
        assert backend.is_available is False

    def test_detect_when_unavailable(self):
        """When unavailable, detect returns empty with cost=0"""
        from src.pipeline.backends.trt_backend import TrtOcrBackend
        import numpy as np
        backend = TrtOcrBackend()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = backend.detect(img)
        assert result["boxes"] == []
        assert result["cost_sec"] == 0.0

    def test_recognize_batch_when_unavailable(self):
        """When unavailable, recognize_batch returns empty for each box"""
        from src.pipeline.backends.trt_backend import TrtOcrBackend
        import numpy as np
        backend = TrtOcrBackend()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        box = np.zeros((4, 2), dtype=np.float32)
        results = backend.recognize_batch(img, [box])
        assert len(results) == 1
        assert results[0]["text"] == ""


class TestOcrBackendIntegration:
    """End-to-end backend integration tests"""

    def test_engine_backward_compat(self):
        """Original OCREngine usage still works"""
        from src.pipeline.ocr_engine import OCREngine
        import numpy as np

        engine = OCREngine(use_easyocr=False, enable_correction=True)
        engine.load_model()

        # Test with a small synthetic image containing text
        import cv2
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.putText(img, "R1 10k", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        results = engine.detect_text_regions(img)
        assert isinstance(results, list)

        # recognize_all should be same as detect_text_regions
        results2 = engine.recognize_all(img)
        assert isinstance(results2, list)

    def test_recognize_regions(self):
        from src.pipeline.ocr_engine import OCREngine
        import numpy as np
        import cv2

        engine = OCREngine(use_easyocr=False, enable_correction=True)
        engine.load_model()

        img = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.putText(img, "U3", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Create an ROI around the text
        results = engine.recognize_regions(img, [[80, 70, 80, 60]])
        assert isinstance(results, list)

    def test_recognize_near_components_basic(self):
        from src.pipeline.ocr_engine import OCREngine
        import numpy as np
        import cv2

        engine = OCREngine(use_easyocr=False, enable_correction=True)
        engine.load_model()

        img = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(img, "R1", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(img, "10k", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Fake YOLO detections near the text
        yolo_dets = [
            {"bbox": [40, 120, 60, 50], "class": "resistor", "confidence": 0.9},
        ]

        results = engine.recognize_near_components(img, yolo_dets)
        assert isinstance(results, list)
