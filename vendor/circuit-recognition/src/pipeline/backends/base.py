"""OCR backend abstraction layer."""
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any


class BackendInitError(Exception):
    """Raised when a backend fails to initialize.

    Caught by OCREngine to trigger automatic fallback to RapidOcrBackend.
    """
    pass


class OcrBackendBase(ABC):
    """Unified OCR backend interface.

    All OCR backends (RapidOCR, TensorRT) implement this contract.
    OCREngine depends only on this abstraction, never on concrete backends.
    """

    @abstractmethod
    def initialize(self) -> None:
        """Load model, build/validate engine, GPU warmup.

        Raises:
            BackendInitError: If model/engine cannot be loaded.
        """
        ...

    @abstractmethod
    def release(self) -> None:
        """Release GPU memory, model handles, file descriptors.

        Called on backend switch or process exit.
        """
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether the backend is ready for inference."""
        ...

    @abstractmethod
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """Pure text detection — no recognition.

        Args:
            image: BGR image array (H, W, 3).

        Returns:
            Dict with:
                "boxes": List[np.ndarray]  — each shape (N,4,2)
                "scores": List[float]       — per-box confidence
                "cost_sec": float           — detection wall-clock time
                "img_hw": tuple[int, int]   — original image (H, W)
        """
        ...

    @abstractmethod
    def recognize_batch(
        self, image: np.ndarray, boxes: List[np.ndarray]
    ) -> List[Dict]:
        """Batch text recognition — the ONLY recognition entry point.

        Args:
            image: BGR image array (H, W, 3).
            boxes: List of text detection polygons, each shape (N,4,2).

        Returns:
            List of dicts, same length as boxes. Each dict:
                {"text": str, "confidence": float, "bbox": list}
            Failed boxes fill: {"text": "", "confidence": 0.0, "bbox": box}
        """
        ...
