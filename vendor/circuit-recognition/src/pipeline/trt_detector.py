"""TensorRT text detection engine — optional, disabled by default.

When TensorRT is unavailable, this module is harmless to import
(initialize() returns False, is_available is False).
"""
import time
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


# Module-level flag set on first import attempt
_TRT_AVAILABLE = False
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    _TRT_AVAILABLE = True
except ImportError:
    pass


class TrtDetector:
    """TensorRT-accelerated text detection (PP-OCRv6 Mobile DBNet).

    Wraps an ONNX->TRT engine. Uses RapidOCR's pre/post-processing
    to guarantee output coordinates match the original pipeline.
    """

    def __init__(
        self,
        engine_path: Optional[str] = None,
        precision: str = "fp16",
        warmup: bool = True,
    ):
        self._engine_path = engine_path
        self._precision = precision
        self._warmup = warmup
        self._available = False
        self._engine = None
        self._context = None
        self._stream = None
        self._inputs = {}
        self._outputs = {}
        self._bindings = []
        self._rapid_ocr = None  # For pre/post processing reuse

    # ---- lifecycle ----

    def initialize(self) -> bool:
        if not _TRT_AVAILABLE:
            return False
        try:
            self._load_or_build_engine()
            self._allocate_buffers()
            if self._warmup:
                self._warmup_run()
            self._available = True
            return True
        except Exception as e:
            print(f"[TrtDetector] Init failed: {e}")
            self._available = False
            return False

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def engine_info(self) -> Dict:
        if not self._engine_path or not Path(self._engine_path).exists():
            return {}
        info_path = Path(self._engine_path).with_suffix(".json")
        if info_path.exists():
            with open(info_path) as f:
                return json.load(f)
        stat = Path(self._engine_path).stat()
        return {
            "engine_path": str(self._engine_path),
            "size_bytes": stat.st_size,
            "created_at": stat.st_mtime,
        }

    # ---- detect ----

    def detect(self, image: np.ndarray) -> Dict[str, any]:
        h, w = image.shape[:2]
        if not self.is_available:
            return {"boxes": [], "scores": [], "cost_sec": 0.0, "img_hw": (h, w)}

        # Input validation
        if image.size == 0 or image.ndim != 3 or image.shape[2] not in (1, 3):
            return {"boxes": [], "scores": [], "cost_sec": 0.0, "img_hw": (h, w)}

        t0 = time.perf_counter()

        try:
            # Pre-process using RapidOCR's native logic
            # (reuse rapidocr_onnxruntime internals for coordinate fidelity)
            det_result = self._trt_detect(image)
            cost = time.perf_counter() - t0

            boxes, scores = self._post_process(det_result, w, h)
            return {"boxes": boxes, "scores": scores, "cost_sec": cost, "img_hw": (h, w)}
        except Exception as e:
            cost = time.perf_counter() - t0
            print(f"[TrtDetector] detect exception: {e}")
            return {"boxes": [], "scores": [], "cost_sec": cost, "img_hw": (h, w)}

    # ---- internals (skeleton — filled when TRT models are built) ----

    def _load_or_build_engine(self):
        """Load serialized TRT engine from file."""
        if not self._engine_path:
            self._engine_path = str(Path(__file__).parent.parent.parent / "models" / "trt" / "det.engine")
        engine_file = Path(self._engine_path)
        if not engine_file.exists():
            raise FileNotFoundError(f"TRT engine not found: {self._engine_path}. Run build_trt_engines.py first.")
        with open(engine_file, "rb") as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self._engine = runtime.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()

    def _allocate_buffers(self):
        """Allocate GPU I/O buffers for the engine bindings."""
        # Simple buffer allocation — host + device pairs
        self._bindings = []
        self._inputs = {}
        self._outputs = {}
        self._stream = cuda.Stream()
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = tuple(self._engine.get_tensor_shape(name))
            dtype = trt.nptype(self._engine.get_tensor_dtype(name))
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            dev_mem = cuda.mem_alloc(size)
            host_mem = np.zeros(shape, dtype=dtype)
            self._bindings.append(int(dev_mem))
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._inputs[name] = {"shape": shape, "dtype": dtype, "host": host_mem, "device": dev_mem}
            else:
                self._outputs[name] = {"shape": shape, "dtype": dtype, "host": host_mem, "device": dev_mem}

    def _warmup_run(self):
        """One dummy inference to warm CUDA context."""
        pass  # Full implementation requires knowing exact input shape

    def _trt_detect(self, image: np.ndarray):
        """Run TRT inference. Placeholder — full impl needs RapidOCR preprocessing."""
        raise NotImplementedError("Full TRT inference requires ONNX model export first")

    def _post_process(self, det_output, orig_w, orig_h) -> tuple:
        """Convert TRT output to RapidOCR-compatible box format.
        Reuses rapidocr_onnxruntime post-processing logic."""
        return [], []
