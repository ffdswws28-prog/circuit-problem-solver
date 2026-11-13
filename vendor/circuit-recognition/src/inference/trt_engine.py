"""TensorRT 统一推理引擎 — GPU 端零拷贝"""
import numpy as np
from typing import List


class TRTEngine:
    """TensorRT 推理引擎，预分配显存缓冲区，零拷贝流水线"""

    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.device_inputs = []
        self.device_outputs = []

    def _load_engine(self):
        """加载 TensorRT engine 文件"""
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(self.engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def load(self):
        """加载并预分配显存"""
        self._load_engine()
        # 预分配显存缓冲区（复用，避免频繁申请）
        # TODO: 根据 engine 的 binding 信息预分配 CUDA 内存

    def infer(self, input_data: np.ndarray) -> List[np.ndarray]:
        """执行推理 — 全程异步"""
        # 1. 输入 → 设备（memcpy_htod_async）
        # 2. 推理（execute_async_v2）
        # 3. 输出 → 主机（memcpy_dtoh_async）
        # TODO: 实现 CUDA 流异步推理
        outputs = []
        return outputs
