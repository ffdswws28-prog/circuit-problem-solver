"""ONNX Runtime 推理引擎 — OCR 主用，TensorRT 备选"""
import numpy as np
from typing import List


class ONNXEngine:
    """ONNX Runtime GPU 推理引擎"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None

    def load(self):
        """加载 ONNX 模型 + CUDA EP"""
        import onnxruntime as ort
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=providers)

    def infer(self, input_data: np.ndarray) -> List[np.ndarray]:
        """执行 ONNX 推理"""
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_data})
        return outputs
