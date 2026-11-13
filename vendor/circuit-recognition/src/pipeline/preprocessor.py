"""Stage 0: GPU 统一预处理 — 读文件、解码、resize、归一化"""
import cv2
import numpy as np
from pathlib import Path


class Preprocessor:
    """统一预处理，输出共享 GPU 张量，零拷贝分发"""

    def __init__(self, target_size=(640, 640)):
        self.target_size = target_size

    def load_image(self, image_path: str) -> np.ndarray:
        """加载图片，兼容中文路径"""
        if isinstance(image_path, str):
            image_path = Path(image_path)
        data = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return img

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize + 归一化，输出模型就绪的张量"""
        resized = cv2.resize(image, self.target_size)
        # 归一化到 [0, 1]，HWC -> CHW
        tensor = resized.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))  # CHW
        tensor = np.expand_dims(tensor, axis=0)     # BCHW
        return tensor
