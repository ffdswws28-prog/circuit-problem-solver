"""中文路径兼容的图片读写工具"""
import numpy as np
import cv2
from pathlib import Path


def imread(path: str) -> np.ndarray:
    """读取图片，兼容中文路径"""
    data = np.fromfile(str(Path(path)), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def imwrite(path: str, img: np.ndarray):
    """写入图片，兼容中文路径"""
    ext = Path(path).suffix
    success, encoded = cv2.imencode(ext, img)
    if not success:
        raise IOError(f"Cannot encode image for: {path}")
    encoded.tofile(str(Path(path)))
