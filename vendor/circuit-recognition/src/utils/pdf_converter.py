"""PyMuPDF PDF → 图片转换器"""
import fitz  # PyMuPDF
import numpy as np
from pathlib import Path
from typing import List


class PDFConverter:
    """PDF 转图片 — 支持多页逐页渲染、空白页过滤"""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi

    def convert(self, pdf_path: str, output_dir: str = None) -> List[np.ndarray]:
        """将 PDF 逐页渲染为图片列表"""
        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))
        images = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            # 渲染为 pixmap
            zoom = self.dpi / 72.0  # PDF default is 72 DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img = img.copy()  # make writable

            # 空白页过滤
            if self._is_blank(img):
                continue
            images.append(img)

        doc.close()
        return images

    def _is_blank(self, img: np.ndarray, threshold: float = 10.0) -> bool:
        """判断是否为空白页（像素方差低于阈值）"""
        gray = img.mean(axis=2) if img.ndim == 3 else img
        return gray.var() < threshold
