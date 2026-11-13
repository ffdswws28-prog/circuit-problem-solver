"""Quick test: RapidOCR on circuit diagram"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.ocr_engine import OCREngine
from src.utils.image_io import imread

img_path = Path(r"d:\桌面\ai\CGHD-1152\CGHD-1152\C1_D1_P1.jpg")
img = imread(str(img_path))

ocr = OCREngine(use_easyocr=False)
ocr.load_model()

t0 = time.time()
results = ocr.recognize_all(img)
elapsed = time.time() - t0

print(f"OCR 识别: {len(results)} 个文本区域")
print(f"耗时: {elapsed:.2f}s (原EasyOCR: ~163s)")
print(f"提速: {163/elapsed:.0f}x")
print()

for i, r in enumerate(results[:15]):
    orig = r.get("original")
    corr = f"  (原OCR: '{orig}')" if orig else ""
    print(f"  [{i}] text='{r['text']}' conf={r['confidence']:.3f}{corr}")
if len(results) > 15:
    print(f"  ... 共 {len(results)} 条")

print()
print("PASS" if elapsed < 10 and len(results) > 0 else "FAIL")
