"""环境验证脚本 — 检查所有依赖是否正确安装"""
import sys
import os

def test_python():
    """检查 Python 版本"""
    assert sys.version_info[:2] == (3, 11), f"Expected Python 3.11, got {sys.version}"
    print(f"[OK] Python {sys.version}")

def test_pytorch():
    """检查 PyTorch + CUDA"""
    import torch
    print(f"[OK] PyTorch {torch.__version__}")
    assert torch.cuda.is_available(), "CUDA not available!"
    print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
    x = torch.randn(3, 3).cuda()
    print(f"[OK] GPU tensor operation OK")

def test_tensorrt():
    """检查 TensorRT"""
    try:
        import tensorrt as trt
        print(f"[OK] TensorRT {trt.__version__}")
    except ImportError as e:
        print(f"[WARN] TensorRT import failed: {e}")

def test_ultralytics():
    """检查 YOLO"""
    import ultralytics
    print(f"[OK] ultralytics {ultralytics.__version__}")

def test_paddleocr():
    """检查 PaddleOCR"""
    try:
        from paddleocr import PaddleOCR
        print(f"[OK] PaddleOCR import OK")
    except ImportError as e:
        print(f"[WARN] PaddleOCR import failed: {e}")

def test_opencv():
    """检查 OpenCV"""
    import cv2
    print(f"[OK] OpenCV {cv2.__version__}")

def test_onnx():
    """检查 ONNX Runtime"""
    import onnxruntime as ort
    print(f"[OK] ONNX Runtime {ort.__version__}")
    providers = ort.get_available_providers()
    print(f"     Providers: {providers}")

def test_sam():
    """检查 SAM"""
    try:
        from segment_anything import sam_model_registry
        print(f"[OK] Segment Anything import OK")
    except ImportError as e:
        print(f"[WARN] SAM import failed: {e}")

def test_mcp():
    """检查 MCP SDK"""
    import mcp
    print(f"[OK] MCP SDK import OK")

def test_pymupdf():
    """检查 PyMuPDF"""
    import fitz
    print(f"[OK] PyMuPDF {fitz.version}")

def test_disk_usage():
    """检查磁盘使用"""
    import shutil
    total, used, free = shutil.disk_usage("D:\\")
    print(f"[INFO] D drive: {free // (2**30)} GB free")

if __name__ == "__main__":
    print("=" * 50)
    print("Circuit Recognition System — Environment Check")
    print("=" * 50)

    tests = [
        ("Python 3.11", test_python),
        ("PyTorch + CUDA", test_pytorch),
        ("TensorRT", test_tensorrt),
        ("ultralytics (YOLO)", test_ultralytics),
        ("PaddleOCR", test_paddleocr),
        ("OpenCV", test_opencv),
        ("ONNX Runtime GPU", test_onnx),
        ("Segment Anything", test_sam),
        ("MCP SDK", test_mcp),
        ("PyMuPDF (PDF)", test_pymupdf),
        ("Disk Usage", test_disk_usage),
    ]

    passed = 0
    failed = 0
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            failed += 1

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
