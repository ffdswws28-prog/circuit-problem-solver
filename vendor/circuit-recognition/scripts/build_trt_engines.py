#!/usr/bin/env python
# scripts/build_trt_engines.py
"""Build TensorRT engines from RapidOCR's embedded ONNX models.

Usage:
    python scripts/build_trt_engines.py [--precision fp16|fp32] [--output models/trt/]

Extracts PP-OCRv6 Mobile detection and recognition ONNX models from
the installed rapidocr_onnxruntime package, converts them to TensorRT
engines, and caches them to disk.

Requires: tensorrt, pycuda
"""
import sys
import os
import json
import hashlib
import shutil
from pathlib import Path
from argparse import ArgumentParser

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def find_onnx_models():
    """Locate RapidOCR's ONNX model files in site-packages."""
    try:
        import rapidocr_onnxruntime
    except ImportError:
        print("  [WARN] rapidocr_onnxruntime not installed")
        return {}

    pkg_dir = Path(rapidocr_onnxruntime.__file__).parent

    models = {}
    # Standard locations in rapidocr_onnxruntime
    candidates = {
        "det": list(pkg_dir.glob("**/*det*.onnx")),
        "rec": list(pkg_dir.glob("**/*rec*.onnx")),
    }

    for name, paths in candidates.items():
        if paths:
            models[name] = str(paths[0])
            print(f"  Found {name}: {models[name]}")
        else:
            print(f"  [WARN] {name} model NOT FOUND in {pkg_dir}")

    return models


def build_engine(onnx_path: str, output_path: str, precision: str = "fp16"):
    """Convert ONNX model to TensorRT engine."""
    try:
        import tensorrt as trt
    except ImportError:
        print("  [SKIP] TensorRT not available — install with: pip install tensorrt")
        return False

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            return False

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print(f"  [fp16] enabled")
    elif precision == "fp32":
        print(f"  [fp32] mode")

    # Allow for dynamic batch if network has dynamic shapes
    profile = builder.create_optimization_profile()
    # (Dynamic shapes require knowing model input names — skipped for now)

    print(f"  Building engine... (this may take 2-5 minutes)")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print(f"  [ERROR] Build failed")
        return False

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(serialized)

    # Save metadata
    h = hashlib.sha256()
    with open(onnx_path, "rb") as f:
        h.update(f.read())
    meta = {
        "model_hash": h.hexdigest(),
        "trt_version": trt.__version__,
        "precision": precision,
        "onnx_path": onnx_path,
    }
    try:
        import torch
        meta["cuda_version"] = torch.version.cuda
    except ImportError:
        pass

    try:
        import pycuda.driver as cuda
        cuda.init()
        dev = cuda.Device(0)
        meta["gpu_name"] = dev.name()
        meta["driver_version"] = cuda.get_driver_version()
    except ImportError:
        pass

    with open(output_path.replace(".engine", ".json"), "w") as f:
        json.dump(meta, f, indent=2)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  [OK] {output_path} ({size_mb:.1f} MB)")
    return True


def main():
    parser = ArgumentParser(description="Build TensorRT engines from RapidOCR ONNX models")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--output", default="models/trt/")
    parser.add_argument("--det-only", action="store_true", help="Build det engine only")
    parser.add_argument("--rec-only", action="store_true", help="Build rec engine only")
    args = parser.parse_args()

    print("=" * 60)
    print("TensorRT Engine Builder for Circuit OCR")
    print("=" * 60)

    models = find_onnx_models()
    if not models:
        print("[ERROR] No RapidOCR ONNX models found. Is rapidocr_onnxruntime installed?")
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    success = True

    if not args.rec_only and "det" in models:
        print("\n--- Building Detection Engine ---")
        ok = build_engine(models["det"], str(out_dir / "det.engine"), args.precision)
        success = success and ok
    else:
        print("\n[SKIP] Detection engine")

    if not args.det_only and "rec" in models:
        print("\n--- Building Recognition Engine ---")
        ok = build_engine(models["rec"], str(out_dir / "rec.engine"), args.precision)
        success = success and ok
    else:
        print("\n[SKIP] Recognition engine")

    print("\n" + "=" * 60)
    if success:
        print("[DONE] All engines built successfully")
    else:
        print("[FAIL] Some engines failed to build")
        sys.exit(1)


if __name__ == "__main__":
    main()
