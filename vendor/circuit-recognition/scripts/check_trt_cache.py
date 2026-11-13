#!/usr/bin/env python
# scripts/check_trt_cache.py
"""Validate TRT engine cache — detect environment mismatch and trigger rebuild.

Usage:
    python scripts/check_trt_cache.py [--engines-dir models/trt/]
    python scripts/check_trt_cache.py --auto-rebuild

Exit codes:
    0 - All engines valid
    1 - Engine missing
    2 - Hash mismatch (environment changed)
    3 - No ONNX source found (cannot rebuild)
"""
import sys
import os
import json
import hashlib
from pathlib import Path
from argparse import ArgumentParser

sys.path.insert(0, str(Path(__file__).parent.parent))


def find_onnx_models():
    import rapidocr_onnxruntime
    pkg_dir = Path(rapidocr_onnxruntime.__file__).parent
    models = {}
    for name in ("det", "rec"):
        paths = list(pkg_dir.glob(f"**/*{name}*.onnx"))
        if paths:
            models[name] = str(paths[0])
    return models


def check_engine(engine_path: str) -> dict:
    """Validate a single engine file. Returns status dict."""
    ep = Path(engine_path)
    info_path = ep.with_suffix(".json")

    if not ep.exists():
        return {"status": "missing", "path": str(ep)}

    if not info_path.exists():
        return {"status": "no_metadata", "path": str(ep)}

    with open(info_path) as f:
        meta = json.load(f)

    # Check GPU driver match
    try:
        import pycuda.driver as cuda
        cuda.init()
        dev = cuda.Device(0)
        current_gpu = dev.name()
        current_driver = cuda.get_driver_version()
    except Exception:
        current_gpu = "unknown"
        current_driver = "unknown"

    stored_gpu = meta.get("gpu_name", "unknown")
    stored_driver = meta.get("driver_version", "unknown")

    if current_gpu != stored_gpu:
        return {
            "status": "gpu_mismatch",
            "path": str(ep),
            "stored_gpu": stored_gpu,
            "current_gpu": current_gpu,
        }

    # Check TRT version
    try:
        import tensorrt as trt
        current_trt = trt.__version__
    except ImportError:
        current_trt = "not_installed"

    if meta.get("trt_version") != current_trt:
        return {
            "status": "version_mismatch",
            "path": str(ep),
            "stored_trt": meta.get("trt_version"),
            "current_trt": current_trt,
        }

    # Check model hash (if ONNX source available)
    stored_hash = meta.get("model_hash")
    if stored_hash:
        models = find_onnx_models()
        engine_type = "det" if "det" in ep.name else "rec"
        onnx_path = models.get(engine_type)
        if onnx_path and Path(onnx_path).exists():
            h = hashlib.sha256()
            with open(onnx_path, "rb") as f:
                h.update(f.read())
            if h.hexdigest() != stored_hash:
                return {
                    "status": "hash_mismatch",
                    "path": str(ep),
                    "stored_hash": stored_hash[:16],
                    "current_hash": h.hexdigest()[:16],
                }

    return {"status": "ok", "path": str(ep), "size_mb": ep.stat().st_size / 1024 / 1024}


def main():
    parser = ArgumentParser(description="Check TRT engine cache validity")
    parser.add_argument("--engines-dir", default="models/trt/")
    parser.add_argument("--auto-rebuild", action="store_true")
    args = parser.parse_args()

    engines_dir = Path(args.engines_dir)
    all_ok = True

    for name in ("det.engine", "rec.engine"):
        engine_path = engines_dir / name
        result = check_engine(str(engine_path))
        status = result["status"]

        if status == "ok":
            print(f"[OK] {name} — {result['size_mb']:.1f} MB")
        elif status == "missing":
            print(f"[MISSING] {name} — not found at {result['path']}")
            all_ok = False
        elif status == "gpu_mismatch":
            print(f"[MISMATCH] {name} — GPU changed: {result['stored_gpu']} → {result['current_gpu']}")
            all_ok = False
        elif status == "version_mismatch":
            print(f"[MISMATCH] {name} — TRT version changed: {result['stored_trt']} → {result['current_trt']}")
            all_ok = False
        elif status == "hash_mismatch":
            print(f"[MISMATCH] {name} — model hash changed")
            all_ok = False
        else:
            print(f"[{status}] {name}")

    if not all_ok and args.auto_rebuild:
        print("\n[ACTION] Auto-rebuilding engines...")
        os.system(f"{sys.executable} scripts/build_trt_engines.py --output {args.engines_dir}")
        # Re-check
        all_ok = True
        for name in ("det.engine", "rec.engine"):
            result = check_engine(str(engines_dir / name))
            if result["status"] != "ok":
                print(f"[FAIL] {name} rebuild failed")
                all_ok = False

    if not all_ok:
        print("\n[WARN] Some engines invalid. Run: python scripts/build_trt_engines.py")
        sys.exit(1)
    else:
        print("\n[ALL OK] All engines valid")


if __name__ == "__main__":
    main()
