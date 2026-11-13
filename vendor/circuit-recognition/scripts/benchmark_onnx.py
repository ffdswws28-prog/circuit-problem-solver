# -*- coding: utf-8 -*-
"""Benchmark: PyTorch vs ONNX Runtime GPU inference speed."""
import sys
import time
import numpy as np
from pathlib import Path

def main():
    import torch
    from ultralytics import YOLO

    ONNX_PATH = "D:/ai/circuit-recognition/models/circuit_yolov8n_finetuned.onnx"
    PT_PATH = "D:/ai/circuit-recognition/models/circuit_yolov8n_finetuned.pt"

    IMGSZ = 512
    warmup, iters = 10, 200

    print("=" * 60)
    print("Inference Benchmark: PyTorch vs ONNX Runtime GPU")
    print(f"  Image: {IMGSZ}x{IMGSZ}, Warmup: {warmup}, Iters: {iters}")
    print("=" * 60)

    dummy = torch.randn(1, 3, IMGSZ, IMGSZ).cuda()
    gpu_name = torch.cuda.get_device_name(0)
    print(f"\nGPU: {gpu_name}")

    # ---- PyTorch ----
    print("\n[1] PyTorch (torch 2.12.1 + cu126)")
    model = YOLO(PT_PATH)
    model.model.cuda().eval()
    for _ in range(warmup):
        model.model(dummy)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        model.model(dummy)
    torch.cuda.synchronize()
    pt_ms = (time.time() - t0) / iters * 1000
    print(f"  {pt_ms:.2f} ms/infer  |  {1000/pt_ms:.0f} FPS")

    # ---- ONNX Runtime (CUDA) ----
    print("\n[2] ONNX Runtime (CUDAExecutionProvider)")
    import onnxruntime as ort
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(ONNX_PATH, providers=providers)
    print(f"  Providers: {session.get_providers()}")
    actual_providers = ort.get_available_providers()
    print(f"  Available: {actual_providers}")

    dummy_np = dummy.cpu().numpy()
    input_name = session.get_inputs()[0].name
    for _ in range(warmup):
        session.run(None, {input_name: dummy_np})
    t0 = time.time()
    for _ in range(iters):
        session.run(None, {input_name: dummy_np})
    onnx_ms = (time.time() - t0) / iters * 1000
    speedup = pt_ms / onnx_ms
    print(f"  {onnx_ms:.2f} ms/infer  |  {1000/onnx_ms:.0f} FPS  |  {speedup:.2f}x vs PyTorch")

    # ---- ONNX Runtime (TensorRT) if available ----
    if 'TensorrtExecutionProvider' in actual_providers:
        print("\n[3] ONNX Runtime (TensorrtExecutionProvider)")
        try:
            session_trt = ort.InferenceSession(ONNX_PATH, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
            for _ in range(warmup):
                session_trt.run(None, {input_name: dummy_np})
            t0 = time.time()
            for _ in range(iters):
                session_trt.run(None, {input_name: dummy_np})
            trt_ms = (time.time() - t0) / iters * 1000
            print(f"  {trt_ms:.2f} ms/infer  |  {1000/trt_ms:.0f} FPS  |  {pt_ms/trt_ms:.2f}x vs PyTorch")
        except Exception as e:
            print(f"  TensorRT EP failed: {e}")
    else:
        print("\n[3] ONNX Runtime (TensorRT EP): not available in this build")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"{'Backend':30s} {'Time':>10s} {'FPS':>8s} {'Speedup':>10s}")
    print(f"{'-'*58}")
    print(f"{'PyTorch CUDA':30s} {pt_ms:7.2f} ms {1000/pt_ms:6.0f} {'1.00x':>10s}")
    if onnx_ms:
        print(f"{'ONNX Runtime CUDA':30s} {onnx_ms:7.2f} ms {1000/onnx_ms:6.0f} {f'{speedup:.2f}x':>10s}")

    # Save ONNX model info
    onnx_file = Path(ONNX_PATH)
    print(f"\nONNX model: {onnx_file.name} ({onnx_file.stat().st_size / 1024**2:.1f} MB)")

    del model, session, dummy
    torch.cuda.empty_cache()
    print("Done!")


if __name__ == "__main__":
    main()
