# -*- coding: utf-8 -*-
"""Export fine-tuned YOLOv8n to TensorRT FP16 engine + benchmark."""
import sys
import time
import numpy as np
from pathlib import Path

def main():
    from ultralytics import YOLO

    MODEL_PATH = "D:/ai/circuit-recognition/models/circuit_yolov8n_finetuned.pt"
    OUTPUT_DIR = Path("D:/ai/circuit-recognition/models")
    IMGSZ = 512

    print("=" * 60)
    print("TensorRT Export: YOLOv8n Circuit Model")
    print(f"  Model: {MODEL_PATH}")
    print("=" * 60)

    # Step 1: Export to ONNX
    print("\n[1/3] Exporting to ONNX...")
    model = YOLO(MODEL_PATH)
    onnx_path = model.export(format="onnx", opset=12, simplify=True, imgsz=IMGSZ, half=False)
    print(f"  ONNX: {onnx_path}")

    # Step 2: Export to TensorRT FP16
    print("\n[2/3] Building TensorRT FP16 engine...")
    try:
        engine_path = model.export(format="engine", device=0, half=True, imgsz=IMGSZ, workspace=4)
        print(f"  TensorRT: {engine_path}")
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Trying manual ONNX -> TensorRT conversion...")
        engine_path = _manual_onnx_to_trt(onnx_path, OUTPUT_DIR)
        if not engine_path:
            print("  TensorRT export FAILED. Continuing with ONNX only.")
            engine_path = None

    # Step 3: Benchmark
    print("\n[3/3] Benchmark comparison...")
    benchmark(MODEL_PATH, str(onnx_path), str(engine_path) if engine_path else None, IMGSZ)

    print("\nDone!")


def _manual_onnx_to_trt(onnx_path: str, output_dir: Path) -> str:
    """Manual ONNX -> TensorRT engine build."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  FP16 enabled")

    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Parse error: {parser.get_error(i)}")
            return ""

    print("  Building engine (may take 3-5 minutes)...")
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        print("  Failed to build engine")
        return ""

    engine_path = str(output_dir / "circuit_yolov8n_fp16.engine")
    with open(engine_path, 'wb') as f:
        f.write(plan)
    print(f"  Engine saved: {engine_path}")
    return engine_path


def benchmark(pt_path: str, onnx_path: str, engine_path: str, imgsz: int):
    """Compare inference speed: PyTorch vs ONNX vs TensorRT."""
    import torch
    from ultralytics import YOLO

    dummy = torch.randn(1, 3, imgsz, imgsz).cuda()
    warmup, iters = 10, 100

    # PyTorch
    model = YOLO(pt_path)
    model.model.cuda().eval()
    for _ in range(warmup):
        model.model(dummy)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        model.model(dummy)
    torch.cuda.synchronize()
    pt_time = (time.time() - t0) / iters * 1000

    # ONNX Runtime
    import onnxruntime as ort
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    dummy_np = dummy.cpu().numpy()
    input_name = session.get_inputs()[0].name
    for _ in range(warmup):
        session.run(None, {input_name: dummy_np})
    t0 = time.time()
    for _ in range(iters):
        session.run(None, {input_name: dummy_np})
    onnx_time = (time.time() - t0) / iters * 1000

    # TensorRT
    trt_time = None
    if engine_path and Path(engine_path).exists():
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        # Allocate I/O
        bindings = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            # Replace -1 dims with actual
            shape = [1 if s == -1 else s for s in shape]
            buf = torch.empty(tuple(shape), dtype=torch.from_numpy(np.dtype(dtype)).dtype).cuda()
            bindings.append(buf.data_ptr())
            context.set_tensor_address(name, buf.data_ptr())

        # Warmup
        input_tensor_name = engine.get_tensor_name(0)
        input_buf = torch.from_numpy(dummy_np).cuda()
        context.set_tensor_address(input_tensor_name, input_buf.data_ptr())
        for _ in range(warmup):
            context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(iters):
            context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        trt_time = (time.time() - t0) / iters * 1000
        del engine, context

    # Report
    print(f"\n  {'Backend':20s} {'Time/Infer':>12s} {'FPS':>8s} {'vs PyTorch':>12s}")
    print(f"  {'-'*52}")
    print(f"  {'PyTorch':20s} {pt_time:8.2f} ms {1000/pt_time:7.1f} fps {'1.00x':>12s}")
    if onnx_time:
        speedup = pt_time / onnx_time
        print(f"  {'ONNX Runtime':20s} {onnx_time:8.2f} ms {1000/onnx_time:7.1f} fps {speedup:9.2f}x")
    if trt_time:
        speedup = pt_time / trt_time
        print(f"  {'TensorRT FP16':20s} {trt_time:8.2f} ms {1000/trt_time:7.1f} fps {speedup:9.2f}x")

    del model, session
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
