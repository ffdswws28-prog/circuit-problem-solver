"""模型导出工具 — ONNX → TensorRT INT8"""
import os
from pathlib import Path


def export_yolo_to_onnx(model_path: str, output_path: str, opset: int = 12):
    """YOLO → ONNX（opset=12, simplify）"""
    from ultralytics import YOLO
    model = YOLO(model_path)
    model.export(format="onnx", opset=opset, simplify=True, imgsz=640)
    print(f"ONNX exported to: {output_path}")


def export_onnx_to_trt(onnx_path: str, engine_path: str, fp16: bool = True, int8: bool = False,
                       calibration_dir: str = None):
    """ONNX → TensorRT Engine"""
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        # TODO: 设置 INT8 校准数据集

    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"ONNX Parse Error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    plan = builder.build_serialized_network(network, config)
    with open(engine_path, 'wb') as f:
        f.write(plan)
    print(f"TensorRT engine exported to: {engine_path}")
