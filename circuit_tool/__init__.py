"""circuit_tool: 电路题分析工具链

模块:
    separate  手写 / 印刷分离（颜色分层 + 铅笔灰度 + 可选深度模型 + 人工框选修正）
    detect    元件检测(YOLO, 来自 HR-DONK/circuit-recognition) + OCR + 导线连通性追踪 → 拓扑草稿
    solve     基于 lcapy 的符号/数值求解（节点法、网孔法、戴维南、相量、暂态）
    render    网表回绘（拓扑图 / lcapy 电路图）用于与原图对比校验
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor"
