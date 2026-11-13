# Circuit Recognition MCP

> 基于 MCP (Model Context Protocol) 的电路图智能识别系统 —— 输入电路图图片，输出结构化 JSON（元器件、文字标注、拓扑连接、网表）。

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 为什么要做这个项目

网上几乎找不到能直接用的电路图识别工具。现有的 OCR 引擎不认识电路符号，YOLO 检测没有针对电路元件的预训练模型，训练数据更是少得可怜——电路图涉及版权问题，公开数据集寥寥无几。

我当时为了做电路仿真自动化，从零开始：
1. **收集了多个公开数据集**（CGHD、HCD 等），自己合成了一批教材风格电路图
2. **训练了 YOLO 模型**，经过 v1 → v2 → v3 多轮迭代，从 8 类扩展到 10 类，从合成图泛化到手绘/照片
3. **搭建了完整流水线**：YOLO 检测 → SAM 分割 → OCR 文字识别 → 拓扑分析 → 融合输出

开源出来，希望别人不用再重复踩坑。拿过来就能用，也可以在我训练好的模型基础上继续微调。

## 能识别什么

### 支持的电路类型
- 📖 **教材/课本电路图**（主力，训练数据最多）
- ✏️ **手绘电路图**（CGHD 数据集训练）
- 📐 **原理图/示意图**（HCD 数据集训练）
- 📸 **实物照片**（部分支持，看清晰度）

### 支持的元件（10 类）

| 元件 | 英文类名 | 示例 |
|------|----------|------|
| 电阻 | `resistor` | R1, R2... |
| 电容 | `capacitor` | C1, C2... |
| 电感 | `inductor` | L1, L2... |
| 二极管 | `diode` | D1, D2... |
| 三极管 | `transistor` | Q1, Q2... |
| 运算放大器 | `opamp` | U1, U2... |
| 集成电路 | `ic` | U1, U2... |
| 接地 | `ground` | GND |
| 电压源 | `voltage_source` | VCC, VDD |
| 电流源 | `current_source` | I1, I2... |

## 训练数据来源

模型训练使用了以下数据集，在此致谢：

| 数据集 | 规模 | 类型 | 来源 |
|--------|------|------|------|
| **CGHD** (Circuit Graphic Hand-Drawn) | ~1,152 张 | 手绘电路图 | [Zenodo 公开数据集](https://zenodo.org/) |
| **HCD** (Hand-drawn Circuit Diagram) | ~200 张 | 手绘原理图 | 公开学术数据集 |
| **自合成教材数据** | ~5,000 张 | SchemDraw 自动生成 | 本项目自行生成 |

训练过程：先用自合成数据打底（量大、标注准确），再混入手绘数据集做泛化。最终模型在教材图上手绘图上都有不错的检测效果。

> 训练代码和数据集因体积过大（~20 GB）未包含在仓库中。如需训练数据或想自己训练，请在 Issues 中提需求，我可以分享数据集准备脚本。

## 快速开始

### 1. 环境准备

```bash
git clone https://github.com/HR-DONK/circuit-recognition.git
cd circuit-recognition

# 安装依赖
pip install -r requirements.txt
```

### 2. 模型下载

| 模型 | 大小 | 用途 | 方式 |
|------|------|------|------|
| `circuit_yolov8n_v3.pt` | 16 MB | 10 类元件检测 | 已内置 |
| `sam_vit_b_01ec64.pth` | 358 MB | 精细分割 | **首次运行自动下载** |

SAM 模型首次使用时会自动从 Facebook 下载到 `models/` 目录，只需等待一次。

### 3. 作为 MCP 工具使用

在 Claude Code / Claude Desktop 的 MCP 配置中添加：

```json
{
  "mcpServers": {
    "circuit-recognition": {
      "command": "python",
      "args": ["src/mcp_server.py"],
      "cwd": "/path/to/circuit-recognition"
    }
  }
}
```

4 个可用工具：

| 工具 | 功能 |
|------|------|
| `analyze_circuit` | 全流程分析：检测 + 分割 + OCR + 拓扑 + 网表 |
| `detect_components` | 快速元件检测（仅 YOLO） |
| `recognize_text` | 文字识别（仅 OCR） |
| `get_cache_status` | 查询缓存状态 |

### 4. 输出示例

```json
{
  "success": true,
  "components": [
    { "class": "resistor", "bbox": [120, 80, 60, 20], "confidence": 0.92 },
    { "class": "capacitor", "bbox": [120, 160, 40, 30], "confidence": 0.88 }
  ],
  "connections": [
    { "from": "R1.pin2", "to": "C1.pin1", "wire_segments": [...] }
  ],
  "nets": {
    "VCC": ["R1.pin1"],
    "GND": ["C1.pin2", "R2.pin1"],
    "NET_1": ["R1.pin2", "C1.pin1"]
  },
  "title_block": { "title": "RC 滤波电路" },
  "timing": {
    "yolo": 0.12, "ocr": 1.68, "total": 2.34
  }
}
```

## 流水线架构

```
输入图片 → YOLO元件检测 → SAM精细分割 → OCR文字识别 → 拓扑分析 → 融合输出
  0.1s        0.1s          按需触发       1.7s        0.2s        JSON
```

```
circuit-recognition/
├── src/
│   ├── mcp_server.py              # MCP 服务入口（4 个工具）
│   ├── pipeline/
│   │   ├── yolo_detector.py       # Stage 1: YOLO 元件检测
│   │   ├── sam_segmentor.py       # Stage 2: SAM 精细分割
│   │   ├── ocr_engine.py          # Stage 3: OCR 引擎（调度层）
│   │   ├── smart_filter.py        #     文字框多级过滤（size/conf/NMS/ROI/margin）
│   │   ├── backends/
│   │   │   ├── base.py            #     OCR 后端抽象基类
│   │   │   ├── rapidocr_backend.py #    RapidOCR 后端（默认，CPU 可用）
│   │   │   └── trt_backend.py     #     TensorRT 后端（GPU 加速）
│   │   ├── topology.py            # Stage 4: 拓扑分析
│   │   ├── fusion.py              # Stage 5: 融合 + 网表生成
│   │   └── preprocessor.py        # 图像预处理 + PDF 转换
│   └── cache.py                   # 多级缓存
├── config/
│   └── ocr_config.yaml            # OCR 参数配置
├── scripts/
│   └── build_trt_engines.py       # TensorRT 引擎构建（需 GPU）
├── tests/                         # 38 个测试
├── models/                        # 模型文件（自动下载）
└── requirements.txt
```

## 替换 OCR 引擎

OCR 模块采用 **可插拔后端架构**。默认使用 RapidOCR（ONNX Runtime，CPU 即可运行），可以无缝替换为其他 OCR 引擎：

只需实现 `OcrBackendBase` 抽象类：

```python
from src.pipeline.backends.base import OcrBackendBase
import numpy as np

class MyOcrBackend(OcrBackendBase):
    """接入你的 OCR 引擎"""

    def initialize(self) -> None:
        pass  # 初始化你的 OCR 模型

    def release(self) -> None:
        pass  # 释放资源

    @property
    def is_available(self) -> bool:
        return True

    def detect(self, image: np.ndarray) -> dict:
        # 文字区域检测
        # 返回 {"boxes": [...], "scores": [...], "cost_sec": 0.0, "img_hw": (h, w)}
        pass

    def recognize_batch(self, image: np.ndarray, boxes: list) -> list:
        # 批量文字识别
        # 返回 [{"text": "...", "confidence": 0.95, "bbox": [x,y,w,h]}, ...]
        pass
```

然后在 `ocr_engine.py` 中注册，在 `config/ocr_config.yaml` 中切换：

```yaml
backend:
  default: myocr
```

**无需修改 mcp_server.py 或任何上层代码。** 支持接入：PaddleOCR、Tesseract、EasyOCR、Azure OCR、任何你喜欢的引擎。

## 性能

| 配置 | OCR 耗时 | 说明 |
|------|---------|------|
| RapidOCR（默认） | ~1.7s/页 | CPU 可用，SmartFilter 过滤无关文字 |
| RapidOCR + TensorRT rec | ~0.3-0.5s/页 | 需 NVIDIA GPU + TensorRT |
| 原始 RapidOCR（优化前） | ~2.5s/页 | 逐框识别，无过滤 |

## 依赖

```
Python 3.10+
├── torch + ultralytics      # YOLO 推理
├── segment-anything         # SAM 分割
├── rapidocr-onnxruntime     # OCR（默认后端）
├── opencv-python-headless   # 图像处理
├── PyMuPDF                  # PDF 支持
└── mcp                      # MCP 协议框架
```

详见 [requirements.txt](requirements.txt)。

## License

MIT © 何睿
