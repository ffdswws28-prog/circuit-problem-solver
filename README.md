# ⚡ Circuit Problem Solver (`circuit_tool` & Agent Skill)

**面向电路分析与模拟电子实拍题目的「去手写干扰 + 拓扑视觉自检 + 符号精确求解」AI Agent 工具链**  
*A Neuro-Symbolic AI Agent Toolchain for Circuit Schematic Parsing, Handwriting Separation, Visual Netlist Verification, and Symbolic Solving.*

[中文文档](#-项目背景与核心痛点) | [English Overview](#-overview-english)

---

## 🎯 项目背景与核心痛点

直接让通用多模态大模型（如 GPT、Gemini、Claude）解答手机拍摄的大学《电路分析》或《模拟电子技术》试卷/作业图时，往往会遭遇**三大致命幻觉**：
1. **手写与印刷信息混淆**：学生在题目上留下的蓝/红/铅笔涂鸦、错误标注的参考方向与草稿算式，极易被大模型当作题目已知条件（上下文锚定效应）。
2. **空间连线与极性幻觉**：大模型直接看像素图容易看错十字交叉导线、T 型节点、受控源（VCVS/CCCS/VCCS/CCVS）控制支路及电流源箭头方向。
3. **复杂方程心算出错**：一阶/二阶动态电路换路、相量复数运算、戴维南/诺顿等效内阻计算经常在最后几步出现正负号或代数错误。

### ✨ 核心方案：5 步神经符号闭环（Neuro-Symbolic 5-Step Loop）

本项目将**计算机视觉（OpenCV + YOLOv8 + RapidOCR）**、**图论网表回绘与零延迟静态审计（NetworkX + ERC Audit）**与**计算机代数系统（Lcapy + SymPy）**封装为可由 AI Agent 自主调度的标准技能（[`.agents/skills/circuit-problem-solver/SKILL.md`](.agents/skills/circuit-problem-solver/SKILL.md)）：

1. **Step 1 — 手写/印刷物理隔离（`separate.py`）**：背景归一化 + 彩笔 HSV/`|B-R|` 滞回分层 + 铅笔灰度检测 + 擦除断线修补（`repair_wire_gaps`），严格隔离出纯净原题层 `01_print.png` 与独立学生手写层 `02_hand.png`。
2. **Step 2 — 元件检测与导线拓扑追踪（`detect.py`）**：融合双 YOLOv8 模型（`v3+v2`）与 RapidOCR，屏蔽文字后用形态学骨架追踪导线连通域与端子接触点，输出带像素标尺的 `topology_annotated.png` 与 `netlist_draft.txt`。
3. **Step 3 — 规范化网表建模（`problem.yaml`）**：严格只依据 `01_print.png` 编写网表，物理隔绝学生错误草稿的干扰。
4. **Step 4 — 网表二部图回绘 + 零延迟 ERC/方向自检（`render.py`）**：将网表还原为「节点-元件二部图」并与原图并排拼接为 `compare.png`，在毫秒级内自动完成参考地校验、悬空节点检查（支持待测开路端口白名单）以及电压源/电流源/受控源的显式物理方向断言（`audit_report.txt`）。
5. **Step 5 — 符号精确求解 + 异常解雷达 + 手写批改（`solve.py`）**：调用 `lcapy` 进行直流、正弦稳态相量、拉普拉斯暂态符号与数值求解，自动拦截怪异大质数分母、负等效电阻或发散时间常数（$\tau < 0$），并结合 `02_hand.png` 精准定位学生的错因步骤。

---

## 📌 实测经典用例（Real Exam Benchmarks）

项目在多道高校考研真题与教材实拍图上完成了全流程验证（目录均保留于 `runs/`）：

| 测试用例 | 题目类型与干扰难点 | 关键识别与求解输出 |
|---|---|---|
| **华科 2016 第 12 题**<br>(`runs/huake_2016_12/`) | 二阶 RLC 动态电路暂态过程，图面含**与印刷同色黑笔**绘制的等效草稿图与演算批注 | 隔离同色黑笔手写后算出初值 $u_C(0^+) = 4\,\mathrm{V}$、$i_L(0^+) = 1\,\mathrm{A}$、$\frac{\mathrm{d}u_C}{\mathrm{d}t}(0^+) = -6\,\mathrm{V/s}$；二阶微分方程 $\frac{\mathrm{d}^2 u_C}{\mathrm{d}t^2} + 5\frac{\mathrm{d}u_C}{\mathrm{d}t} + 6u_C = 0$，响应 $u_C(t) = (6e^{-2t}-2e^{-3t})\,\mathrm{V}\;(t \ge 0)$，并准确指出黑笔手写中的符号错误（误写为 $+6\,\mathrm{V\cdot s^{-1}}$） |
| **清华 2005 第 13 题**<br>(`runs/tsinghua_2005_13/`) | 双开关同时换路的一阶 RC 暂态电路，要求求电流 $i(t)$ 并对比手绘波形 | 剥离题面黑笔批注后求得换路初值 $u_C(0^+) = -8\,\mathrm{V}$；换路后响应 $i(t) = \left(\frac{5}{3} - \frac{6}{5}e^{-t}\right)\,\mathrm{A}\;(t \ge 0)$，同时生成标准波形图 `waveform_correct.png` 批改手绘波形 |
| **武理 2020 第 15 题**<br>(`runs/wuli_2020_15/`) | 含压控电压源（$2u_1$）网络的诺顿等效电路，周边带大量手写演算 | 过滤手写草稿后显式自检 VCVS `E1: [3](+)—[0](-) | 控制 V(2)-V(1)` 与待测开路端口 `[a]`，精确求得 $U_{oc}=12\,\mathrm{V},\; I_{sc}=2\,\mathrm{A},\; R_{eq}=6\,\Omega$ |

---

## 🧩 整合的开源项目与依赖

| 模块 | 开源项目 / 依赖 | 作用 | 本地位置 |
|---|---|---|---|
| 元件检测 & OCR | [HR-DONK/circuit-recognition](https://github.com/HR-DONK/circuit-recognition)（YOLOv8 `v3` + `v2`, RapidOCR） | 检测电阻、电容、电感、二极管、三极管、运放、独立电源、接地等基础元件与参数文字 | `vendor/circuit-recognition/` |
| 同色手写擦除（可选） | [AndSonder/HandWritingEraser-Pytorch](https://github.com/AndSonder/HandWritingEraser-Pytorch)（DeepLabV3+） | 针对中文试卷与印刷同色黑笔手写的语义分割与擦除模型架构 | `vendor/HandWritingEraser-Pytorch-main/` |
| 符号与数值求解 | [mph-/lcapy](https://github.com/mph-/lcapy) + SymPy + NetworkX | 精确构建 MNA/节点/网孔方程、戴维南/诺顿等效、相量与拉普拉斯暂态符号解；并绘制二部图 | `requirements.txt` |

> **黑笔手写模型权重说明**：蓝/红彩笔与铅笔手写已可**开箱即用**自动分离；若需擦除与印刷完全同色的黑笔字迹，可将 `HandWritingEraser-Pytorch` 预训练 `.pth` 权重放入 `models/handwriting_eraser/`。

---

## 📂 仓库目录结构

```text
├── GEMINI.md                              # Antigravity 工作区规则（强制走 5 步闭环）
├── README.md                              # 中英文项目说明文档
├── requirements.txt                       # Python 依赖清单
├── .agents/skills/circuit-problem-solver/
│   └── SKILL.md                           # Antigravity 专用解题 Skill 规范
├── circuit_tool/
│   ├── separate.py                        # Step 1: 手写/印刷分离 + 断线修补 + 坐标标尺
│   ├── detect.py                          # Step 2: YOLO 双模型 + OCR + 导线连通追踪
│   ├── render.py                          # Step 4: 二部图回绘 + 零延迟 ERC/方向自检
│   ├── solve.py                           # Step 5: Lcapy 符号/数值求解 + 异常解雷达
│   └── run.py                             # 一键流水线入口 (separate -> detect)
├── vendor/                                # 上游开源模型代码与权重
├── tests/                                 # 合成电路图与受控源/运放回归测试
└── runs/                                  # 实测真题样例 (huake_2016_12, tsinghua_2005_13, wuli_2020_15 等)
```

---

## 🚀 快速开始 (Quick Start)

```powershell
# 1. 安装依赖
pip install -r requirements.txt

# 2. 手写分离 + 元件与导线拓扑初检
python -m circuit_tool.run <题目图片路径> -o runs/my_prob [--crop x,y,w,h]

# 3. (可选) 人工补框受控源/开关等未覆盖元件
python -m circuit_tool.detect runs/my_prob/01_print_crop.png -o runs/my_prob --add "vcvs:100,120,60,60" --drop "R9"

# 4. 编写 problem.yaml 后回绘网表并执行零延迟 ERC/方向自检
python -m circuit_tool.render runs/my_prob/problem.yaml -o runs/my_prob --original runs/my_prob/01_print_crop.png

# 5. 运行 lcapy 精确求解与异常解审计
python -m circuit_tool.solve runs/my_prob/problem.yaml -o runs/my_prob/solution.md
```

---

## 🌍 Overview (English)

### Why This Project?
General Multimodal LLMs (such as GPT, Gemini, and Claude) frequently fail on real-world photos of university circuit analysis and analog electronics problems due to three bottlenecks:
1. **Handwriting Contamination**: Blue/red pen scribbles, guessed reference arrows, and incorrect handwritten scratchwork on exam photos are easily mistaken by VLMs for given problem conditions.
2. **Topological & Polarity Hallucination**: VLMs often misread wire crossings, T-junctions, current-source arrows, and dependent-source control branches (VCVS, CCCS, VCCS, CCVS).
3. **Arithmetic & Symbolic Drift**: Mental calculations by LLMs on nodal/mesh equations, complex phasors, or Laplace transients regularly introduce sign or fraction errors.

### Architecture: 5-Step Neuro-Symbolic Closed Loop
This repository packages OpenCV, YOLOv8, RapidOCR, NetworkX graph visualization, deterministic ERC auditing, and `lcapy`/`SymPy` into an AI Agent Skill (`.agents/skills/circuit-problem-solver/SKILL.md`):
1. **Strict Information Barrier (`separate.py`)**: Separates printed schematics (`01_print.png`) from handwritten student ink (`02_hand.png`) using HSV/chromatic hysteresis + grayscale morphology + wire-gap healing (`repair_wire_gaps`). The netlist builder reads *only* `01_print.png`; `02_hand.png` is reserved strictly for grading and error diagnosis.
2. **Component & Wire Tracing (`detect.py`)**: Ensembles `v3 + v2` YOLOv8 detectors with RapidOCR masking and connected-component skeleton tracing to produce coordinate-ruled drafts (`topology_annotated.png`, `netlist_draft.txt`).
3. **Visual Netlist Re-Rendering & Zero-Latency ERC Audit (`render.py`)**: Renders the YAML netlist back into a node-element bipartite graph side-by-side with the printed schematic (`compare.png`), while running `< 0.1s` deterministic ERC checks (ground presence, dangling nodes with open-port whitelist) and printing explicit physical polarity/current-direction assertions (`audit_report.txt`).
4. **Exact Symbolic Solving & Abnormal-Answer Radar (`solve.py`)**: Solves DC, AC phasor, Thevenin/Norton, and Laplace transient circuits via `lcapy` and flags suspiciously large fraction denominators, negative equivalent resistances, or unstable poles ($\tau < 0$).

### Verified Benchmark Cases
- **HUST 2016 Q12 (`runs/huake_2016_12/`)**: 2nd-order RLC transient circuit with **same-color black-pen** student equivalent circuits and annotations separated via `--hand-boxes` -> steady-state $u_C(0^+)=4\,\mathrm{V},\; i_L(0^+)=1\,\mathrm{A},\; \frac{\mathrm{d}u_C}{\mathrm{d}t}(0^+)=-6\,\mathrm{V/s}$, ODE $\frac{\mathrm{d}^2 u_C}{\mathrm{d}t^2} + 5\frac{\mathrm{d}u_C}{\mathrm{d}t} + 6u_C = 0$, and $u_C(t) = (6e^{-2t}-2e^{-3t})\,\mathrm{V}\;(t \ge 0)$, catching the $+6\,\mathrm{V\cdot s^{-1}}$ sign error in the black-pen draft.
- **Tsinghua 2005 Q13 (`runs/tsinghua_2005_13/`)**: Dual-switch 1st-order RC transient response with black-pen scribbles separated -> $u_C(0^+)=-8\,\mathrm{V}$ and $i(t) = \left(\frac{5}{3} - \frac{6}{5}e^{-t}\right)\,\mathrm{A}\;(t \ge 0)$, with reference vs. student waveform comparison.
- **WHUT 2020 Q15 (`runs/wuli_2020_15/`)**: VCVS ($2u_1$) network Norton equivalent -> $U_{oc}=12\,\mathrm{V},\; I_{sc}=2\,\mathrm{A},\; R_{eq}=6\,\Omega$, verified cleanly by explicit VCVS polarity audit and open-port whitelist.
