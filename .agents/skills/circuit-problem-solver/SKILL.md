---
name: circuit-problem-solver
description: >-
  识别并求解电路分析与模拟电子题目图片（区分手写批注与印刷原题、YOLO+CV 拓扑追踪、
  网表回绘对比校验、lcapy 符号与数值求解）。当用户发送电路图或要求解电路/模电题时必须激活。
---

# 电路与模电题目分析求解规程

核心原则：**严禁看原图直接“脑补”网表或心算！** 必须走完“**分离 → 视觉核对 → 网表回绘对比 → lcapy 精确求解**”的 5 步闭环，彻底隔离学生手写批注与印刷原题。

---

## Step 1：手写/印刷分离 + 拓扑初检

先对原始题目图片运行一键流水线（若题图上方或周围有大段印刷题干文字，先大致估计电路图区域或先不带 `--crop` 跑一次看 `03_overlay.png` 上的坐标标尺，再带 `--crop` 精确裁剪电路区）：

```powershell
cd d:\circuit-solver
python -m circuit_tool.run <题目图片路径> -o runs\<题目名> [--crop x,y,w,h]
```

> - 若全图手写与印刷同为黑笔、且铅笔判别误伤或漏检，可通过 `--sep-mode` (`auto|color|pencil|model|none`)、`--dark-thr`（默认 `105`）、`--hand-boxes "x,y,w,h;..."`、`--print-boxes "x,y,w,h;..."` 精确修正。
> - 坐标统一使用输出图（`03_overlay.png` 等）边缘像素标尺的**工作图坐标**；加了 `--crop` 后，`detect` 的坐标相对于 `01_print_crop.png`。

---

## Step 2：视觉双重核对（核心防错步）

用 `read_file` 依次检查以下图片：

1. **查看 `runs\<题目名>\03_overlay.png` 与 `01_print.png`（或 `01_print_crop.png`）**
   - 确认所有学生手写批注（蓝色/红色笔迹=洋红、铅笔/模型检出=橙色）都已从 `01_print.png` 擦除。
   - **信息隔离铁律**：只将 `01_print.png` 中的电路与文字当作**题目已知条件**；`02_hand.png` 仅视为**学生草稿/待批改结论**，绝不可当作已知参数！
   - 若擦除处压在导线上导致断线未被 `repair_wire_gaps` 补齐，后续在编写网表时以原图拓扑为准人工连通，或在 `detect` 时注意提示。
2. **查看 `runs\<题目名>\topology_annotated.png` 与 `netlist_draft.txt`**
   - 对照图像顶部与左侧的像素坐标标尺，逐个核对红框元件（YOLO 双模型 `v3+v2` 检出）与彩色节点连通域。
   - 若遇到 YOLO 未支持的元件（**受控源 `vcvs/vccs/ccvs/cccs`、开关 `switch`、端口 `port`、MOS管 `mosfet`、变压器 `transformer`、耦合电感 `coupled_inductor`、电流/电压表 `ammeter/voltmeter`**），或有误检/粘连，运行 `detect` 人工修正：

```powershell
python -m circuit_tool.detect runs\<题目名>\01_print_crop.png -o runs\<题目名> ^
  --add "ccvs:x,y,w,h;port:x,y,w,h" --drop "R9" --box "R1:x,y,w,h" --mask "x,y,w,h"
```

---

## Step 3：编写 `problem.yaml`（严格遵守 lcapy 语法与参考方向）

在 `runs\<题目名>\problem.yaml` 中整理干净的网表与查询项：

```yaml
title: "题目名称"
given: "仅来自 01_print.png 的已知条件（注明忽略了哪些手写批注）"
ask: "题目要求求解的物理量"
domain: dc   # dc | ac | transient
netlist: |
  Vs N1 0 12
  R1 N1 N2 4
  ...
queries:
  - V(N2)
  - I(R1)
  - thevenin(a, 0)
equations: nodal   # 可选: nodal | mesh | mna
table: true
```

### lcapy 元件与方向速查表（务必逐项核对！）

> **命名规则**：所有元件名**必须以标准大写字母开头**（如 `Rbe`、`Rf`、`Cin`、`Vs`，不可写成小写 `rbe`）。

| 元件类型 | lcapy 网表写法 | 极性 / 参考方向说明（极其重要！） |
|---|---|---|
| 电阻 / 电容 / 电感 | `R1 n1 n2 10`<br>`C1 n1 n2 1e-6 [v0]`<br>`L1 n1 n2 0.1 [i0]` | 关联参考方向：电压 `n1+ → n2−`，电流 `n1 → 元件 → n2` |
| 独立电压源 | `Vs n+ n- dc 12`<br>`Vs n+ n- ac 10 0 100`<br>`Vs n+ n- step 10` | `n+` 为正极，`n-` 为负极；`ac 幅值 初相(rad) 角频率ω`；流过电源的 `I(Vs)` 为从 `n+` 流向 `n-`（发出功率时 `I(Vs) < 0`） |
| **独立电流源** | `Is n_head n_tail 2` | ⚠️ **箭头指向第 1 节点 `n_head`**（源内部从第 2 节点 `n_tail` 流向第 1 节点 `n_head`，向 `n_head` 注入电流） |
| VCVS 压控电压源 | `E1 n+ n- nc+ nc- gain` | $U_{n+,n-} = \text{gain}\cdot(U_{nc+} - U_{nc-})$ |
| **VCCS 压控电流源** | `G1 n_head n_tail nc+ nc- gm` | ⚠️ 与 `I` 相同：**箭头指向第 1 节点 `n_head`**，$I = g_m (U_{nc+} - U_{nc-})$ 流入 `n_head` |
| CCVS 流控电压源 | `Vm ca cb 0`<br>`H1 n+ n- Vm rm` | 先串入 0V 电压源 `Vm` 取样控制电流（方向 `ca → cb`）；$U_{n+,n-} = r_m I_{Vm}$ |
| **CCCS 流控电流源** | `Vm ca cb 0`<br>`F1 n_head n_tail Vm beta` | ⚠️ **与 `I`/`G` 相反（遵循 SPICE 习惯）**：电流从第 1 节点 `n_head` 流向第 2 节点 `n_tail`（箭头指向 `n_tail`），大小为 $\beta I_{Vm}$ |
| 理想运算放大器 | `E1 out 0 in_p in_n 1e8` | 用高增益 VCVS 建模（或配合负反馈验证虚短 $U_+ \approx U_-$、虚断） |
| BJT / MOS 小信号模型 | `Vm b b1 0`<br>`Rbe b1 0 2000`<br>`F1 c 0 Vm 100` | BJT 共射小信号：基极电流 `b → Vm → Rbe → e`，受控源 `F1 c e Vm β`（电流从集电极 `c` 流向发射极 `e`）；MOS 用 `G1 d s g s gm`（或 `G1 s d g s gm` 注意箭头指向第 1 节点） |

---

## Step 4：网表回绘与并排对比校验

```powershell
python -m circuit_tool.render runs\<题目名>\problem.yaml -o runs\<题目名> --original runs\<题目名>\01_print_crop.png
```

用 `read_file` 查看 `runs\<题目名>\compare.png` 与 `audit_report.txt`：
- 左侧是干净的原题电路，右侧是网表还原的**节点-元件二部图**（红色小圆圈 `1`、`2` 标注网表中第 1、第 2 节点）及底部黄色框内的**【ERC与方向自检报告】**。
- 逐项核对自动打印的电源极性、电流源/受控流源流向箭头与原图是否完全一致，确认无 `[ERC错误]` / `[ERC警告]`；若有偏差，直接修改 `problem.yaml` 并重新 `render` 直到 100% 一致。

---

## Step 5：lcapy 精确求解与分步讲解

```powershell
python -m circuit_tool.solve runs\<题目名>\problem.yaml -o runs\<题目名>\solution.md
```

阅读 `runs\<题目名>\solution.md`，先检查末尾的 **`### 自动审计结论 (Zero-Latency Audit)`**（若触发 `🚨 [怪异数值警报]` / `[物理失稳警报]` / `[负等效电阻警报]`，必须回退检查受控源极性与擦除断线后再输出），随后结合题目要求向用户输出最终解答：
1. 明确列出**识别出的印刷原题条件**与**忽略的手写批注**；
2. 给出参考方向定义、核心电路方程（节点电压/网孔电流/KCL-KVL/运放虚短虚断）与关键化简过程；
3. 给出 `lcapy` 计算出的精确符号解与数值解（含单位）；
4. 若 `02_hand.png` 中包含学生的手写解题草稿或答案（如 `I3=1.5A?`），单独点评其对错及错因。
