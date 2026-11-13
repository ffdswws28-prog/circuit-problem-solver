## 北理 2017 第 10 题：一阶受控源电路三要素法求解 i(t)

**分析类型**: transient

**网表** (lcapy):
```
Is1 0 1 step 12
Vm 0 1a 0
R8 1a 1 8
F1 1 0 Vm 0.2
V4 b 1 step 4
C1 b 0 0.01 20
```

**题目已知 (仅来自印刷内容)**: 电路如图所示，Is1 = 12A (向上)，R1 = 8Ω (流过电流 i，向上)，受控电流源 0.2i (向下)，恒压源 4V (左负右正)，电容 C = 0.01F，右侧支路包含 R2 = 10Ω 与 20V 电压源。开关在 t = 0 时由 a 端投向 b 端，动作前处于稳态。

**所求**: 用三要素法求 t > 0 时的电流 i(t)

### 结果

- `c.Vm.i` = $ \begin{cases} 15 - 17 e^{- 10 t} & \text{for}\: t \geq 0 \\\text{NaN} & \text{otherwise} \end{cases} $；s 域 $ \frac{2 \left(- s + 75\right)}{s \left(s + 10\right)} $
- `c.C1.v` = $ \begin{cases} -116 + 136 e^{- 10 t} & \text{for}\: t \geq 0 \\\text{NaN} & \text{otherwise} \end{cases} $；s 域 $ \frac{20 \left(s - 58\right)}{s \left(s + 10\right)} $
- `c['1'].v` = $ \begin{cases} -120 + 136 e^{- 10 t} & \text{for}\: t \geq 0 \\\text{NaN} & \text{otherwise} \end{cases} $；s 域 $ \frac{16 \left(s - 75\right)}{s \left(s + 10\right)} $
