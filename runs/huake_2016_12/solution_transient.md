## 华科2016考研真题12题换路后暂态过程(t>0)

**分析类型**: transient

**网表** (lcapy):
```
C1 2 0 0.5 4
R2 2 0 2
R1 2 1 4
L1 1 0 1 1
```

**题目已知 (仅来自印刷内容)**: 开关打开后的无源二阶RLC电路，初始储能 uC(0+)=4V, iL(0+)=1A

**所求**: uC(t) 暂态响应与微分方程验证

### 结果

- 节点电压 $U_{2}$ = $ \begin{cases} 2 \left(3 e^{t} - 1\right) e^{- 3 t} & \text{for}\: t \geq 0 \\\text{NaN} & \text{otherwise} \end{cases} $；s 域 $ \frac{2 \left(2 s + 7\right)}{s^{2} + 5 s + 6} $
- $I_{L1}$ (参考: 1 → L1 → 0) = $ \begin{cases} \left(3 e^{t} - 2\right) e^{- 3 t} & \text{for}\: t \geq 0 \\\text{NaN} & \text{otherwise} \end{cases} $；s 域 $ \frac{s + 5}{s^{2} + 5 s + 6} $
- $I_{C1}$ (参考: 2 → C1 → 0) = $ \begin{cases} 3 \left(- 2 e^{t} + 1\right) e^{- 3 t} & \text{for}\: t \geq 0 \\\text{NaN} & \text{otherwise} \end{cases} $；s 域 $ \frac{3 \left(- s - 4\right)}{s^{2} + 5 s + 6} $

### 元件 电压/电流/功率 一览 (关联参考方向: 第1节点→第2节点)

| 元件 | 节点 | U | I | P(吸收) |
|---|---|---|---|---|

### 自动审计结论 (Zero-Latency Audit)

- ✅ **自动审计通过**：所有计算结果分母整洁、等效内阻物理合理且无暂态发散项。
