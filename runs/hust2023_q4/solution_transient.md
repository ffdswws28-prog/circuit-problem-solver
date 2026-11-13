## 华科 2023 第 4 题 (2)：t >= 0 开关 S 打开后的暂态响应

**分析类型**: transient

**网表** (lcapy):
```
Vs nuc 0 step 8
C1 A nuc 0.05 12
R2 A 0 10
R1 T1 A 10
L1 T1 A 0.5 2
R3 T2 0 10
Is T2 0 step 4
```

**题目已知 (仅来自印刷内容)**: R1 = R2 = R3 = 10Ω, L = 0.5H, C = 0.05F, Us = 8V, Is = 4A；初值 iL(0+) = 2A (向下), uC(0+) = 12V (上+下-)；t = 0 时开关 S 打开

**所求**: 求 t >= 0 时的 iL(t)、uC(t) 以及开关 S 两端的电压 uS(t)

### 结果

- $i_{L1}(t)$ (T1→A) = $ \begin{cases} 2 e^{- 20 t} & \text{for}\: t \geq 0 \\\text{NaN} & \text{otherwise} \end{cases} $
- $v_{C1}(t)$ (A→nuc) = $ \begin{cases} -8 + 20 e^{- 2 t} & \text{for}\: t \geq 0 \\\text{NaN} & \text{otherwise} \end{cases} $
- $v_{A}(t)$ (对地) = $ \begin{cases} 20 e^{- 2 t} & \text{for}\: t \geq 0 \\\text{NaN} & \text{otherwise} \end{cases} $
- $v_{T1}(t)$ (对地) = $ \begin{cases} \left(20 e^{18 t} - 20\right) e^{- 20 t} & \text{for}\: t \geq 0 \\\text{NaN} & \text{otherwise} \end{cases} $
- $v_{T2}(t)$ (对地) = $ \begin{cases} 40 & \text{for}\: t \geq 0 \\\text{NaN} & \text{otherwise} \end{cases} $
- `(c['T2'].v - c['T1'].v).simplify()` = $ \begin{cases} 40 - 20 e^{- 2 t} + 20 e^{- 20 t} & \text{for}\: t \geq 0 \\\text{NaN} & \text{otherwise} \end{cases} $；s 域 $ \frac{40 \left(s^{2} + 13 s + 40\right)}{s \left(s^{2} + 22 s + 40\right)} $
- `(c['T1'].v - c['T2'].v).simplify()` = $ \begin{cases} -40 + 20 e^{- 2 t} - 20 e^{- 20 t} & \text{for}\: t \geq 0 \\\text{NaN} & \text{otherwise} \end{cases} $；s 域 $ - \frac{20}{s + 20} + \frac{20}{s + 2} - \frac{40}{s} $
