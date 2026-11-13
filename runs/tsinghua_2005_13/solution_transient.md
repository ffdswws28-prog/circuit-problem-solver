## 清华2005考研真题13题换路后暂态过程(t>0)

**分析类型**: transient

**网表** (lcapy):
```
Vs 4 0 30
R0 4 1 12
Ri 1 0 6
C1 1 2 0.1 -8
R2 2 0 6
```

**题目已知 (仅来自印刷内容)**: 换路后 S1 闭合、S2 打开，初始储能 uC(0+)=-8V

**所求**: 求 i(t) 响应

### 结果

- $I_{Ri}$ (参考: 1 → Ri → 0) = $ \begin{cases} \frac{5}{3} - \frac{6 e^{- t}}{5} & \text{for}\: t \geq 0 \\\text{NaN} & \text{otherwise} \end{cases} $；s 域 $ \frac{7 s + 25}{15 s \left(s + 1\right)} $
- $U_{C1}$ (参考: 1+ → 2−) = $ \begin{cases} 10 - 18 e^{- t} & \text{for}\: t \geq 0 \\\text{NaN} & \text{otherwise} \end{cases} $；s 域 $ \frac{2 \left(- 4 s + 5\right)}{s \left(s + 1\right)} $
- 节点电压 $U_{1}$ = $ \begin{cases} 10 - \frac{36 e^{- t}}{5} & \text{for}\: t \geq 0 \\\text{NaN} & \text{otherwise} \end{cases} $；s 域 $ \frac{2 \left(7 s + 25\right)}{5 s \left(s + 1\right)} $
- 节点电压 $U_{2}$ = $ \begin{cases} \frac{54 e^{- t}}{5} & \text{for}\: t \geq 0 \\\text{NaN} & \text{otherwise} \end{cases} $；s 域 $ \frac{54}{5 \left(s + 1\right)} $

### 元件 电压/电流/功率 一览 (关联参考方向: 第1节点→第2节点)

| 元件 | 节点 | U | I | P(吸收) |
|---|---|---|---|---|

### 自动审计结论 (Zero-Latency Audit)

- ✅ **自动审计通过**：所有计算结果分母整洁、等效内阻物理合理且无暂态发散项。
