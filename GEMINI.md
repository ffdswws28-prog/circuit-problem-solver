# `d:\circuit-solver` 工作区规则

1. **严禁直接看图猜拓扑或心算电路题**：在 `d:\circuit-solver` 工作区处理任何电路分析或模拟电子题目图片时，必须调用 `circuit_tool` 工具链并遵循 `.agents/skills/circuit-problem-solver/SKILL.md` 的 5 步规程。
2. **严格隔离手写与印刷**：必须先运行 `circuit_tool.separate`（或 `circuit_tool.run`）生成 `01_print.png`、`02_hand.png` 与 `03_overlay.png`。后续读题、OCR 与拓扑追踪只能基于 `01_print.png`；`02_hand.png` 中的学生批注只能用于最后点评对错，绝不可混入题目已知条件。
3. **网表先回绘、再求解**：编写 `problem.yaml` 后，必须先运行 `circuit_tool.render` 生成 `compare.png` 并核对所有元件连接与电源/受控源方向，确认无误后再运行 `circuit_tool.solve` 输出解答。
