"""生成测试题图: schemdraw 印刷电路 + 题干 + 模拟蓝笔/铅笔手写批注"""
import sys
from pathlib import Path

import cv2
import numpy as np
import schemdraw
import schemdraw.elements as elm
import matplotlib
matplotlib.use("Agg")

out = Path(sys.argv[1] if len(sys.argv) > 1 else "tests")
out.mkdir(parents=True, exist_ok=True)

with schemdraw.Drawing(show=False) as d:
    d.config(unit=3, fontsize=14)
    V = d.add(elm.SourceV().up().label("12V", loc="left"))
    d.add(elm.Line().right(0.5))
    R1 = d.add(elm.Resistor().right().label("R1=4Ω"))
    d.add(elm.Dot())
    d.push()
    R2 = d.add(elm.Resistor().down().label("R2=6Ω", loc="bottom"))
    d.add(elm.Dot())
    d.pop()
    R3 = d.add(elm.Resistor().right().label("R3=3Ω"))
    d.push()
    I = d.add(elm.SourceI().down().reverse().label("2A", loc="bottom"))
    d.pop()
    d.add(elm.Line().right(1.5))
    d.add(elm.Dot(open=True).label("a", loc="right"))
    d.add(elm.Line().down(3).at(I.end).to((I.end[0] + 0, I.end[1])))
    d.add(elm.Line().left().tox(V.start))
    d.add(elm.Dot(open=True).at((I.end[0] + 1.5, I.end[1])).label("b", loc="right"))
    d.add(elm.Line().at(I.end).tox(I.end[0] + 1.5))
    d.save(str(out / "_circ.png"), dpi=110)

circ = cv2.imread(str(out / "_circ.png"))
H, W = circ.shape[:2]
page = np.full((H + 260, max(W + 120, 900), 3), 255, np.uint8)
page[150:150 + H, 60:60 + W] = circ
from PIL import Image, ImageDraw, ImageFont
pil = Image.fromarray(cv2.cvtColor(page, cv2.COLOR_BGR2RGB))
dr = ImageDraw.Draw(pil)
font = ImageFont.truetype("C:/Windows/Fonts/simsun.ttc", 24)
dr.text((40, 30), "3-7  电路如图所示，求 R3 中的电流 I3，", fill=(0, 0, 0), font=font)
dr.text((40, 70), "     并求 a、b 端的戴维南等效电路。", fill=(0, 0, 0), font=font)
page = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

rng = np.random.default_rng(0)
def scribble(img, pts, color, thick):
    pts = np.array(pts, np.int32) + rng.integers(-2, 3, (len(pts), 2))
    cv2.polylines(img, [pts], False, color, thick, cv2.LINE_AA)

# 蓝笔: 学生写的错误结论 + 圈 + 电流箭头
blue = (200, 80, 20)
cv2.putText(page, "I3=1.5A?", (W - 120, 210), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.1, blue, 2, cv2.LINE_AA)
scribble(page, [(330, 330), (380, 300), (430, 310), (440, 360), (390, 390), (340, 370), (330, 330)], blue, 2)
scribble(page, [(200, 235), (260, 235), (250, 225)], blue, 2)
# 铅笔: 浅灰手写节点标号和一条辅助线 (同色系, 但比印刷浅)
pencil = (150, 150, 150)
cv2.putText(page, "n1", (215, 290), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, pencil, 2, cv2.LINE_AA)
cv2.putText(page, "Uoc=?", (60, page.shape[0] - 40), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, pencil, 2, cv2.LINE_AA)
# 拍照效果: 偏黄纸张 + 光照渐变 + 噪声
h, w = page.shape[:2]
grad = np.linspace(0.85, 1.0, w)[None, :, None]
paper = np.array([0.93, 0.97, 1.0])[None, None, :]
photo = np.clip(page.astype(np.float32) * grad * paper + rng.normal(0, 4, page.shape), 0, 255).astype(np.uint8)
cv2.imwrite(str(out / "test_problem.jpg"), photo)
print(out / "test_problem.jpg")
