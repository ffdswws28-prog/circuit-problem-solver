"""一键流水线: 手写分离 → 元件检测/导线追踪

用法:
    python -m circuit_tool.run <题目图片> [-o runs/<名字>] [--sep-mode auto] [detect 的其它参数...]

完成后智能体应:
    1. 查看 03_overlay.png 核对手写/印刷划分 (必要时 --hand-boxes/--print-boxes 重跑)
    2. 查看 topology_annotated.png 逐个核对元件与节点 (必要时 detect --add/--drop/--box/--mask 重跑)
    3. 写 problem.yaml → render 回绘对比 → solve 求解
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from . import ROOT
from .detect import analyze, summarize
from .separate import separate


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="电路题一键分析流水线")
    ap.add_argument("image")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--sep-mode", default="auto", choices=["auto", "color", "pencil", "model", "none"])
    ap.add_argument("--hand-boxes", default=None)
    ap.add_argument("--print-boxes", default=None)
    ap.add_argument("--dark-thr", type=int, default=105)
    ap.add_argument("--sat-thr", type=int, default=28)
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270, -90, -180, -270])
    ap.add_argument("--upscale", type=float, default=1.0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--add", default=None)
    ap.add_argument("--drop", default=None)
    ap.add_argument("--box", default=None)
    ap.add_argument("--mask", default=None)
    ap.add_argument("--crop", default=None, help='只分析电路图区域 "x,y,w,h" (工作图坐标); 题干文字区不参与导线追踪')
    a = ap.parse_args(argv)

    name = Path(a.image).stem
    out = Path(a.out) if a.out else ROOT / "runs" / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{name}"
    out.mkdir(parents=True, exist_ok=True)

    print("=== 1/2 手写/印刷分离 ===")
    sep = separate(a.image, str(out), a.sep_mode, a.hand_boxes, a.print_boxes, a.dark_thr, a.sat_thr,
                   rotate=a.rotate, upscale=a.upscale)
    print(json.dumps({k: sep.get(k) for k in ("size", "scale", "color", "pencil", "model", "hand_share_of_ink", "repaired_pixels", "notes")},
                     ensure_ascii=False, indent=1))

    target = Path(sep["outputs"]["print"])
    if a.crop:
        import cv2
        from .separate import imread, imwrite
        x, y, w, h = [int(v) for v in a.crop.split(",")]
        img = imread(target)
        crop_img = img[y:y + h, x:x + w]
        if min(crop_img.shape[:2]) < 360:
            crop_img = cv2.resize(crop_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        target = out / "01_print_crop.png"
        imwrite(target, crop_img)
        print(f"[crop] 电路区域 -> {target} (之后的坐标相对于裁剪图)")

    print("\n=== 2/2 元件检测 + 导线追踪 ===")
    res = analyze(str(target), str(out), a.conf, a.add, a.drop, a.box, a.mask)
    print(summarize(res))
    print(f"\n[输出目录] {out}")
    print("请查看: 03_overlay.png (手写划分) / topology_annotated.png (拓扑) / netlist_draft.txt")


if __name__ == "__main__":
    main()
