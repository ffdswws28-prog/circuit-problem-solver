"""元件检测 + 导线追踪 → 拓扑草稿

用法:
    python -m circuit_tool.detect <print_image> -o <out_dir>
        [--conf 0.35]
        [--add "cls:x,y,w,h;cls:x,y,w,h"]      人工补框 (受控源/开关/端口等 YOLO 不认识的元件)
        [--drop "R2,C1"]                         删除误检 (用上次输出的 ID)
        [--box "R1:x,y,w,h"]                     修正某个元件的框
        [--mask "x,y,w,h;..."]                   把区域内墨迹当作非导线 (文字/箭头/残留手写)
        [--no-ocr] [--no-yolo]

思路 (CircuitToSPICE 同款"感知 + 几何"分工):
    1. YOLO(HR-DONK/circuit-recognition v3 权重, 10 类) 找元件框
    2. RapidOCR 找文字框 (元件参数/标号), 从导线图里挖掉
    3. 墨迹 - 元件框 - 文字框 = 导线; 连通域 = 电气节点 (net)
    4. 元件框外扩一圈, 与哪些 net 相交 → 元件端子连接
    5. 接地符号接触到的 net 合并为节点 0

输出:
    topology.json         元件/节点/端子/文字 全部结构化
    topology_annotated.png  每个节点不同颜色 + 节点名, 元件框 + ID  ← 智能体逐一核对
    netlist_draft.txt     lcapy 网表草稿 (值来自 OCR 就近匹配, 需人工/智能体确认)
"""
from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import cv2
import numpy as np

from . import VENDOR
from .separate import draw_ruler, imread, imwrite, parse_boxes

YOLO_WEIGHTS_LIST = [
    VENDOR / "circuit-recognition" / "models" / "circuit_yolov8n_v3.pt",
    VENDOR / "circuit-recognition" / "models" / "circuit_yolov8n_v2.pt",
]

PREFIX = {
    "resistor": "R", "capacitor": "C", "inductor": "L", "diode": "D", "transistor": "Q",
    "opamp": "E", "ic": "U", "voltage_source": "V", "current_source": "I", "ground": "GND",
    # 人工补充类别 (YOLO 不支持)
    "vcvs": "E", "vccs": "G", "ccvs": "H", "cccs": "F", "switch": "SW", "port": "P",
    "mosfet": "M", "transformer": "TF", "coupled_inductor": "K", "element": "X", "lamp": "X",
    "ammeter": "AM", "voltmeter": "VM",
}
TWO_TERMINAL = {"resistor", "capacitor", "inductor", "diode", "voltage_source", "current_source",
                "vcvs", "vccs", "ccvs", "cccs", "switch", "element", "lamp", "ammeter", "voltmeter"}


# ----------------------------------------------------------------------------
def run_yolo(img: np.ndarray, conf: float) -> list[dict]:
    from ultralytics import YOLO
    dets = []
    for weights in YOLO_WEIGHTS_LIST:
        if not weights.exists():
            continue
        model = YOLO(str(weights))
        res = model(img, conf=conf, verbose=False)
        for r in res:
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                dets.append({"class": model.names[int(b.cls[0])], "bbox": [x1, y1, x2 - x1, y2 - y1],
                             "confidence": round(float(b.conf[0]), 3), "source": f"yolo:{weights.stem}"})
    return nms_cross_class(dets, thr=0.5)


def iou(a, b) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix = max(0, min(ax1 + aw, bx1 + bw) - max(ax1, bx1))
    iy = max(0, min(ay1 + ah, by1 + bh) - max(ay1, by1))
    inter = ix * iy
    u = aw * ah + bw * bh - inter
    return inter / u if u else 0.0


def nms_cross_class(dets: list[dict], thr: float = 0.6) -> list[dict]:
    dets = sorted(dets, key=lambda d: -d["confidence"])
    keep = []
    for d in dets:
        if all(iou(d["bbox"], k["bbox"]) < thr for k in keep):
            keep.append(d)
    return keep


def run_ocr(img: np.ndarray) -> list[dict]:
    from rapidocr_onnxruntime import RapidOCR
    eng = RapidOCR()
    result, _ = eng(img)
    texts = []
    for box, text, score in (result or []):
        pts = np.array(box, dtype=np.float32)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        texts.append({"text": text, "score": round(float(score), 3),
                      "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]})
    return texts


def assign_ids(comps: list[dict]) -> None:
    counters: dict[str, int] = {}
    # 从上到下、从左到右编号, 便于对照
    for c in sorted(comps, key=lambda c: (c["bbox"][1] // 40, c["bbox"][0])):
        if c.get("id"):
            continue
        p = PREFIX.get(c["class"], "X")
        counters[p] = counters.get(p, 0) + 1
        c["id"] = f"{p}{counters[p]}"


def distinct_colors(n: int) -> list[tuple[int, int, int]]:
    cols = []
    for i in range(max(n, 1)):
        h = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.85)
        cols.append((int(b * 255), int(g * 255), int(r * 255)))
    return cols


# ----------------------------------------------------------------------------
def analyze(image_path: str, out_dir: str, conf: float = 0.25, add: str | None = None, drop: str | None = None,
            fix_boxes: str | None = None, mask: str | None = None, use_ocr: bool = True, use_yolo: bool = True,
            ring: int = 6, ink_thr: int = 170) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    img = imread(image_path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ink = (gray < ink_thr).astype(np.uint8)
    warnings: list[str] = []

    # ---- 1. components
    comps: list[dict] = run_yolo(img, conf) if use_yolo else []
    for part in (add or "").replace("；", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        cls, coords = part.split(":", 1)
        cls = cls.strip()
        cid = None
        if "=" in cls:                      # 允许 "H1=ccvs:x,y,w,h" 指定 ID
            cid, cls = [s.strip() for s in cls.split("=", 1)]
        x, y, bw, bh = [int(float(v)) for v in coords.split(",")]
        comps.append({"class": cls, "bbox": [x, y, bw, bh], "confidence": 1.0, "source": "manual", **({"id": cid} if cid else {})})
    assign_ids(comps)
    drops = {s.strip() for s in (drop or "").split(",") if s.strip()}
    comps = [c for c in comps if c["id"] not in drops]
    for part in (fix_boxes or "").replace("；", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        cid, coords = part.split(":", 1)
        for c in comps:
            if c["id"] == cid.strip():
                c["bbox"] = [int(float(v)) for v in coords.split(",")]
                c["source"] += "+fixed"

    # ---- 2. OCR texts
    texts = run_ocr(img) if use_ocr else []

    # ---- 3. wire mask
    comp_mask = np.zeros_like(ink)
    for c in comps:
        x, y, bw, bh = c["bbox"]
        comp_mask[max(0, y):y + bh, max(0, x):x + bw] = 1
    text_mask = np.zeros_like(ink)
    for t in texts:
        x, y, bw, bh = t["bbox"]
        # 文字框若与元件框大面积重叠, 交给元件框处理; 太长的"文字"可能是导线被误识别
        if t["score"] < 0.5 or bw > 0.5 * w:
            continue
        text_mask[max(0, y):y + bh, max(0, x):x + bw] = 1
    for (x, y, bw, bh) in parse_boxes(mask):
        text_mask[max(0, y):y + bh, max(0, x):x + bw] = 1
    wire = ink & (1 - comp_mask) & (1 - text_mask)
    wire = cv2.morphologyEx(wire, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    wire &= (1 - comp_mask)
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(wire, connectivity=8)

    # ---- 4. terminals: ring around each component box
    contacts: dict[str, dict[int, list]] = {}
    for c in comps:
        x, y, bw, bh = c["bbox"]
        x0, y0 = max(0, x - ring), max(0, y - ring)
        x1, y1 = min(w, x + bw + ring), min(h, y + bh + ring)
        sub = lab[y0:y1, x0:x1].copy()
        inner = np.zeros_like(sub, dtype=bool)
        inner[(y - y0):(y - y0 + bh), (x - x0):(x - x0 + bw)] = True
        sub[inner] = 0
        ids, counts = np.unique(sub[sub > 0], return_counts=True)
        cdict = {}
        for li, cnt in zip(ids.tolist(), counts.tolist()):
            if cnt < 3:
                continue
            ys, xs = np.nonzero(sub == li)
            cdict[li] = [cnt, [int(xs.mean() + x0), int(ys.mean() + y0)]]
        contacts[c["id"]] = cdict

    # 有用的连通域: 至少接触 1 个元件端子
    touched: dict[int, list[str]] = {}
    for cid, cd in contacts.items():
        for li in cd:
            touched.setdefault(li, []).append(cid)

    # ---- 5. ground merge & naming
    ground_ids = {c["id"] for c in comps if c["class"] == "ground"}
    net_name: dict[int, str] = {}
    for li, members in touched.items():
        if any(m in ground_ids for m in members):
            net_name[li] = "0"
    others = [li for li in touched if li not in net_name]
    others.sort(key=lambda li: (stats[li][1] // 40, stats[li][0]))
    for k, li in enumerate(others, 1):
        net_name[li] = f"N{k}"

    nets: dict[str, dict] = {}
    for li, name in net_name.items():
        x, y, bw, bh = stats[li][:5][:4]
        nets.setdefault(name, {"labels": [], "members": [], "bbox": [int(x), int(y), int(bw), int(bh)], "pixels": 0})
        nets[name]["labels"].append(int(li))
        nets[name]["pixels"] += int(stats[li][4])

    for c in comps:
        cd = contacts[c["id"]]
        terms = []
        for li, (cnt, xy) in sorted(cd.items(), key=lambda kv: -kv[1][0]):
            if li not in net_name:
                continue
            nm = net_name[li]
            cx, cy = c["bbox"][0] + c["bbox"][2] / 2, c["bbox"][1] + c["bbox"][3] / 2
            dx, dy = xy[0] - cx, xy[1] - cy
            side = ("right" if dx > 0 else "left") if abs(dx) * c["bbox"][3] > abs(dy) * c["bbox"][2] else ("bottom" if dy > 0 else "top")
            terms.append({"net": nm, "contact_px": cnt, "xy": xy, "side": side})
        # 同一 net 的多个接触合并
        merged = {}
        for t in terms:
            if t["net"] in merged:
                merged[t["net"]]["contact_px"] += t["contact_px"]
            else:
                merged[t["net"]] = t
        c["terminals"] = list(merged.values())
        for t in c["terminals"]:
            nets[t["net"]]["members"].append(f'{c["id"]}.{t["side"]}')
        if c["class"] == "ground":
            continue
        nt = len(c["terminals"])
        if c["class"] in TWO_TERMINAL:
            if nt < 2:
                warnings.append(f'{c["id"]}({c["class"]}) 只找到 {nt} 个端子: 可能被短接/框太大/导线断开, 请核对')
            elif nt > 2:
                warnings.append(f'{c["id"]}({c["class"]}) 找到 {nt} 个端子 (>2): 可能有导线贴着框经过或框过大, 请核对')
            elif c["terminals"][0]["net"] == c["terminals"][1]["net"]:
                warnings.append(f'{c["id"]} 两端在同一节点 (被短路?)')

    # 悬空节点 (只接 1 个端子): 可能是题目端口 a/b, 也可能是识别残片
    for name, n in nets.items():
        if len(n["members"]) == 1 and name != "0":
            n["dangling"] = True
            warnings.append(f'节点 {name} 只接 1 个端子 ({n["members"][0]}): 若不是题目端口(a,b 等), 可能是元件框太小留下的残片')
    unused = [i for i in range(1, n_lab) if i not in touched and stats[i][4] > 150]
    if unused:
        warnings.append(f'{len(unused)} 段较大的墨迹未接触任何元件 (可能是未识别元件/题目文字/残留手写), 已在标注图中用灰色虚框标出')

    # ---- 6. OCR value hints
    for c in comps:
        x, y, bw, bh = c["bbox"]
        cx, cy = x + bw / 2, y + bh / 2
        best = None
        for t in texts:
            tx, ty, tw, th = t["bbox"]
            d = np.hypot(tx + tw / 2 - cx, ty + th / 2 - cy)
            lim = 1.6 * max(bw, bh) + 30
            if d < lim and (best is None or d < best[0]):
                best = (d, t["text"])
        if best:
            c["text_hint"] = best[1]

    # ---- 7. annotated image
    vis = cv2.addWeighted(img, 0.35, np.full_like(img, 255), 0.65, 0)
    names_sorted = sorted(nets.keys(), key=lambda s: (s != "0", len(s), s))
    cols = distinct_colors(len(names_sorted))
    color_of = {nm: ((0, 0, 0) if nm == "0" else cols[i]) for i, nm in enumerate(names_sorted)}
    for nm, n in nets.items():
        m = np.isin(lab, n["labels"]).astype(np.uint8)
        m = cv2.dilate(m, np.ones((3, 3), np.uint8))
        vis[m > 0] = color_of[nm]
    for i in unused:
        x, y, bw, bh = stats[i][:4]
        cv2.rectangle(vis, (int(x), int(y)), (int(x + bw), int(y + bh)), (150, 150, 150), 1, cv2.LINE_4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.5, min(w, h) / 1200)
    for nm, n in nets.items():
        m = np.isin(lab, n["labels"])
        ys, xs = np.nonzero(m)
        k = len(xs) // 2
        px, py = int(xs[k]), int(ys[k])
        label = "GND(0)" if nm == "0" else nm
        cv2.putText(vis, label, (px + 4, py - 4), font, fs, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(vis, label, (px + 4, py - 4), font, fs, color_of[nm], 2, cv2.LINE_AA)
    for c in comps:
        x, y, bw, bh = c["bbox"]
        col = (0, 0, 255) if c["source"].startswith("yolo") else (255, 120, 0)
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), col, 2)
        tag = f'{c["id"]}'
        cv2.putText(vis, tag, (x, max(12, y - 5)), font, fs, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(vis, tag, (x, max(12, y - 5)), font, fs, col, 2, cv2.LINE_AA)
        for t in c.get("terminals", []):
            cv2.circle(vis, tuple(t["xy"]), 5, color_of[t["net"]], -1)
            cv2.circle(vis, tuple(t["xy"]), 5, (0, 0, 0), 1)
    imwrite(out / "topology_annotated.png", draw_ruler(vis))

    # ---- 8. draft netlist
    lines = ["# 网表草稿 (lcapy 语法). 节点名对应 topology_annotated.png 中的颜色标签.",
             "# !!! 元件值/方向/受控源控制量 必须对照题图确认后再求解 !!!"]
    for c in sorted(comps, key=lambda c: c["id"]):
        if c["class"] == "ground":
            continue
        nets_c = [t["net"] for t in c.get("terminals", [])]
        hint = c.get("text_hint", "")
        sides = ",".join(f'{t["net"]}@{t["side"]}' for t in c.get("terminals", []))
        if c["class"] in TWO_TERMINAL and len(nets_c) >= 2:
            lines.append(f'{c["id"]} {nets_c[0]} {nets_c[1]} {{?}} ; class={c["class"]} ocr="{hint}" terminals={sides}')
        else:
            lines.append(f'# {c["id"]} class={c["class"]} terminals={sides} ocr="{hint}"  <- 多端/端子异常, 需手工写')
    (out / "netlist_draft.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = {
        "image": str(image_path), "size": [w, h],
        "components": comps, "nets": nets, "texts": texts, "warnings": warnings,
        "outputs": {"annotated": str(out / "topology_annotated.png"), "netlist_draft": str(out / "netlist_draft.txt")},
        "yolo_classes": "resistor capacitor inductor diode transistor opamp ic ground voltage_source current_source",
        "manual_classes": "vcvs vccs ccvs cccs switch port mosfet transformer coupled_inductor element ammeter voltmeter",
    }
    (out / "topology.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def summarize(res: dict) -> str:
    lines = [f'元件 {len([c for c in res["components"] if c["class"] != "ground"])} 个, 节点 {len(res["nets"])} 个']
    for c in sorted(res["components"], key=lambda c: c["id"]):
        terms = " ".join(f'{t["net"]}({t["side"]})' for t in c.get("terminals", []))
        lines.append(f'  {c["id"]:<5} {c["class"]:<15} conf={c["confidence"]:<5} bbox={c["bbox"]}  -> {terms}  ocr="{c.get("text_hint", "")}"')
    lines.append("节点:")
    for nm, n in sorted(res["nets"].items()):
        lines.append(f'  {nm:<4} <- {", ".join(n["members"])}{"  [悬空]" if n.get("dangling") else ""}')
    if res["texts"]:
        lines.append("OCR 文字: " + " | ".join(f'{t["text"]}@{t["bbox"][0]},{t["bbox"][1]}' for t in res["texts"]))
    for wmsg in res["warnings"]:
        lines.append("⚠ " + wmsg)
    lines.append(f'标注图: {res["outputs"]["annotated"]}')
    lines.append(f'网表草稿: {res["outputs"]["netlist_draft"]}')
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="元件检测 + 导线追踪 → 拓扑草稿")
    ap.add_argument("image")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--add", default=None, help='人工补框 "cls:x,y,w,h;ID=cls:x,y,w,h"')
    ap.add_argument("--drop", default=None, help='删除误检 ID, 逗号分隔')
    ap.add_argument("--box", default=None, help='修正框 "R1:x,y,w,h;..."')
    ap.add_argument("--mask", default=None, help='忽略区域 "x,y,w,h;..." (非导线墨迹)')
    ap.add_argument("--ring", type=int, default=6, help="端子搜索外扩像素")
    ap.add_argument("--ink-thr", type=int, default=170)
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--no-yolo", action="store_true")
    a = ap.parse_args(argv)
    res = analyze(a.image, a.out, a.conf, a.add, a.drop, a.box, a.mask, not a.no_ocr, not a.no_yolo, a.ring, a.ink_thr)
    print(summarize(res))


if __name__ == "__main__":
    main()
