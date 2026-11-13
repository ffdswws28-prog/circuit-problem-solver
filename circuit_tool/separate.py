"""手写 / 印刷分离

用法:
    python -m circuit_tool.separate <image> -o <out_dir> [--mode auto|color|pencil|model|none]
                                    [--hand-boxes "x,y,w,h;x,y,w,h"] [--print-boxes "..."]
                                    [--dark-thr 105] [--sat-thr 60]

输出（out_dir）:
    00_normalized.png   背景拉平后的图
    01_print.png        印刷层（题目原始信息，手写已擦除）  ← 后续识别只用它
    02_hand.png         手写层（学生批注，仅供参考，绝不当作已知条件）
    03_overlay.png      对比图：印刷=灰黑，彩色笔=洋红，铅笔=橙色，人工框=青色
    separation.json     统计信息

策略:
    1. color  : 归一化后 HSV 饱和度高的墨迹 → 彩色笔手写（蓝/红/绿笔）
    2. pencil : 墨迹"最暗核心"明显浅于印刷油墨 → 铅笔手写
    3. model  : 若存在 HandWritingEraser (DeepLabV3+, 中文试卷训练) 权重, 用模型识别黑笔手写
    4. 人工框 : --hand-boxes 强制标为手写, --print-boxes 强制标为印刷 (智能体看图后修正用)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import ROOT, VENDOR

MODEL_DIR = ROOT / "models" / "handwriting_eraser"


# ----------------------------------------------------------------------------
# IO helpers (支持中文路径)
# ----------------------------------------------------------------------------
def imread(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return img


def imwrite(path: str | Path, img: np.ndarray) -> None:
    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"写入失败: {path}")
    buf.tofile(str(path))


def parse_boxes(s: str | None) -> list[tuple[int, int, int, int]]:
    if not s:
        return []
    boxes = []
    for part in s.replace("；", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        x, y, w, h = [int(float(v)) for v in part.split(",")]
        boxes.append((x, y, w, h))
    return boxes


# ----------------------------------------------------------------------------
# Core steps
# ----------------------------------------------------------------------------
def resize_max(img: np.ndarray, max_side: int = 2000) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img, 1.0
    k = max_side / s
    return cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA), k


def normalize_background(img: np.ndarray) -> np.ndarray:
    """拉平光照/纸张颜色: 每个通道除以估计的背景 (大核闭运算 + 中值滤波)."""
    h, w = img.shape[:2]
    k = max(15, (min(h, w) // 40) | 1)
    out = np.empty_like(img)
    for c in range(3):
        ch = img[:, :, c]
        bg = cv2.morphologyEx(ch, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        bg = cv2.medianBlur(bg, 21 if k > 21 else (k | 1))
        norm = np.clip(ch.astype(np.float32) / np.maximum(bg.astype(np.float32), 1) * 255.0, 0, 255)
        out[:, :, c] = norm.astype(np.uint8)
    return out


def ink_mask(norm: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
    otsu_t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    t = min(max(otsu_t, 150), 215)
    return (gray < t).astype(np.uint8)


def color_hand_mask(norm: np.ndarray, ink: np.ndarray, sat_thr: int = 28) -> tuple[np.ndarray, dict]:
    hsv = cv2.cvtColor(norm, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    b = norm[:, :, 0].astype(np.int16)
    r = norm[:, :, 2].astype(np.int16)
    br_diff = np.abs(b - r)
    br_thr = max(15, (sat_thr * 2) // 3)
    seed = ((ink > 0) & ((s >= sat_thr) | (br_diff >= br_thr)) & (v > 25)).astype(np.uint8)
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    fringe = ((ink > 0) & ((s >= 16) | (br_diff >= 10))).astype(np.uint8)
    m = cv2.dilate(seed, np.ones((5, 5), np.uint8)) & fringe
    m = cv2.dilate(m, np.ones((3, 3), np.uint8)) & ink
    hues = hsv[:, :, 0][m > 0]
    info = {"pixels": int(m.sum())}
    if hues.size:
        hist = np.bincount((hues // 15).astype(int), minlength=12)
        names = ["red", "orange", "yellow", "yellow-green", "green", "green-cyan",
                 "cyan", "blue", "blue", "violet", "magenta", "red"]
        top = np.argsort(hist)[::-1][:3]
        info["dominant_hues"] = [{"hue": names[i], "share": round(float(hist[i] / hues.size), 3)} for i in top if hist[i] > 0]
    return m, info


def pencil_hand_mask(norm: np.ndarray, ink: np.ndarray, exclude: np.ndarray, dark_thr: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """铅笔: 墨迹最暗处都明显比印刷油墨浅.

    返回 (pencil_mask, print_core_region, info)
    """
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
    cand = (ink > 0) & (exclude == 0)
    dark_core = (gray < dark_thr) & cand
    # 印刷笔画的抗锯齿边缘较浅, 用暗核膨胀覆盖
    print_region = cv2.dilate(dark_core.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2)
    light = cand & (print_region == 0)
    light = light.astype(np.uint8)
    # 只保留有一定规模的浅色连通块 (排除纸张纹理/压缩噪声)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(light, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= 25 and max(w, h) >= 8:
            keep[i] = True
    pencil = keep[lab].astype(np.uint8)
    ink_px = max(int(cand.sum()), 1)
    info = {"pixels": int(pencil.sum()), "share_of_ink": round(float(pencil.sum()) / ink_px, 4), "dark_thr": dark_thr}
    return pencil, print_region, info


def model_hand_mask(img_bgr: np.ndarray) -> tuple[np.ndarray | None, str]:
    """HandWritingEraser-Pytorch (DeepLabV3+) 推理. 无权重时返回 None."""
    weights = sorted(MODEL_DIR.glob("*.pth")) if MODEL_DIR.exists() else []
    if not weights:
        return None, f"未找到模型权重 (放到 {MODEL_DIR}\\*.pth 即可启用)"
    import sys
    import torch

    repo = next(VENDOR.glob("HandWritingEraser-Pytorch*"), None)
    if repo is None:
        return None, "vendor/HandWritingEraser-Pytorch 不存在"
    sys.path.insert(0, str(repo))
    import network  # type: ignore

    ckpt = torch.load(str(weights[0]), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model = None
    for builder in ("deeplabv3plus_resnet101", "deeplabv3plus_resnet50", "deeplabv3plus_mobilenet"):
        try:
            m = getattr(network.modeling, builder)(num_classes=3, output_stride=16)
            m.load_state_dict(state)
            model = m
            break
        except Exception:
            continue
    if model is None:
        return None, f"权重 {weights[0].name} 与已知结构不匹配"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    tile = 1024
    out = np.zeros((h, w), dtype=np.uint8)
    with torch.no_grad():
        for y0 in range(0, h, tile):
            for x0 in range(0, w, tile):
                patch = rgb[y0:y0 + tile, x0:x0 + tile]
                t = torch.from_numpy(patch).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                pred = model(t.to(device)).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
                out[y0:y0 + tile, x0:x0 + tile] = pred
    hand = (out == 1).astype(np.uint8)
    hand = cv2.dilate(hand, np.ones((2, 2), np.uint8))
    return hand, f"model={weights[0].name}"


def draw_ruler(img: np.ndarray, step: int = 100) -> np.ndarray:
    """在图像顶部和左侧边缘绘制像素坐标标尺, 便于读取精确坐标."""
    out = img.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    half = max(1, step // 2)
    color = (80, 80, 80)
    # 顶部横轴刻度
    for x in range(0, w, half):
        is_major = (x % step == 0)
        tick_len = 10 if is_major else 5
        cv2.line(out, (x, 0), (x, tick_len), color, 1)
        if is_major and x > 0:
            cv2.putText(out, str(x), (x + 2, 20), font, 0.4, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(out, str(x), (x + 2, 20), font, 0.4, color, 1, cv2.LINE_AA)
    # 左侧纵轴刻度
    for y in range(0, h, half):
        is_major = (y % step == 0)
        tick_len = 10 if is_major else 5
        cv2.line(out, (0, y), (tick_len, y), color, 1)
        if is_major and y > 0:
            cv2.putText(out, str(y), (12, y + 4), font, 0.4, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(out, str(y), (12, y + 4), font, 0.4, color, 1, cv2.LINE_AA)
    return out


def repair_wire_gaps(print_img: np.ndarray, erased: np.ndarray, max_gap: int = 120) -> np.ndarray:
    """在被擦除的区域内, 用水平/竖直闭运算接上共线的导线断口. 原地修改 print_img, 返回补上的像素掩码."""
    gray = cv2.cvtColor(print_img, cv2.COLOR_BGR2GRAY)
    ink = (gray < 140).astype(np.uint8)
    # 只考虑"像导线"的墨迹: 长直线段 (开运算保留长度 >= 15 的水平/竖直笔画)
    h_lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, 15), np.uint8))
    v_lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((15, 1), np.uint8))
    zone = cv2.dilate(erased, np.ones((5, 5), np.uint8))
    fill = np.zeros_like(ink)
    for lines, k in ((h_lines, (1, max_gap)), (v_lines, (max_gap, 1))):
        closed = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, np.ones(k, np.uint8))
        fill |= (closed & zone & (1 - ink))
    if fill.any():
        val = int(np.median(gray[(h_lines | v_lines) > 0])) if (h_lines | v_lines).any() else 30
        print_img[fill > 0] = (val, val, val)
    return fill


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def separate(image_path: str, out_dir: str, mode: str = "auto", hand_boxes=None, print_boxes=None,
             dark_thr: int = 105, sat_thr: int = 28, max_side: int = 2000,
             rotate: int = 0, upscale: float = 1.0) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    img0 = imread(image_path)
    rot_map = {
        90: cv2.ROTATE_90_CLOCKWISE, -270: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180, -180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE, -90: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    if rotate in rot_map:
        img0 = cv2.rotate(img0, rot_map[rotate])
    if upscale > 1.0:
        img0 = cv2.resize(img0, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    img, scale = resize_max(img0, max_side)
    norm = normalize_background(img)
    ink = ink_mask(norm)
    h, w = ink.shape
    ink_px = max(int(ink.sum()), 1)
    report: dict = {"input": str(image_path), "size": [w, h], "scale": scale, "rotate": rotate,
                    "upscale": upscale, "mode": mode, "notes": []}

    zero = np.zeros_like(ink)
    color_m, pencil_m, model_m, print_region = zero, zero, zero, zero

    if mode in ("auto", "color", "pencil", "model"):
        color_m, cinfo = color_hand_mask(norm, ink, sat_thr)
        report["color"] = cinfo
    if mode in ("auto", "pencil"):
        gray_norm = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
        p15_gray = float(np.percentile(gray_norm[ink > 0], 15)) if ink.any() else 255.0
        if mode == "auto" and (color_m.sum() / ink_px > 0.02 or p15_gray > 95):
            report["notes"].append(f"auto 跳过铅笔擦除 (彩色手写占比={color_m.sum() / ink_px:.3f}, 墨迹p15灰度={p15_gray:.1f})")
        else:
            pencil_m, print_region, pinfo = pencil_hand_mask(norm, ink, color_m, dark_thr)
            report["pencil"] = pinfo
            if mode == "auto" and pinfo["share_of_ink"] < 0.01:
                pencil_m = zero
                report["notes"].append("浅色墨迹占比 <1%，判定无铅笔手写")
    if mode in ("auto", "model"):
        mm, msg = model_hand_mask(img)
        report["model"] = msg
        if mm is not None:
            model_m = mm & ink
    hand = ((color_m | pencil_m | model_m) > 0).astype(np.uint8)

    manual = np.zeros_like(ink)
    for (x, y, bw, bh) in parse_boxes(hand_boxes):
        manual[y:y + bh, x:x + bw] = 1
    manual &= ink
    hand |= manual
    for (x, y, bw, bh) in parse_boxes(print_boxes):
        hand[y:y + bh, x:x + bw] = 0
        color_m[y:y + bh, x:x + bw] = 0
        pencil_m[y:y + bh, x:x + bw] = 0
        model_m[y:y + bh, x:x + bw] = 0
        manual[y:y + bh, x:x + bw] = 0

    # 印刷层: 擦掉手写 (彩色笔迹扩一圈去光晕, 但不擦印刷暗核)
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
    erase = cv2.dilate(hand, np.ones((3, 3), np.uint8))
    if print_region.any():
        erase &= (1 - print_region)
    erase_colored = cv2.dilate(color_m, np.ones((3, 3), np.uint8))
    erase = ((erase | erase_colored | manual) > 0)
    print_img = norm.copy()
    print_img[erase] = 255
    # 修补: 手写压在导线上被擦掉后留下的缺口 (仅沿水平/竖直方向, 仅在擦除区域内)
    bridged = repair_wire_gaps(print_img, erase.astype(np.uint8))
    report["repaired_pixels"] = int(bridged.sum())
    # 手写层
    hand_img = np.full_like(norm, 255)
    hand_img[hand > 0] = norm[hand > 0]
    # 对比图
    faded = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    faded = cv2.addWeighted(faded, 0.55, np.full_like(faded, 255), 0.45, 0)
    overlay = faded.copy()
    overlay[(ink > 0) & ~erase] = (40, 40, 40)
    overlay[pencil_m > 0] = (0, 140, 255)
    overlay[model_m > 0] = (0, 140, 255)
    overlay[color_m > 0] = (255, 0, 255)
    overlay[manual > 0] = (255, 200, 0)
    for (x, y, bw, bh) in parse_boxes(hand_boxes):
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (255, 200, 0), 2)
    cv2.putText(overlay, "print=black  color-pen=magenta  pencil/model=orange  manual=cyan",
                (45, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
    overlay = draw_ruler(overlay)

    imwrite(out / "00_normalized.png", norm)
    imwrite(out / "01_print.png", print_img)
    imwrite(out / "02_hand.png", hand_img)
    imwrite(out / "03_overlay.png", overlay)
    ink_px = max(int(ink.sum()), 1)
    report["hand_share_of_ink"] = round(float(hand.sum()) / ink_px, 4)
    report["outputs"] = {k: str(out / f) for k, f in [("normalized", "00_normalized.png"), ("print", "01_print.png"),
                                                      ("hand", "02_hand.png"), ("overlay", "03_overlay.png")]}
    report["coords_note"] = "所有输出图共用同一坐标系 (工作图 = 原图 × scale); 人工框也用此坐标"
    (out / "separation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="手写/印刷分离")
    ap.add_argument("image")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--mode", default="auto", choices=["auto", "color", "pencil", "model", "none"])
    ap.add_argument("--hand-boxes", default=None, help='强制手写区域 "x,y,w,h;..." (工作图坐标, 即输出图坐标)')
    ap.add_argument("--print-boxes", default=None, help='强制印刷区域 "x,y,w,h;..." (工作图坐标, 即输出图坐标)')
    ap.add_argument("--dark-thr", type=int, default=105, help="印刷油墨最暗核心阈值 (铅笔判别)")
    ap.add_argument("--sat-thr", type=int, default=28, help="彩色笔饱和度阈值")
    ap.add_argument("--max-side", type=int, default=2000)
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270, -90, -180, -270], help="输入图像旋转角度(正=顺时针, -90=逆时针90度)")
    ap.add_argument("--upscale", type=float, default=1.0, help="输入图像双三次放大倍数")
    a = ap.parse_args(argv)
    rep = separate(a.image, a.out, a.mode, a.hand_boxes, a.print_boxes, a.dark_thr, a.sat_thr, a.max_side, a.rotate, a.upscale)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
