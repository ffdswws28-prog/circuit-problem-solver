# -*- coding: utf-8 -*-
"""V2 vs V3 对比测试：教材风格 + 原始数据集"""
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from ultralytics import YOLO

V2_PATH = _project_root / "models" / "circuit_yolov8n_v2.pt"
V3_PATH = _project_root / "models" / "circuit_yolov8n_v3.pt"
TEXTBOOK_VAL = _project_root / "datasets" / "textbook_synth" / "images" / "val"
CIRCUIT_V2_VAL = _project_root / "datasets" / "circuit_v2" / "images" / "val"

CONF = 0.3


def test_on_dataset(model, name, val_dir, max_images=100):
    """测试模型在指定数据集上的表现"""
    images = sorted(val_dir.glob("*.jpg"))[:max_images]
    t0 = time.time()
    total_dets = 0
    total_gt = 0
    class_dets = Counter()
    class_gt = Counter()

    for img_path in images:
        # 读取GT标签
        lbl_path = val_dir.parent.parent / "labels" / "val" / (img_path.stem + ".txt")
        if lbl_path.exists():
            try:
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            cls_id = int(parts[0])
                            class_gt[model.names[cls_id]] += 1
                        total_gt += len([l for l in f.readlines() if l.strip()])
                    f.seek(0)
                    total_gt = sum(1 for l in f if l.strip())
                with open(lbl_path) as f:
                    total_gt = sum(1 for l in f if l.strip())
            except:
                pass

        results = model(str(img_path), conf=CONF, verbose=False)
        for r in results:
            if r.boxes:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names.get(cls_id, f"class_{cls_id}")
                    class_dets[cls_name] += 1
                    total_dets += 1

    elapsed = time.time() - t0
    rate = len(images) / elapsed if elapsed > 0 else 0
    return {
        "images": len(images),
        "time": elapsed,
        "rate": rate,
        "total_dets": total_dets,
        "class_dets": dict(class_dets),
    }


def main():
    print("=" * 70)
    print("V2 vs V3 对比测试 — 教材合成图检测能力")
    print("=" * 70)

    model_v2 = YOLO(str(V2_PATH))
    model_v3 = YOLO(str(V3_PATH))

    # ── 测试1: 教材合成图验证集 ──
    print(f"\n{'─' * 70}")
    print("测试 1: 教材合成图验证集 (textbook_synth/val)")
    print(f"{'─' * 70}")

    # 统计GT
    lbl_dir = _project_root / "datasets" / "textbook_synth" / "labels" / "val"
    gt_counter = Counter()
    if lbl_dir.exists():
        for lbl_file in lbl_dir.iterdir():
            with open(lbl_file) as f:
                for line in f:
                    cls_id = int(line.strip().split()[0])
                    gt_counter[cls_id] += 1

    expected_classes = ["resistor", "capacitor", "inductor", "diode", "transistor",
                        "opamp", "ic", "ground", "voltage_source", "current_source"]

    print(f"\n  GT 实例数:")
    for i, name in enumerate(expected_classes):
        print(f"    {name:16s}: {gt_counter.get(i, 0):6,d}")
    print(f"    {'TOTAL':16s}: {sum(gt_counter.values()):6,d}")

    r2 = test_on_dataset(model_v2, "V2", TEXTBOOK_VAL)
    r3 = test_on_dataset(model_v3, "V3", TEXTBOOK_VAL)

    print(f"\n  {'指标':20s} {'V2':>15s} {'V3':>15s} {'变化':>15s}")
    print(f"  {'─' * 65}")
    print(f"  {'检测数':20s} {r2['total_dets']:15,d} {r3['total_dets']:15,d} {r3['total_dets']-r2['total_dets']:>+15,d}")
    print(f"  {'速度':20s} {r2['rate']:14.1f} 张/秒 {r3['rate']:14.1f} 张/秒")

    print(f"\n  各类检测对比:")
    print(f"  {'类别':20s} {'V2':>10s} {'V3':>10s} {'变化':>10s}")
    print(f"  {'─' * 55}")
    for name in expected_classes:
        v2c = r2['class_dets'].get(name, 0)
        v3c = r3['class_dets'].get(name, 0)
        delta = v3c - v2c
        sign = '+' if delta >= 0 else ''
        print(f"  {name:20s} {v2c:10,d} {v3c:10,d} {sign}{delta:>9,d}")

    # ── 测试2: 原始验证集 ──
    print(f"\n{'─' * 70}")
    print("测试 2: 原始验证集 (circuit_v2/val — CGHD + HCD)")
    print(f"{'─' * 70}")

    r2_orig = test_on_dataset(model_v2, "V2", CIRCUIT_V2_VAL)
    r3_orig = test_on_dataset(model_v3, "V3", CIRCUIT_V2_VAL)

    print(f"\n  {'指标':20s} {'V2':>15s} {'V3':>15s} {'变化':>15s}")
    print(f"  {'─' * 65}")
    print(f"  {'检测数':20s} {r2_orig['total_dets']:15,d} {r3_orig['total_dets']:15,d} {r3_orig['total_dets']-r2_orig['total_dets']:>+15,d}")
    print(f"  {'速度':20s} {r2_orig['rate']:14.1f} 张/秒 {r3_orig['rate']:14.1f} 张/秒")

    # ── 结论 ──
    print(f"\n{'=' * 70}")
    print("结论")
    print(f"{'=' * 70}")
    textbook_gain = r3['total_dets'] - r2['total_dets']
    orig_change = r3_orig['total_dets'] - r2_orig['total_dets']
    print(f"  教材图检测数: {r2['total_dets']:,d} → {r3['total_dets']:,d} ({textbook_gain:+d})")
    print(f"  原始图检测数: {r2_orig['total_dets']:,d} → {r3_orig['total_dets']:,d} ({orig_change:+d})")

    if textbook_gain > 0:
        print(f"\n  ✅ V3 在教材风格图上检测能力提升了 {textbook_gain} 个实例")
    elif textbook_gain < 0:
        print(f"\n  ⚠️ V3 在教材风格图上反而下降了 {abs(textbook_gain)} 个实例")
    else:
        print(f"\n  ➡️ V3 在教材风格图上检测数不变")

    return 0


if __name__ == "__main__":
    exit(main())
