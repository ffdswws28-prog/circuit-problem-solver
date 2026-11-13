"""网表回绘: 把(确认中的)网表画回图, 与原题图并排对比, 拓扑错误一眼可见

用法:
    python -m circuit_tool.render problem.yaml -o out_dir [--original 01_print.png] [--schematic]

输出:
    netgraph.png      节点(圆) — 元件(方块) 二部图, 不需要任何布局提示, 永远可画
    compare.png       原题图 | 网表图 并排 (给了 --original 时)
    schematic.png     lcapy+circuitikz 规范电路图 (需 --schematic 且网表带 right/down 等布局提示, 需 MiKTeX)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from .separate import imread, imwrite


def parse_elements(netlist: str):
    from lcapy import Circuit
    c = Circuit(netlist)
    elems = []
    for name, e in c.elements.items():
        nodes = list(getattr(e, "node_names", []) or [])
        if not nodes:
            continue
        ctrl = ""
        ctrl_nodes = ()
        ctrl_elem = ""
        tn = type(e).__name__
        if tn in ("E", "G") and len(nodes) >= 4:
            ctrl_nodes = (nodes[2], nodes[3])
            ctrl = f"ctrl V({nodes[2]},{nodes[3]})"
            nodes = nodes[:2]
        elif tn in ("F", "H"):
            try:
                ctrl_elem = str(e.args[0])
                ctrl = f"ctrl I({ctrl_elem})"
            except Exception:
                ctrl = "ctrl I(?)"
        elems.append({
            "name": name,
            "type": tn,
            "nodes": nodes,
            "ctrl": ctrl,
            "ctrl_nodes": ctrl_nodes,
            "ctrl_elem": ctrl_elem,
        })
    return c, elems


def extract_query_ports(prob: dict) -> set[str]:
    """从 queries 提取待测端口/开路节点白名单 (避免合法单引脚端口误报悬空)."""
    ports: set[str] = set()
    for q in prob.get("queries", []) or []:
        if isinstance(q, dict):
            continue
        m = re.fullmatch(r"\s*(thevenin|norton|Z|V|v)\((.*)\)\s*", str(q))
        if not m:
            continue
        for arg in m.group(2).split(","):
            name = arg.strip()
            if name:
                ports.add(name)
    return ports


def audit_netlist(elems: list[dict], port_nodes: set[str] | None = None) -> tuple[list[str], list[str]]:
    """确定性 ERC 拓扑校验 + 显式物理方向断言 (零延迟)."""
    erc_warnings: list[str] = []
    dir_lines: list[str] = []
    port_nodes = port_nodes or set()

    elem_map = {e["name"]: e for e in elems}
    all_nodes: set[str] = set()
    deg: dict[str, int] = {}
    for e in elems:
        if e["type"] in ("K", "W"):
            continue
        for n in e["nodes"]:
            all_nodes.add(n)
            deg[n] = deg.get(n, 0) + 1

    if "0" not in all_nodes:
        erc_warnings.append("[ERC错误] 缺少参考地节点 '0'!")

    for n in sorted(all_nodes):
        if deg.get(n, 0) < 2:
            if n in port_nodes:
                dir_lines.append(f"[端口节点] 节点 [{n}] 为待测开路端口 (单引脚正常)")
            else:
                erc_warnings.append(f"[ERC警告] 节点 [{n}] 仅连接 {deg.get(n, 0)} 个引脚，疑似断线或命名笔误!")

    for e in elems:
        tn = e["type"]
        nm = e["name"]
        ns = e["nodes"]
        if len(ns) < 2:
            continue
        n1, n2 = ns[0], ns[1]
        if tn == "V":
            dir_lines.append(f"[电压源极性] {nm}: [{n1}](+) — [{n2}](-)")
        elif tn == "I":
            dir_lines.append(f"[电流源方向] {nm}: 源内从 [{n2}] -> 流向 [{n1}] (箭头指向第1节点 [{n1}])")
        elif tn == "G":
            nc1, nc2 = e["ctrl_nodes"] if len(e["ctrl_nodes"]) == 2 else ("?", "?")
            dir_lines.append(f"[受控流源 VCCS] {nm}: 从 [{n2}] -> 流向 [{n1}] | 控制 V({nc1})-V({nc2})")
        elif tn == "F":
            ce = e["ctrl_elem"]
            c_info = f"{elem_map[ce]['nodes'][0]}->{elem_map[ce]['nodes'][1]}" if ce in elem_map and len(elem_map[ce]["nodes"]) >= 2 else ce
            dir_lines.append(f"[受控流源 CCCS] {nm}: 从 [{n1}] -> 流向 [{n2}] (SPICE:箭头指向第2节点 [{n2}]) | 控制 {ce}({c_info})")
        elif tn == "E":
            nc1, nc2 = e["ctrl_nodes"] if len(e["ctrl_nodes"]) == 2 else ("?", "?")
            dir_lines.append(f"[受控压源 VCVS] {nm}: [{n1}](+) — [{n2}](-) | 控制 V({nc1})-V({nc2})")
        elif tn == "H":
            ce = e["ctrl_elem"]
            dir_lines.append(f"[受控压源 CCVS] {nm}: [{n1}](+) — [{n2}](-) | 控制 I({ce})")

    return erc_warnings, dir_lines


def draw_netgraph(elems, path: Path, title: str = "", audit_lines: list[str] | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    G = nx.Graph()
    nodes = sorted({n for e in elems for n in e["nodes"]})
    for n in nodes:
        G.add_node(("n", n))
    for e in elems:
        G.add_node(("e", e["name"]))
        for k, n in enumerate(e["nodes"]):
            G.add_edge(("e", e["name"]), ("n", n), pin=k)
    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=1)
    extra_h = 0.35 * min(len(audit_lines or []), 8)
    fig, ax = plt.subplots(figsize=(9, 7 + extra_h), dpi=130)
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.6, edge_color="#555")
    nn = [x for x in G if x[0] == "n"]
    ee = [x for x in G if x[0] == "e"]
    nx.draw_networkx_nodes(G, pos, nodelist=nn, node_shape="o", node_size=900,
                           node_color=["#222" if x[1] == "0" else "#4C9BE8" for x in nn], ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=ee, node_shape="s", node_size=1300, node_color="#FFD966",
                           edgecolors="#B8860B", ax=ax)
    nx.draw_networkx_labels(G, pos, labels={x: ("GND" if x[1] == "0" else x[1]) for x in nn},
                            font_color="white", font_weight="bold", ax=ax)
    lab = {}
    for e in elems:
        lab[("e", e["name"])] = e["name"] + ("\n" + e["ctrl"] if e["ctrl"] else "")
    nx.draw_networkx_labels(G, pos, labels=lab, font_size=9, ax=ax)
    # 标出端子序号 (1=第一节点: 电压源+极 / 电流源箭头所指 / 参考方向起点)
    for (u, v, d) in G.edges(data=True):
        a, b = (u, v) if u[0] == "e" else (v, u)
        p = 0.72 * np.array(pos[a]) + 0.28 * np.array(pos[b])
        ax.text(p[0], p[1], str(d["pin"] + 1), fontsize=8, color="#C00000", ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.1", fc="white", ec="#C00000", lw=0.6))
    ax.set_title((title + "\n" if title else "") + "圆=节点  方块=元件  红色数字=端子序号(1=网表第1节点)", fontsize=10)
    ax.axis("off")
    if audit_lines:
        fig.tight_layout(rect=[0, 0.20, 1, 1])
        txt = "\n".join(audit_lines[:8])
        fig.text(0.03, 0.02, txt, fontsize=8.5, va="bottom", ha="left",
                 family="Microsoft YaHei",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#FFF9E6", ec="#D4A017", lw=0.9))
    else:
        fig.tight_layout()
    fig.savefig(str(path))
    plt.close(fig)


def side_by_side(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    h = max(a.shape[0], b.shape[0])
    def fit(x):
        k = h / x.shape[0]
        return cv2.resize(x, (int(x.shape[1] * k), h))
    a, b = fit(a), fit(b)
    sep = np.full((h, 12, 3), 200, np.uint8)
    return np.hstack([a, sep, b])


def draw_schematic(c, path: Path) -> str:
    pdf = path.with_suffix(".pdf")
    try:
        c.draw(str(pdf))
    except Exception as ex:
        return f"lcapy 绘图失败 (网表需要 right/down/left/up 布局提示, 并安装 circuitikz): {ex}"
    try:
        import fitz
        doc = fitz.open(str(pdf))
        pix = doc[0].get_pixmap(dpi=200)
        pix.save(str(path))
        return f"已生成 {path}"
    except Exception as ex:
        return f"PDF 转 PNG 失败: {ex} (PDF 在 {pdf})"


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="网表回绘与对比")
    ap.add_argument("problem", help="problem.yaml (含 netlist) 或纯网表 .txt")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--original", default=None, help="原题图 (印刷层) 用于并排对比")
    ap.add_argument("--schematic", action="store_true", help="尝试用 lcapy 画规范电路图")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    text = Path(a.problem).read_text(encoding="utf-8")
    title = ""
    port_nodes: set[str] = set()
    if a.problem.endswith((".yaml", ".yml")):
        prob = yaml.safe_load(text)
        net = prob["netlist"] if isinstance(prob["netlist"], str) else "\n".join(prob["netlist"])
        title = prob.get("title", "")
        port_nodes = extract_query_ports(prob)
    else:
        net = "\n".join(l for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#"))
    c, elems = parse_elements(net)
    erc_warnings, dir_lines = audit_netlist(elems, port_nodes=port_nodes)
    audit_all = erc_warnings + dir_lines
    draw_netgraph(elems, out / "netgraph.png", title, audit_lines=audit_all)
    print(f"[netgraph] {out / 'netgraph.png'}")
    print("元件连接:")
    for e in elems:
        print(f"  {e['name']:<6} {e['type']:<4} {' — '.join(e['nodes'])} {e['ctrl']}")
    report_lines = ["[ERC与方向自检报告]"]
    if erc_warnings:
        report_lines.extend(erc_warnings)
    else:
        report_lines.append("[ERC通过] 接地节点完整，未发现单引脚悬空节点。")
    report_lines.extend(dir_lines)
    report_txt = "\n".join(report_lines)
    (out / "audit_report.txt").write_text(report_txt + "\n", encoding="utf-8")
    print(report_txt)
    if a.original:
        cmp_img = side_by_side(imread(a.original), imread(out / "netgraph.png"))
        imwrite(out / "compare.png", cmp_img)
        print(f"[compare] {out / 'compare.png'}")
    if a.schematic:
        print("[schematic]", draw_schematic(c, out / "schematic.png"))


if __name__ == "__main__":
    main()
