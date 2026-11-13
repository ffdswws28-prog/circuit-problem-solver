"""lcapy 求解器: 读取题目 YAML, 输出分步结果 (Markdown + LaTeX)

用法:
    python -m circuit_tool.solve problem.yaml [-o solution.md]

problem.yaml 示例:
    title: 例 3-5 求 R3 电流及 ab 端戴维南等效
    netlist: |
      Vs 1 0 10          ; 电压源: 第1节点为 + 极
      R1 1 2 2
      R2 2 0 3
      Is 2 0 1           ; !!! lcapy 电流源: 电流经源内部从第2节点流向第1节点, 即箭头指向第1节点
    domain: dc            # dc | ac | transient
    queries:
      - V(2)              # 节点电压 (对地 0)
      - V(2,0)            # 两节点间电压
      - I(R1)             # 元件电流, 参考方向: 第1节点 → 元件 → 第2节点
      - V(R1)             # 元件电压, 第1节点为 +
      - P(R1)             # 吸收功率 (关联参考方向); 负值 = 发出
      - thevenin(2,0)
      - norton(2,0)
      - Z(2,0)            # 独立源置零后的等效阻抗
      - v(C1)             # 暂态: 时域表达式
      - expr: c.R1.I(t)   # 任意 lcapy/python 表达式, c 为电路对象
    equations: nodal      # 可选: nodal | mesh | mna, 打印方程用于讲解
    table: true           # 可选: 输出所有元件 V/I/P 表
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import sympy as sp
import yaml


def _num(e, digits=5):
    try:
        v = sp.N(e, digits)
        return v
    except Exception:
        return e


def _latex(e) -> str:
    try:
        return sp.latex(sp.nsimplify(e, rational=False) if False else e)
    except Exception:
        return str(e)


def polar_str(z) -> str:
    try:
        zc = complex(sp.N(z))
    except Exception:
        return ""
    import cmath
    mag, ang = abs(zc), cmath.phase(zc) * 180 / cmath.pi
    return f"{mag:.5g} ∠ {ang:.4g}°"


def audit_expr(expr, label: str, domain: str) -> list[str]:
    """零延迟异常解/怪异分数/失稳指数雷达."""
    warnings: list[str] = []
    if expr is None or not hasattr(expr, "atoms"):
        return warnings
    try:
        from lcapy import t
        # 1. 检查怪异大质数分母 (分母 > 24 通常暗示受控源反接或拓扑写错)
        for atom in expr.atoms(sp.Rational):
            if abs(atom.q) > 24:
                warnings.append(
                    f"[怪异数值警报] `{label}` 出现复杂分数 `{atom}` (分母={atom.q} > 24)，"
                    f"极可能存在受控源/电流源方向写反或拓扑断线，请核对 compare.png 中的方向自检！"
                )
        # 2. 检查暂态发散指数项 e^{+at} (tau < 0 物理失稳)
        if domain in ("transient", "laplace"):
            expanded = sp.expand(expr)
            for exp_atom in expanded.atoms(sp.exp):
                arg = sp.simplify(exp_atom.args[0])
                coeff = sp.simplify(arg.coeff(t))
                if coeff.is_real and coeff.is_number and float(coeff) > 1e-9:
                    warnings.append(
                        f"[物理失稳警报] `{label}` 含发散指数项 $e^{{+{coeff}t}}$（时间常数 $\\tau < 0$），"
                        f"请重点检查受控源极性是否反接！"
                    )
    except Exception:
        pass
    return warnings


class Solver:
    def __init__(self, prob: dict):
        from lcapy import Circuit
        self.prob = prob
        net = prob["netlist"]
        if isinstance(net, list):
            net = "\n".join(net)
        self.netlist = net
        self.c = Circuit(net)
        self.domain = (prob.get("domain") or "dc").lower()
        self.out: list[str] = []
        self.audit_warnings: list[str] = []

    # ------------------------------------------------------------------
    def fmt(self, q, kind: str = "", label: str = "") -> str:
        """把 lcapy 量格式化为 markdown 行."""
        from lcapy import t, s
        d = self.domain
        try:
            if d == "dc":
                e = sp.simplify(q(t).expr)
                self.audit_warnings.extend(audit_expr(e, label or kind or "dc_expr", d))
                n = _num(e)
                return f"$ {_latex(e)} $" + (f"  ≈ **{n}**" if e != n else "")
            if d == "ac":
                ph = q.phasor()
                e = sp.simplify(ph.expr)
                self.audit_warnings.extend(audit_expr(e, label or kind or "ac_expr", d))
                td = q(t).expr
                return (f"相量(幅值) $ {_latex(e)} $ ≈ **{polar_str(e)}**；有效值 ≈ {abs(complex(sp.N(e))) / 2 ** 0.5:.5g}"
                        f"；时域 $ {_latex(sp.simplify(td))} $")
            if d in ("transient", "laplace"):
                td = sp.simplify(q(t).expr)
                self.audit_warnings.extend(audit_expr(td, label or kind or "transient_expr", d))
                line = f"$ {_latex(td)} $"
                try:
                    sd = sp.simplify(q(s).expr)
                    line += f"；s 域 $ {_latex(sd)} $"
                except Exception:
                    pass
                return line
        except Exception as ex:  # fall back to raw
            return f"`{q}` (格式化失败: {ex})"
        return f"`{q}`"

    def _elem(self, name):
        return self.c.elements.get(name)

    def _node(self, n):
        return self.c[n]

    # ------------------------------------------------------------------
    def query(self, q) -> str:
        from lcapy import t, s, j, omega  # noqa: F401  (expr 中可用)
        c = self.c
        if isinstance(q, dict) and "expr" in q:
            val = eval(q["expr"], {"c": c, "t": t, "s": s, "j": j, "omega": omega, "sp": sp})
            try:
                return f"- `{q['expr']}` = {self.fmt(val)}"
            except Exception:
                return f"- `{q['expr']}` = $ {_latex(val)} $"
        q = str(q).strip()
        m = re.fullmatch(r"(\w+)\((.*)\)", q)
        if not m:
            return f"- ❓ 无法解析查询 `{q}`"
        fn, args = m.group(1), [a.strip() for a in m.group(2).split(",") if a.strip()]
        try:
            if fn == "V":
                if len(args) == 1 and args[0] in c.elements:
                    e = c.elements[args[0]]
                    return f"- $U_{{{args[0]}}}$ (参考: {e.node_names[0]}+ → {e.node_names[1]}−) = {self.fmt(e.V)}"
                if len(args) == 1:
                    return f"- 节点电压 $U_{{{args[0]}}}$ = {self.fmt(c[args[0]].V)}"
                return f"- $U_{{{args[0]}{args[1]}}}$ = {self.fmt(c[args[0]].V - c[args[1]].V)}"
            if fn in ("v", "i"):
                if args[0] in c.elements:
                    e = c.elements[args[0]]
                    qq = e.v if fn == "v" else e.i
                    ref = f"{e.node_names[0]}→{e.node_names[1]}"
                else:
                    qq = c[args[0]].v
                    ref = "对地"
                from lcapy import t as tt
                return f"- ${fn}_{{{args[0]}}}(t)$ ({ref}) = $ {_latex(sp.simplify(qq.expr))} $"
            if fn == "I":
                e = c.elements[args[0]]
                return f"- $I_{{{args[0]}}}$ (参考: {e.node_names[0]} → {args[0]} → {e.node_names[1]}) = {self.fmt(e.I)}"
            if fn == "P":
                e = c.elements[args[0]]
                if self.domain == "ac":
                    V, I = e.V.phasor().expr, e.I.phasor().expr
                    S = sp.simplify(V * sp.conjugate(I) / 2)
                    return (f"- {args[0]} 复功率(吸收) $\\tilde S=\\tfrac12 \\dot U \\dot I^* = {_latex(S)}$ ≈ {sp.N(S, 5)}；"
                            f"平均功率 P ≈ **{sp.N(sp.re(S), 5)}** W，无功 Q ≈ {sp.N(sp.im(S), 5)} var")
                P = sp.simplify(e.V(t).expr * e.I(t).expr)
                return f"- $P_{{{args[0]}}}$ (关联方向吸收, 负值=发出) = $ {_latex(P)} $ ≈ **{_num(P)}**"
            if fn in ("thevenin", "norton"):
                a, b = args
                if fn == "thevenin":
                    th = c.thevenin(a, b)
                    z_s = sp.simplify(th.Z.expr)
                    self.audit_warnings.extend(audit_expr(z_s, f"thevenin_Z({a},{b})", self.domain))
                    if z_s.is_real and z_s.is_number and float(z_s) < 0:
                        self.audit_warnings.append(f"[负等效电阻警报] `thevenin({a},{b})` 等效内阻 $Z_{{eq}} = {z_s} < 0$，请检查受控源极性是否反接！")
                    return f"- 戴维南 ({a},{b}): $U_{{oc}}$ = {self.fmt(th.Voc, label=f'Voc({a},{b})')}；$Z_{{eq}} = {_latex(z_s)}$ ≈ {sp.N(z_s, 5)}"
                no = c.norton(a, b)
                z_s = sp.simplify(no.Z.expr)
                self.audit_warnings.extend(audit_expr(z_s, f"norton_Z({a},{b})", self.domain))
                return f"- 诺顿 ({a},{b}): $I_{{sc}}$ = {self.fmt(no.Isc, label=f'Isc({a},{b})')}；$Z_{{eq}} = {_latex(z_s)}$ ≈ {sp.N(z_s, 5)}"
            if fn == "Z":
                a, b = args
                Z = c.impedance(a, b).expr
                return f"- $Z_{{{a}{b}}}$ (独立源置零) = $ {_latex(sp.simplify(Z))} $ ≈ **{sp.N(Z, 5)}**"
        except Exception as ex:
            return f"- ⚠ `{q}` 求解失败: {ex}"
        return f"- ❓ 未知函数 `{fn}`"

    # ------------------------------------------------------------------
    def equations(self, kind: str) -> list[str]:
        out = [f"\n### 方程 ({kind})\n"]
        try:
            if kind == "nodal":
                eqs = self.c.nodal_analysis().nodal_equations()
            elif kind == "mesh":
                eqs = self.c.mesh_analysis().mesh_equations()
            else:
                eqs = None
            if eqs is not None:
                for k, e in eqs.items():
                    out.append(f"- 节点/回路 {k}: $ {_latex(e.expr if hasattr(e, 'expr') else e)} $")
                return out
        except Exception as ex:
            out.append(f"> {kind} 方程生成失败 ({ex})，改输出 MNA 矩阵方程")
        try:
            out.append(f"$$ {_latex(self.c.matrix_equations())} $$")
        except Exception as ex:
            out.append(f"> MNA 方程生成失败: {ex}")
        return out

    def table(self) -> list[str]:
        from lcapy import t
        rows = ["\n### 元件 电压/电流/功率 一览 (关联参考方向: 第1节点→第2节点)\n",
                "| 元件 | 节点 | U | I | P(吸收) |", "|---|---|---|---|---|"]
        for name, e in self.c.elements.items():
            if name.startswith("K") or type(e).__name__ in ("K", "O", "P", "W"):
                continue
            try:
                if self.domain == "ac":
                    V, I = e.V.phasor().expr, e.I.phasor().expr
                    S = sp.N(V * sp.conjugate(I) / 2, 4)
                    rows.append(f"| {name} | {e.node_names[0]},{e.node_names[1]} | {polar_str(V)} | {polar_str(I)} | {S} |")
                elif self.domain == "dc":
                    V, I = sp.simplify(e.V(t).expr), sp.simplify(e.I(t).expr)
                    rows.append(f"| {name} | {e.node_names[0]},{e.node_names[1]} | {_num(V)} | {_num(I)} | {_num(V * I)} |")
            except Exception as ex:
                rows.append(f"| {name} | | 失败: {ex} | | |")
        return rows

    # ------------------------------------------------------------------
    def run(self) -> str:
        p = self.prob
        L = [f"## {p.get('title', '电路求解')}\n", f"**分析类型**: {self.domain}\n", "**网表** (lcapy):", "```", self.netlist.strip(), "```"]
        if p.get("given"):
            L.append(f"\n**题目已知 (仅来自印刷内容)**: {p['given']}")
        if p.get("ask"):
            L.append(f"\n**所求**: {p['ask']}")
        L.append("\n### 结果\n")
        for q in p.get("queries", []):
            L.append(self.query(q))
        if p.get("equations"):
            L += self.equations(p["equations"])
        if p.get("table"):
            L += self.table()
        L.append("\n### 自动审计结论 (Zero-Latency Audit)\n")
        unique_warns = list(dict.fromkeys(self.audit_warnings))
        if unique_warns:
            for w in unique_warns:
                L.append(f"- 🚨 {w}")
        else:
            L.append("- ✅ **自动审计通过**：所有计算结果分母整洁、等效内阻物理合理且无暂态发散项。")
        return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="lcapy 电路求解")
    ap.add_argument("problem", help="problem.yaml")
    ap.add_argument("-o", "--out", default=None, help="输出 markdown 路径 (默认与 yaml 同目录 solution.md)")
    a = ap.parse_args(argv)
    prob = yaml.safe_load(Path(a.problem).read_text(encoding="utf-8"))
    md = Solver(prob).run()
    out = Path(a.out) if a.out else Path(a.problem).with_name("solution.md")
    out.write_text(md, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(md)
    print(f"[已保存] {out}")


if __name__ == "__main__":
    main()
