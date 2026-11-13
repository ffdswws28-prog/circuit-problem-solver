"""验证受控源 (CCVS/VCCS) 与模拟电子 (运放 / BJT 共射小信号模型) 求解与网表回绘."""
from pathlib import Path
import sys
import sympy as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from circuit_tool.render import draw_netgraph, parse_elements
from circuit_tool.solve import Solver

OUT = ROOT / "runs" / "test_dep_opamp"
OUT.mkdir(parents=True, exist_ok=True)


def run_case(name: str, prob: dict, checks: dict):
    solver = Solver(prob)
    md = solver.run()
    (OUT / f"{name}.md").write_text(md, encoding="utf-8")

    _, elems = parse_elements(solver.netlist)
    draw_netgraph(elems, OUT / f"{name}_netgraph.png", prob["title"])

    from lcapy import t
    c = solver.c
    for label, (actual_fn, expected, tol) in checks.items():
        val = actual_fn(c, t)
        num = complex(sp.N(val))
        err = abs(num - expected)
        assert err <= tol, f"[{name}] {label}: got {num}, expected {expected} (err={err})"
        print(f"  [PASS] {name} | {label}: {num:.6g} == {expected:.6g}")


def main():
    # 1. 含受控源 (CCVS + VCCS) 的直流电路:
    #    左边回路: Vs=10V, R1=2Ω, Vm=0V(测 I1 方向 N2->0), R2=3Ω -> I1 = 10/(2+3) = 2A?
    #    加上 VCCS G1 在右边对左边的影响或单独级联:
    #    设左回路: Vs(N1,0)=12V, R1(N1,N2)=2Ω, Vm(N2,N3)=0V, R2(N3,0)=4Ω -> I1 = 2A, V(N3) = 8V.
    #    右半边: CCVS H1(N4,0, Vm, 3) => U_H1 = 3 * I1 = 6V;
    #            G1(0, N5, N3, 0, 0.5) => 电流从 N5 经源流向 0? 注意 lcapy: G n_head n_tail nc+ nc- gm 电流从 n_tail 到 n_head,
    #            所以 G1(N5, 0, N3, 0, 0.5) 表示箭头指向 N5, 即向 N5 注入 0.5 * V(N3) = 4A;
    #            R3(N4, N5)=2Ω, R4(N5, 0)=2Ω:
    #            节点 N5 KCL: (V5 - 6)/2 + V5/2 = 4 => V5 = 7V, V4 = 6V.
    dep_prob = {
        "title": "受控源验证: CCVS (H1) + VCCS (G1)",
        "domain": "dc",
        "given": "Vs=12V, R1=2Ω, R2=4Ω, H1=3*I1 (CCVS), G1=0.5*U_R2 (VCCS), R3=2Ω, R4=2Ω",
        "ask": "求控制电流 I1, 节点电压 U_N3, U_N4, U_N5",
        "netlist": """
Vs N1 0 dc 12
R1 N1 N2 2
Vm N2 N3 0
R2 N3 0 4
H1 N4 0 Vm 3
R3 N4 N5 2
G1 N5 0 N3 0 0.5
R4 N5 0 2
""",
        "queries": ["I(Vm)", "V(N3)", "V(N4)", "V(N5)", "I(R3)"],
        "table": True,
    }
    run_case(
        "dependent_sources",
        dep_prob,
        {
            "I1 (Vm)": (lambda c, t: c.Vm.I(t).expr, 2.0, 1e-6),
            "V(N3)": (lambda c, t: c["N3"].V(t).expr, 8.0, 1e-6),
            "V(N4)": (lambda c, t: c["N4"].V(t).expr, 6.0, 1e-6),
            "V(N5)": (lambda c, t: c["N5"].V(t).expr, 7.0, 1e-6),
            "I(R3)": (lambda c, t: c.R3.I(t).expr, -0.5, 1e-6),
        },
    )

    # 2. 模拟电子 (一): 反相比例运算放大器 (高增益 VCVS E1 out 0 0 inv 1e8)
    #    Vin=1V, R1=2kΩ, Rf=10kΩ, RL=5kΩ -> Au = Vout/Vin = -Rf/R1 = -5
    opamp_prob = {
        "title": "模电验证 1: 反相比例运算放大器",
        "domain": "dc",
        "given": "Vin=1V, R1=2kΩ, Rf=10kΩ, RL=5kΩ, 理想运放 (开环增益 1e8)",
        "ask": "求输出电压 U_out 及输入电流 I_in",
        "netlist": """
Vin in 0 dc 1
R1 in inv 2000
Rf inv out 10000
E1 out 0 0 inv 1e8
RL out 0 5000
""",
        "queries": ["V(out)", "V(inv)", "I(R1)"],
        "table": True,
    }
    run_case(
        "opamp_inverting",
        opamp_prob,
        {
            "V(out)": (lambda c, t: c["out"].V(t).expr, -5.0, 1e-4),
            "V(inv)": (lambda c, t: c["inv"].V(t).expr, 0.0, 1e-5),
            "I(R1)": (lambda c, t: c.R1.I(t).expr, 5e-4, 1e-7),
        },
    )

    # 3. 模拟电子 (二): BJT 共射放大器小信号等效模型 (交流相量域)
    #    Rs=1kΩ, Rb=100kΩ, rbe=2kΩ, beta=100, Rc=4kΩ, RL=4kΩ
    #    测量 Ib: 从基极 b 经 Vm 到 rbe 再到地; CCCS F1 集电极 c 指向地(箭头向下: F1 0 c Vm 100)
    #    理论: Ri = Rb || rbe = 100k || 2k = 200/102 kΩ ≈ 1960.78 Ω
    #          RL' = Rc || RL = 2kΩ
    #          Au = Vout / Vb = -beta * RL' / rbe = -100 * 2k / 2k = -100
    #          Aus = Vout / Vs = Ri / (Rs + Ri) * (-100)
    ri_expected = 100000 * 2000 / (100000 + 2000)
    aus_expected = (ri_expected / (1000 + ri_expected)) * (-100.0)
    bjt_prob = {
        "title": "模电验证 2: BJT 共射放大器小信号交流等效模型",
        "domain": "ac",
        "given": "Us=10mV∠0° (ω=1000 rad/s), Rs=1kΩ, Rb=100kΩ, rbe=2kΩ, β=100, Rc=4kΩ, RL=4kΩ",
        "ask": "求基极电流 Ib、输出电压 Uo 及源电压放大倍数 Aus = Uo/Us",
        "netlist": """
Vs s 0 ac 0.01 0 1000
Rs s b 1000
Rb b 0 100000
Vm b b1 0
Rbe b1 0 2000
F1 c 0 Vm 100
Rc c 0 4000
RL c 0 4000
""",
        "queries": [
            "V(b)",
            "I(Vm)",
            "V(c)",
            {"expr": "c['c'].V.phasor().expr / c['s'].V.phasor().expr"},
            {"expr": "c['c'].V.phasor().expr / c['b'].V.phasor().expr"},
            "Z(b, 0)",
        ],
        "table": True,
    }
    run_case(
        "bjt_common_emitter",
        bjt_prob,
        {
            "Au (Uo/Ub)": (lambda c, t: c["c"].V.phasor().expr / c["b"].V.phasor().expr, -100.0, 1e-4),
            "Aus (Uo/Us)": (lambda c, t: c["c"].V.phasor().expr / c["s"].V.phasor().expr, aus_expected, 1e-4),
        },
    )

    print("All dependent source & analog electronics checks passed!")


if __name__ == "__main__":
    main()
