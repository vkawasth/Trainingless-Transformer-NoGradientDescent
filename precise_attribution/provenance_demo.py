"""provenance_demo.py -- synthetic four-stage provenance experiment.

Stage 1  corpus C1           : M1 outputs mix the C1 sources A1, A2.
Stage 2  add corpus C2       : M2 reweights A1, A2 and adds the C2 source N; outcome y6 appears.
Stage 3  local weight edit   : declared scope {x1}; the edit also leaks into x2.
Stage 4  RLHF                : exponential tilt with weight ratio K = 4 (reward range beta*ln 4);
                               a "drifted" variant departs from the tilt class at x2.
Then: order and loop defects, the composition bound, and a coarse view.

All distributions are exact rationals; LP intervals are floats; certificates are verified exactly.
"""
from fractions import Fraction as F
from provenance import *

OUT = ["y1", "y2", "y3", "y4", "y5", "y6"]
PROBES = ["x1", "x2", "x3"]
def d(*v): return [F(a, 10) for a in v]
A1 = {"x1": d(4, 3, 2, 1, 0, 0), "x2": d(1, 1, 4, 4, 0, 0), "x3": d(5, 5, 0, 0, 0, 0)}
A2 = {"x1": d(0, 1, 2, 3, 4, 0), "x2": d(3, 3, 1, 1, 2, 0), "x3": d(0, 0, 5, 5, 0, 0)}
N  = {"x1": d(0, 0, 0, 2, 3, 5), "x2": d(0, 0, 0, 0, 5, 5), "x3": d(0, 0, 0, 0, 5, 5)}
w1 = {x: (F(3, 5), F(2, 5), F(0)) for x in PROBES}
w2 = {"x1": (F(9, 20), F(1, 4), F(3, 10)), "x2": (F(1, 2), F(3, 10), F(1, 5)), "x3": (F(3, 5), F(2, 5), F(0))}
M1 = {x: mix(w1[x], [A1[x], A2[x], N[x]]) for x in PROBES}
M2 = {x: mix(w2[x], [A1[x], A2[x], N[x]]) for x in PROBES}
EPS, TAU = F(1, 100), F(1, 100)
edit = {"x1": mixing(F(1, 5), point_mass(6, 2)),           # in scope: move 20% to y3
        "x2": mixing(F(1, 20), point_mass(6, 0)),          # leak: move 5% to y1
        "x3": (lambda p: p)}
M3 = {x: edit[x](M2[x]) for x in PROBES}
K = F(4); W = [F(1), F(2), F(4), F(1), F(3), F(1)]          # reward weights in [1, K]
M4 = {x: tilt(M3[x], W) for x in PROBES}
drift = [F(1), F(1), F(1), F(1), F(1), F(10)]
M4d = dict(M4); M4d["x2"] = tilt(M4["x2"], drift)            # extra boost of y6 at x2

def fmt(v): return "[" + ", ".join(f"{float(a):.4f}" for a in v) + "]"
def iv(t): return f"[{max(t[0], 0) + 0:.4f}, {max(t[1], 0) + 0:.4f}]"
report = {}

print("=" * 78); print("STAGE 2: add corpus C2 (declared sources A1, A2 from C1; N from C2)"); print("=" * 78)
for x in PROBES:
    srcs = [("exact", A1[x]), ("exact", A2[x]), ("exact", N[x])]
    rep = attribution_report(M2[x], srcs, 0, outcomes_of_interest=[4])
    print(f"{x}: weights A1 {iv(rep['lambda'][0])}  A2 {iv(rep['lambda'][1])}  N {iv(rep['lambda'][2])}"
          f"   max width {max(rep['widths']):.0e} (LP tolerance)")
    new = [OUT[b] for b in range(6) if M1[x][b] <= EPS < M2[x][b]]
    print(f"     outcomes crossing threshold eps = {EPS}: {new or 'none'};  TV(M1, M2) = {float(tv(M1[x], M2[x])):.4f}")
    if N[x][5] > 0 and A1[x][5] == 0 and A2[x][5] == 0:
        print(f"     exclusive region y6: lambda_N = p(y6)/N(y6) = {M2[x][5] / N[x][5]}  (exact)")
    if 4 in rep["pi"]:
        print(f"     P(source | y5): A1 {iv(rep['pi'][4][0])} A2 {iv(rep['pi'][4][1])} N {iv(rep['pi'][4][2])}")
    report[f"stage2_{x}"] = rep
x = "x1"
print("\nPrecision ladder at x1 (widths of A1, A2, N):")
ladder = [("exact sources, budget 0", [("exact", A1[x]), ("exact", A2[x]), ("exact", N[x])], 0),
          ("supports only, budget 0", [("support", {b for b in range(6) if s[x][b] > 0}) for s in (A1, A2, N)], 0),
          ("supports only, budget 0.1", [("support", {b for b in range(6) if s[x][b] > 0}) for s in (A1, A2, N)], 0.1)]
for name, srcs, bud in ladder:
    r = attribution_report(M2[x], srcs, bud)
    print(f"  {name:28s} A1 {iv(r['lambda'][0])} A2 {iv(r['lambda'][1])} N {iv(r['lambda'][2])}  mu {iv(r['mu'])}")
    report[f"ladder_{name}"] = r

print("\n" + "=" * 78); print("STAGE 3: weight edit, declared scope {x1}, locality tolerance tau = 1/100"); print("=" * 78)
for x in PROBES:
    t = tv(M3[x], M2[x])
    status = "in scope" if x == "x1" else ("PASS" if t <= TAU else "FAIL (leak)")
    mm, lb, y = mu_min_sources(M3[x], [A1[x], A2[x], N[x]])
    print(f"{x}: TV(M2, M3) = {float(t):.4f}  locality: {status};  unexplained by C1+C2 sources: "
          f"mu_min = {mm:.4f} (certified >= {float(lb):.4f})")
    if lb > 0:
        top = sorted(range(6), key=lambda b: -y[b])[:2]
        print(f"     certificate y = {fmt(y)}:  min_i <y, A_i> = {float(min(sum(a*b for a, b in zip(y, S[x])) for S in (A1, A2, N)))}"
              f", <y, p> = {float(sum(a*b for a, b in zip(y, M3[x]))):.4f}")
    report[f"stage3_{x}"] = (float(t), mm, float(lb))

print("\n" + "=" * 78); print("STAGE 4: RLHF as exponential tilt, claimed weight ratio K = 4"); print("=" * 78)
for label, M in (("tilt", M4), ("drifted", M4d)):
    for x in PROBES:
        ks = implied_ratio(M3[x], M[x])
        ok = ks is not None and ks <= K
        line = f"{label:8s} {x}: implied ratio K* = {float(ks):.3f}  -> {'reachable' if ok else 'NOT reachable'}"
        if not ok:
            mm, s, lb, (y, nu) = cone_residual(M[x], tilt_cone_rows(M3[x], K))
            line += f";  residual mu_min = {mm:.4f} (certified >= {float(lb):.4f})"
            report[f"stage4_{label}_{x}_res"] = (mm, float(lb))
        print(line); report[f"stage4_{label}_{x}"] = float(ks)

print("\n" + "=" * 78); print("PATHS at x1: E = edit, R = tilt"); print("=" * 78)
E, Rt = edit["x1"], tilt_map(W)
Rinv = tilt_map([1 / v for v in W])
p = M2["x1"]
od = tv(Rt(E(p)), E(Rt(p)))
loop = tv(Rinv(Rt(p)), p)
print(f"order defect TV(R(E(p)), E(R(p))) = {float(od):.4f}  (exact {od})")
print(f"loop defect of R then R^-1        = {loop}")
E2 = mixing(F(1, 4), point_mass(6, 2))
lhs = tv(Rt(E(p)), Rt(E2(p))); dEE = tv(E(p), E2(p))
print(f"composition: TV(R E p, R E' p) = {float(lhs):.4f} <= Lip(R) * TV(E p, E' p) = "
      f"{float(lipschitz_bound_tilt(K))} * {float(dEE):.4f} = {float(lipschitz_bound_tilt(K) * dEE):.4f};"
      f"  observed amplification {float(lhs / dEE):.3f}")
groups = [[0, 1], [2, 3], [4, 5]]
odc = tv(coarsen(Rt(E(p)), groups), coarsen(E(Rt(p)), groups))
print(f"coarse view {{y1,y2}},{{y3,y4}},{{y5,y6}}: order defect {float(odc):.4f} <= {float(od):.4f}, still > 0")
report["paths"] = (float(od), float(loop), float(lhs), float(dEE), float(odc))
