"""Checks for section 12.8 of the companion note (Stage IV): a path through the column-binding example (module pce/pentpath.py).
  PP1  the path f = (2/5, 3/5), g = (7/10 - gamma, 3/10, gamma) stays in one Kahler chamber: the facets are 11, 12, 13, 22, 23 at 24 rational points of
       [3/20, 11/40], and no blow-up wall f1 = g(J) meets gamma in (1/10, 3/10) (exact)
  PP2  the leximin recursion exactly at 9 points: below 1/5 the levels are gamma/2 (column 3), 3/20 (column 2), 1/4 - gamma/2 (cell 11); between 1/5
       and 4/15 they are gamma/2, (4/5 - gamma)/4 (row 1 residual), then cell 22; above 4/15 row 1 binds at 2/15; at gamma = 1/5 cells 11, 12, 22 tie
       (a level-2 wall) and at 4/15 cells 11, 12, 13, 23 tie (the level-1 wall)
  PP3  the reduction (sympy): y1 W_1 - y2 W_2 = y1 - y2 + B/y2, so y1 = (s^2 - B)/s, y2 = s, and y1 W_1 = 0 becomes (s^2 - B)^2 (1 + C s) = A s; at
       gamma = 1/5 (C = A/B^2) with tau = T^(1/10) this is the quintic q(s) of section 12.7 up to the factor tau^-2
  PP4  the Newton polygon of the quintic at 10 rational gamma: fibres and multiplicities (exact) are 4 x (1/4 - g/2, 3/20) + 1 x (g - 1/10, 2/5 - g)
       below 1/5, 4 x ((4/5 - g)/4)(1, 1) + 1 x (1/10, 1/5) between, 3 x (2/15, 2/15) + 1 x (2/5 - g)(1, 1) + 1 x (1/10, 1/5) above 4/15; and the
       leximin fibre is the multiple one in every regime
  PP5  the critical points at L = 200, 400: valuations (slopes) match PP4 within 2e-3; sign patterns: 5 real below 1/5 (++, +-, +-, -+, --);
       3 real (++, --, -+) and 2 complex between and above; leading phases of the complex pair +-pi/2 between, +-2pi/3 above (within 0.03 pi)
  PP6  collisions: the discriminant of the quintic (in A, B, C) has exactly one zero on gamma in [0.12, 0.275] for L = 50, 100, 200, 400, at
       gamma_c(L) -> 1/5 from below; no zero in (0.21, 0.275), so the level-1 wall is crossed without collision; at the zero the two (+,-) points meet
  PP7  critical values: on a grid of the path (L = 50, 100), the least relative separation |W_i - W_j| / max|W| of distinct points is attained at the
       collision; away from a window of width 0.01 around gamma_c it stays positive (above 1e-7; the closest pairs are objects over one fibre whose
       values agree to leading order), and the order of the real values is constant on each side
  PP8  the Clifford shadow along the path (slopes L = 400 -> 800): the positive object has scales 0 and h_2 - h_1 = 3/20 - gamma/2 below 1/5 and
       1/5 - 3 gamma/4 between (within 2e-3); above 4/15 both scales are 0 (rank 2); the four leximin-fibre objects share the positive object's scale
       below 4/15; R has rank 2 throughout
  PP9  Stokes alignments, separately (phases 0.3, 1.0, pi/2; L = 25, 50, 100): every alignment lies within 1.5/L of one of the two walls, and at
       L = 100 within 1/L; alignments occur near both walls
"""
import os, sys, itertools
from fractions import Fraction as Fr
import numpy as np
import mpmath as mp
import sympy as sp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import pentpath as PP, couplingfibre as CF, colbind as CB
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)

# PP1
gs = [PP.LO + (PP.HI - PP.LO) * Fr(k, 23) for k in range(24)]; ok = True
for g in gs:
    f, gg = PP.margins(g); ok &= CF.facet_cells([float(x) for x in f], [float(x) for x in gg]) == [0, 1, 2, 4, 5]
gam = sp.Symbol("gamma"); walls = []
for J in [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]:
    gJ = sum([sp.Rational(7, 10) - gam, sp.Rational(3, 10), gam][j] for j in J); walls += [r for r in sp.solve(sp.Eq(gJ, sp.Rational(2, 5)), gam) if 0.1 < r < 0.3]
check("PP1 one Kahler chamber along the path: facets 11, 12, 13, 22, 23; no blow-up wall for gamma in (1/10, 3/10)", ok and not walls)

# PP2
ok = True; info = []
for g in [Fr(3, 20), Fr(1, 6), Fr(19, 100), Fr(1, 5), Fr(9, 40), Fr(1, 4), Fr(4, 15), Fr(27, 100), Fr(11, 40)]:
    lx = PP.leximin(g); lv, st = lx["levels"], lx["sets"]
    if g < Fr(1, 5): ok &= lv == [g / 2, Fr(3, 20), Fr(1, 4) - g / 2] and st == [("13", "23"), ("12", "22"), ("11",)]
    elif g == Fr(1, 5): ok &= lv == [Fr(1, 10), Fr(3, 20)] and st[1] == ("11", "12", "22")
    elif g < Fr(4, 15): ok &= lv[:2] == [g / 2, (Fr(4, 5) - g) / 4] and st[:2] == [("13", "23"), ("11", "12")]
    elif g == Fr(4, 15): ok &= lv[0] == Fr(2, 15) and st[0] == ("11", "12", "13", "23")
    else: ok &= lv[0] == Fr(2, 15) and st[0] == ("11", "12", "13")
check("PP2 leximin exactly: walls at 1/5 (level 2: 11, 12, 22 tie) and 4/15 (level 1: 11, 12, 13, 23 tie); levels as stated in each regime", ok)

# PP3
d = PP.quintic(); A, B, C, s = d["A"], d["B"], d["C"], d["s"]
ok = sp.simplify(d["diff"] - (sp.Symbol("y1") - sp.Symbol("y2") + B / sp.Symbol("y2"))) == 0
ok &= sp.expand(sp.numer(sp.together(d["e1"])) - d["q"]) == 0 or sp.expand(sp.numer(sp.together(d["e1"])) + d["q"]) == 0
t = sp.Symbol("tau"); q15 = sp.expand(d["q"].subs({A: t ** 4, B: t ** 3, C: t ** -2}) * t ** 2); ok &= sp.expand(q15 - CB.symbolic()["q"]) == 0
check("PP3 exact reduction (s^2 - B)^2 (1 + Cs) = As; at gamma = 1/5 it is the quintic of section 12.7", ok)

# PP4
ok = True
for g in [Fr(3, 20), Fr(1, 6), Fr(37, 200), Fr(39, 200), Fr(41, 200), Fr(9, 40), Fr(1, 4), Fr(53, 200), Fr(27, 100), Fr(11, 40)]:
    _, _, fib = PP.newton(g); fd = dict(fib); lx = tuple(PP.leximin(g)["point"])
    if g < Fr(1, 5): exp = {(Fr(1, 4) - g / 2, Fr(3, 20)): 4, (g - Fr(1, 10), Fr(2, 5) - g): 1}
    elif g < Fr(4, 15): exp = {((Fr(4, 5) - g) / 4, (Fr(4, 5) - g) / 4): 4, (Fr(1, 10), Fr(1, 5)): 1}
    else: exp = {(Fr(2, 15), Fr(2, 15)): 3, (Fr(2, 5) - g, Fr(2, 5) - g): 1, (Fr(1, 10), Fr(1, 5)): 1}
    ok &= fd == exp and max(fd, key=fd.get) == lx
check("PP4 Newton polygon (exact): 4 + 1 below 1/5, 4 + 1 between, 3 + 1 + 1 above 4/15; the leximin fibre carries the multiple root", ok)

# PP5
ok = True; info = {}
for g in [0.15, 0.18, 0.22, 0.25, 0.27, 0.275]:
    r2 = PP.roots(g, 200.0); r4 = PP.roots(g, 400.0)
    _, _, fib = PP.newton(Fr(g).limit_denominator(1000)); want = sorted(itertools.chain.from_iterable([[tuple(float(x) for x in u)] * m for u, m in fib]))
    got = []
    for o in r4:
        o2 = min(r2, key=lambda q: sum(abs(mp.arg(q["y"][i]) - mp.arg(o["y"][i])) for i in range(2)) + 50 * sum(abs(mp.log(abs(q["y"][i])) / 200 - mp.log(abs(o["y"][i])) / 400) for i in range(2)))
        got.append(tuple(float(-(mp.log(abs(o["y"][i])) - mp.log(abs(o2["y"][i]))) / 200) for i in range(2)))
    ok &= all(max(abs(a - b) for a, b in zip(x, y)) < 2e-3 for x, y in zip(sorted(got), want))
    kinds = sorted(PP.kind(o) for o in r4); info[g] = kinds
    if g < 0.2: ok &= kinds == ["+-", "+-", "++", "-+", "--"] or kinds == sorted(["++", "+-", "+-", "-+", "--"])
    else:
        ok &= kinds == sorted(["++", "--", "-+", "cx", "cx"])
        ph = sorted(abs(float(mp.arg(o["y"][1]) / mp.pi)) for o in r4 if PP.kind(o) == "cx"); want_ph = 0.5 if g < 4 / 15 else 2 / 3
        ok &= all(abs(p - want_ph) < 0.03 for p in ph)
check("PP5 valuations at L = 200 -> 400 match PP4; 5 real below 1/5, 3 real above; complex phases +-pi/2 between, +-2pi/3 above", ok, f"{info}")

# PP6
zs = {}; ok = True
for L in (50.0, 100.0, 200.0, 400.0): z = PP.disc_zeros(L); zs[L] = [float(x) for x in z]; ok &= len(z) == 1 and 0.18 < z[0] < 0.2
ok &= zs[50.0][0] < zs[100.0][0] < zs[200.0][0] <= zs[400.0][0] < 0.2
L = 100.0; zc = zs[L][0]; before = sorted(PP.kind(o) for o in PP.roots(zc - 2e-4, L)); after = sorted(PP.kind(o) for o in PP.roots(zc + 2e-4, L))
ok &= before.count("+-") == 2 and after.count("+-") == 0 and after.count("cx") == 2 and after.count("-+") == 1
check("PP6 one collision on the path, at gamma_c(L) -> 1/5 (the level-2 wall); none near the level-1 wall; the two (+,-) points meet", ok,
      f"gamma_c {zs}")

# PP7
ok = True; info = {}
for L in (50.0, 100.0):
    zc = zs[L][0]; grid = np.linspace(0.13, 0.275, 300); seps = []; orders = {}
    for g in grid:
        r = PP.roots(g, L); Ws = [o["W"] for o in r]; m = max(abs(w) for w in Ws)
        sep = min(abs(a - b) for a, b in itertools.combinations(Ws, 2)) / m; seps.append((float(sep), g))
        re = sorted((float(mp.re(o["W"])), PP.kind(o)) for o in r if PP.kind(o) != "cx"); side = "lo" if g < zc else ("mid" if g < 4 / 15 else "hi")
        orders.setdefault(side, set()).add(tuple(k for _, k in re))
    far = [x for x, g in seps if abs(g - zc) > 0.005]; info[L] = (round(min(far), 6), {k: len(v) for k, v in orders.items()})
    ok &= min(far) > 1e-7 and all(len(v) == 1 for v in orders.values())
check("PP7 no coincidence of critical values away from the collision; the order of the real values is constant on each side", ok, f"{info}")

# PP8
ok = True; info = []
for g in [0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.275]:
    r4 = PP.roots(g, 400.0); r8 = PP.roots(g, 800.0)
    def scales(o, rr, L1, L2):
        o2 = min(rr, key=lambda q: sum(abs(mp.arg(q["y"][i]) - mp.arg(o["y"][i])) for i in range(2)) + 50 * sum(abs(mp.log(abs(q["y"][i])) / L2 - mp.log(abs(o["y"][i])) / L1) for i in range(2)))
        e1 = sorted(mp.eig(o["Qn"])[0], key=abs); e2 = sorted(mp.eig(o2["Qn"])[0], key=abs)
        return sorted(float(-(mp.log(abs(b)) - mp.log(abs(a))) / (L2 - L1)) for a, b in zip(e1, e2))
    P = next(o for o in r4 if PP.kind(o) == "++"); sp_ = scales(P, r8, 400.0, 800.0)
    pred = 3 / 20 - g / 2 if g < 0.2 else (1 / 5 - 3 * g / 4 if g < 4 / 15 else 0.0)
    ok &= abs(sp_[0]) < 2e-3 and abs(sp_[1] - pred) < 2e-3; info.append((g, round(sp_[1], 4), round(pred, 4)))
    lexu = tuple(float(x) for x in PP.leximin(Fr(g).limit_denominator(1000))["point"])
    for o in r4:
        u = tuple(float(-mp.log(abs(v)) / 400) for v in o["y"]); sc = scales(o, r8, 400.0, 800.0)
        if max(abs(a - b) for a, b in zip(u, lexu)) < 0.01 and g < 4 / 15: ok &= abs(sc[1] - pred) < 3e-3
        if abs(u[0] - 0.1) < 0.01 and abs(u[1] - 0.2) < 0.01: ok &= max(abs(x) for x in sc) < 3e-3
check("PP8 the Clifford shadow along the path: scale h2 - h1 bends at 1/5 and closes at 4/15 (rank 1 -> 2); leximin objects share it; R has rank 2", ok, f"{info}")

# PP9
ok = True; info = {}
for L in (25.0, 50.0, 100.0):
    gams = np.linspace(0.12, 0.275, 1551); P_, W_ = PP.track(L, gams); near = {"1/5": 0, "4/15": 0}; worst = 0
    for th in (0.3, 1.0, np.pi / 2):
        for g, i, j, conj in PP.stokes(L, gams, th, P_, W_):
            d1, d2 = abs(g - 0.2), abs(g - 4 / 15); worst = max(worst, L * min(d1, d2)); near["1/5" if d1 < d2 else "4/15"] += 1
    info[L] = (round(worst, 2), near); ok &= worst < 1.5 and (L < 100 or worst < 1.0)
ok &= all(v > 0 for v in info[100.0][1].values())
check("PP9 Stokes alignments (phases 0.3, 1.0, pi/2) lie within 1.5/L of a wall, within 1/L at L = 100, near both walls", ok, f"{info}")
print(f"{sum(results)}/{len(results)} passed")
