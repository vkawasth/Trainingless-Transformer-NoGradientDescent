"""Checks for the several-parameter theorem of Step 4f of "Probability, Context and Differentiation: The Layers" (module pce/degen_d.py).
  DD1  per arc, any d: along 300 random arcs in three parameters (non-negative cells with signed coefficients) the limit read from leading
       terms equals exact evaluation at t = 10^-30, and truncating the arc above the order v of P(A) along it never changes the limit
  DD2  the degenerate example f1 = (x - y)^2 z, f2 = x^2 y^2: arcs x = t^4, y = t^4 + a t^e, z = c t^4 are explained by source 1 for e = 5,
       by source 2 for e = 7, and give share a^2 c / (1 + a^2 c) at e = 6 (exactly, 60 cases); the exact branch y = x gives source 2
  DD3  the degeneracy detector: the degenerate example has exactly one degenerate bounded face, in_F P = (x - y)^2 z; the signed but
       non-degenerate example f1 = (x^2 - xy + y^2) z has none
  DD4  construction in the non-degenerate case: along 300 arcs with random higher-order terms, the limit is normalise(in_w f(c)) from the
       leading terms alone; in the degenerate example arcs with the same leading terms reach different limits
  DD5  finiteness: the sampled limit set of the degenerate example is one compact interval [0, 1] (largest gap < 0.02, endpoints attained)
"""
import os, sys, random, itertools
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import degen_d as DD
random.seed(20261102); results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
sq = lambda p: mulpoly(p, p)
def mono(a): return {a: Fr(1)}
def mulpoly(p, q):
    out = {}
    for a, x in p.items():
        for b, y in q.items():
            k = tuple(i + j for i, j in zip(a, b)); out[k] = out.get(k, 0) + x * y
    return {k: v for k, v in out.items() if v != 0}
def total(cells):
    P = {}
    for f in cells.values():
        for k, v in f.items(): P[k] = P.get(k, 0) + v
    return {k: v for k, v in P.items() if v != 0}

# DD1
ok = True; n = 0
for _ in range(300):
    lin = {(1, 0, 0): Fr(random.choice([1, 2])), (0, 1, 0): Fr(-random.choice([1, 2, 3])), (0, 0, 1): Fr(random.choice([-1, 0, 1]))}
    lin = {k: v for k, v in lin.items() if v != 0}
    cells = {1: mulpoly(sq(lin), mono((0, 0, random.randint(0, 2)))), 2: mono(tuple(random.randint(0, 3) for _ in range(3))),
             3: mono(tuple(random.randint(1, 3) for _ in range(3)))}
    w = [random.randint(1, 4) for _ in range(3)]
    arc = [{w[i]: Fr(random.randint(1, 4))} for i in range(3)]
    for i in range(3):
        for _k in range(2): arc[i][w[i] + random.randint(1, 6)] = Fr(random.randint(-3, 3))
    m, lim = DD.arc_limit(cells, arc)
    if lim is None: continue
    n += 1; ev = DD.evaluate(cells, arc, Fr(1, 10 ** 30))
    ok &= all(abs(float(ev[c] - lim[c])) < 1e-6 for c in cells)
    vP = DD.lead(DD.compose(total(cells), arc))[0]; m2, lim2 = DD.arc_limit(cells, DD.truncate(arc, vP))
    ok &= lim2 == lim
check("DD1 per arc in three parameters: leading terms = exact evaluation; truncation above v(P along the arc) never changes the limit", ok, f"{n} arcs")

# DD2
f1 = {(2, 0, 1): Fr(1), (1, 1, 1): Fr(-2), (0, 2, 1): Fr(1)}; f2 = {(2, 2, 0): Fr(1)}; cells = {1: f1, 2: f2}; ok = True
for _ in range(60):
    a, c = Fr(random.randint(1, 9), random.randint(1, 5)), Fr(random.randint(1, 9), random.randint(1, 5))
    arc = lambda e: [{4: Fr(1)}, {4: Fr(1), e: a}, {4: c}]
    ok &= DD.arc_limit(cells, arc(5))[1][1] == 1 and DD.arc_limit(cells, arc(7))[1][2] == 1
    ok &= DD.arc_limit(cells, arc(6))[1][1] == a * a * c / (1 + a * a * c)
ok &= DD.arc_limit(cells, [{4: Fr(1)}, {4: Fr(1)}, {4: Fr(2)}])[1][2] == 1
check("DD2 the degenerate example: source 1 for e = 5, source 2 for e = 7, share a^2c/(1+a^2c) at e = 6; the exact branch gives source 2", ok)

# DD3
W = [(a, b, c) for a in range(1, 6) for b in range(1, 6) for c in range(1, 6)]
P = total(cells); deg = [F for F in DD.faces(P, W) if DD.degeneracy_gap(P, F) < 1e-8]
g = {(2, 0, 1): Fr(1), (1, 1, 1): Fr(-1), (0, 2, 1): Fr(1)}; cells_nd = {1: g, 2: f2}; Pn = total(cells_nd)
deg_nd = [F for F in DD.faces(Pn, W) if DD.degeneracy_gap(Pn, F) < 1e-8]
check("DD3 detector: one degenerate face ((x-y)^2 z) in the degenerate example, none in the signed non-degenerate example",
      len(deg) == 1 and set(deg[0]) == set(f1) and not deg_nd, f"{len(DD.faces(P, W))} and {len(DD.faces(Pn, W))} bounded faces seen")

# DD4
ok = True
for _ in range(300):
    w = [random.randint(1, 3) for _ in range(3)]; c0 = [Fr(random.randint(1, 5), random.randint(1, 3)) for _ in range(3)]
    arc = [{w[i]: c0[i]} for i in range(3)]
    for i in range(3): arc[i][w[i] + random.randint(1, 5)] = Fr(random.randint(-4, 4))
    m, lim = DD.arc_limit(cells_nd, arc)
    lf = {k: DD.initial_form(f, w) if min(sum(wi * ai for wi, ai in zip(w, al)) for al in f) == min(sum(wi * ai for wi, ai in zip(w, al)) for al in Pn) else {} for k, f in cells_nd.items()}
    val = {k: sum(co * np.prod([c0[i] ** al[i] for i in range(3)]) for al, co in f.items()) for k, f in lf.items()}
    tot = sum(val.values()); ok &= tot > 0 and all(lim[k] == val[k] / tot for k in cells_nd)
same = [DD.arc_limit(cells, [{4: Fr(1)}, {4: Fr(1), 6: Fr(b)}, {4: Fr(1)}])[1][1] for b in (1, 2, 3)]
check("DD4 non-degenerate: the limit is normalise(in_w f(c)) from leading terms alone (300 arcs); degenerate: same leading terms, different limits",
      ok and len(set(same)) == 3, f"degenerate example, one leading direction: shares {[str(x) for x in same]}")

# DD5
grid_ = sorted({Fr(k, 64) for k in range(1, 400)} | {Fr(64, k) for k in range(1, 400)})
vals = sorted({float(DD.arc_limit(cells, [{4: Fr(1)}, {4: Fr(1), 6: Fr(1)}, {4: c}])[1][1]) for c in grid_}
              | {1.0, 0.0})
gap = max(b - a for a, b in zip(vals, vals[1:]))
check("DD5 finiteness: the sampled limit set of the degenerate example is one compact interval [0, 1]", gap < 0.02 and vals[0] == 0 and vals[-1] == 1,
      f"largest gap {gap:.4f}, {len(vals)} distinct limits")
print(f"{sum(results)}/{len(results)} passed")
