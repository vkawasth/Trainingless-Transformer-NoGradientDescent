"""Checks for Section "Geometric realization: the toric picture" of "Probability, Context and Differentiation: The Layers"
(module pce/toricreal.py).
  TR1  Birch's theorem: for 40 random supports S with positive coefficients (two and three parameters), every point of the relative
       interior of Q = conv(S) is the moment of exactly one monomial law (four starts agree to 1e-8); on a homogeneous support the
       monomial law is constant exactly along the directions orthogonal to S - S
  TR2  arc limits are face laws: along 300 arcs x = c t^w (1 + higher terms), w > 0, the monomial law tends exactly to
       normalise(c_alpha c^alpha) on the face F_w minimising <w, .>; F_w is a lower face; its moment lies in the relative interior of F_w
  TR3  the limit set is the lower boundary: for every face of Q exposed by weights of either sign, the linear programme's verdict is
       certified: a lower face comes with a positive weight exposing exactly it, and every face exposed by a positive weight is lower;
       upper and side faces never support a limit
  TR4  cells as aggregated monomials: for 200 random positive-coefficient cells, the cell limit is the aggregation of the monomial limit
       and equals normalise(in_w f(c)) (the order fan theorem); no face is degenerate
  TR5  the blueprint: Q = conv{(3,0),(2,1),(1,1),(0,3)}; the lower faces are the three vertices (3,0), (1,1), (0,3) and the two edges
       between them, with normal rays of slopes 2 and 1/2 (the order fan of P(A)); (2,1), the cell 7/4 s^2 t, lies on the upper edge and
       has weight 0 in every limit; the signed cell (x - y)^2 z gives a monomial 'law' with a negative entry, where the realization stops
"""
import os, sys, random, itertools
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import toricreal as TR
from pce import degen_d as DD
from pce import conewise as CW
random.seed(20261102); rng = np.random.default_rng(20261102); results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
def rand_support(d, k):
    while True:
        S = sorted({tuple(random.randint(0, 4) for _ in range(d)) for _ in range(k)})
        if len(S) >= d + 1 and np.linalg.matrix_rank(np.array(S, float)[1:] - np.array(S[0], float)) == d: return S

# TR1
ok = True; n = 0
for _ in range(40):
    d = random.choice([2, 3]); S = rand_support(d, random.randint(d + 2, 7)); c = [random.uniform(0.2, 3) for _ in S]
    for _ in range(3):
        lam = rng.dirichlet(np.ones(len(S))); y = lam @ np.array(S, float); sols = TR.inverse_moment(S, c, y, rng=rng)
        ok &= max(np.max(np.abs(s - sols[0])) for s in sols) < 1e-8 and np.max(np.abs(sols[0] @ np.array(S, float) - y)) < 1e-8; n += 1
Sh = [(3, 0), (2, 1), (1, 2), (0, 3)]; ch = [1.0, 2.0, 0.5, 1.0]; x = np.array([0.3, 0.7])
ok &= np.allclose(TR.monomial_law(Sh, ch, x), TR.monomial_law(Sh, ch, x * np.exp(np.array([1.0, 1.0]) * 0.8)))      # (1,1) is orthogonal to S - S
ok &= not np.allclose(TR.monomial_law(Sh, ch, x), TR.monomial_law(Sh, ch, x * np.exp(np.array([1.0, -1.0]) * 0.3)))
check("TR1 Birch: each relative-interior point is the moment of exactly one monomial law; homogeneous supports are invariant along (S-S)-perp",
      ok, f"{n} targets")

# TR2, TR3
ok2 = True; ok3 = True; n = 0; supports = set()
for _ in range(300):
    d = random.choice([2, 3]); S = rand_support(d, random.randint(d + 1, 6)); c = [Fr(random.randint(1, 5), random.randint(1, 3)) for _ in S]
    w = [random.randint(1, 4) for _ in range(d)]; cv = [Fr(random.randint(1, 5), random.randint(1, 3)) for _ in range(d)]
    arc = [{w[i]: cv[i]} for i in range(d)]
    for i in range(d): arc[i][w[i] + random.randint(1, 4)] = Fr(random.randint(-3, 3))
    cells = {i: {S[i]: c[i]} for i in range(len(S))}; m, lim = DD.arc_limit(cells, arc)
    v = [sum(wi * ai for wi, ai in zip(w, a)) for a in S]; F = tuple(i for i in range(len(S)) if v[i] == min(v))
    fl = TR.face_law(S, c, F, cv); ok2 &= all(lim[i] == fl.get(i, 0) for i in range(len(S))) and TR.lower(S, F)
    mu = sum(float(lim[i]) * np.array(S[i], float) for i in range(len(S))); ok2 &= all(float(lim[i]) > 0 for i in F)
    supports.add((tuple(S), F))
    allW = [ww for ww in itertools.product(range(-3, 4), repeat=d) if any(ww)]
    pos = TR.faces_by_weight(S, [x_ for x_ in allW if min(x_) > 0])
    for G, ww in TR.faces_by_weight(S, allW).items():
        isl, wit = TR.lower(S, G, witness=True)
        if isl:                                                               # the LP's witness is positive and exposes exactly G
            vv = [float(np.dot(wit, a)) for a in S]; mm = min(vv)
            ok3 &= min(wit) > 0 and tuple(i for i in range(len(S)) if abs(vv[i] - mm) < 1e-7) == G
        ok3 &= (G not in pos) or isl                                          # a face exposed by a positive weight is lower
    n += 1
check("TR2 arc limits are face laws on the face F_w, which is lower, with full support on F_w (300 arcs, exact)", ok2)
check("TR3 the limit set is the lower boundary: faces exposed by positive weights = lower faces (LP), for every face of Q", ok3, f"{n} supports")

# TR4
ok = True
for _ in range(200):
    d = random.choice([2, 3]); S = rand_support(d, random.randint(d + 2, 7)); c = [Fr(random.randint(1, 5), random.randint(1, 3)) for _ in S]
    k = random.randint(2, 3); grp = [random.randrange(k) for _ in S]
    cells = {g: {S[i]: c[i] for i in range(len(S)) if grp[i] == g} for g in range(k)}; cells = {g: f for g, f in cells.items() if f}
    w = [random.randint(1, 4) for _ in range(d)]; cv = [Fr(random.randint(1, 5), random.randint(1, 3)) for _ in range(d)]
    arc = [{w[i]: cv[i]} for i in range(d)]
    for i in range(d): arc[i][w[i] + random.randint(1, 4)] = Fr(random.randint(-3, 3))
    m, lim = DD.arc_limit(cells, arc); mono = DD.arc_limit({i: {S[i]: c[i]} for i in range(len(S))}, arc)[1]
    agg = {g: sum(mono[i] for i in range(len(S)) if grp[i] == g) for g in cells}
    P = {}
    for f in cells.values():
        for a_, v_ in f.items(): P[a_] = P.get(a_, 0) + v_
    vals = {g: CW.init_val(f, w, cv) if CW.trop(f, w) == CW.trop(P, w) else Fr(0) for g, f in cells.items()}; tot = sum(vals.values())
    ok &= all(lim[g] == agg[g] == vals[g] / tot for g in cells)
    ok &= all(DD.degeneracy_gap(P, F) == 1.0 for F in DD.faces(P, [tuple(random.randint(1, 4) for _ in range(d)) for _ in range(20)]))
check("TR4 cells are aggregated monomials: cell limit = aggregated face law = normalise(in_w f(c)); no face degenerates (200 families)", ok)

# TR5
S = [(3, 0), (2, 1), (1, 1), (0, 3)]; c = [Fr(1), Fr(7, 4), Fr(1), Fr(1)]
W = [ww for ww in itertools.product(range(-6, 7), repeat=2) if any(ww)]
lowF = sorted({F for F in TR.faces_by_weight(S, W) if TR.lower(S, F)})
expect = sorted([(0,), (2,), (3,), (0, 2), (2, 3)])
upper = [F for F in TR.faces_by_weight(S, W) if 1 in F]
rays = sorted({Fr(w[1], w[0]) for w in [(a, b) for a in range(1, 13) for b in range(1, 13)] if len(TR.faces_by_weight(S, [(a_, b_) for a_, b_ in [w]]).popitem()[0]) == 2})
zero = True
for _ in range(100):
    w = [random.randint(1, 6), random.randint(1, 6)]; cv = [Fr(random.randint(1, 5)), Fr(random.randint(1, 5))]
    lim = DD.arc_limit({i: {S[i]: c[i]} for i in range(4)}, [{w[0]: cv[0]}, {w[1]: cv[1]}])[1]; zero &= lim[1] == 0
signed = {(2, 0, 1): 1.0, (1, 1, 1): -2.0, (0, 2, 1): 1.0, (2, 2, 0): 1.0}
xs = np.array([0.5, 0.5, 0.5]); mon = np.array([cf * np.prod(xs ** np.array(a)) for a, cf in signed.items()]); qs = mon / mon.sum()
check("TR5 the blueprint: lower faces = three vertices and two edges, normal rays of slopes 2 and 1/2; the cell 7/4 s^2 t sits on the upper "
      "edge and has weight 0 in every limit; a signed cell gives a negative 'law', where the realization stops",
      lowF == expect and all(not TR.lower(S, F) for F in upper) and rays == [Fr(1, 2), Fr(2)] and zero and qs.min() < 0,
      f"lower faces {lowF}; faces containing (2,1): {upper}; rays {[str(r) for r in rays]}; signed 'law' {np.round(qs, 3).tolist()}")
print(f"{sum(results)}/{len(results)} passed")
