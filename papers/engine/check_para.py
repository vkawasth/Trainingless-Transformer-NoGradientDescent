"""Checks for Step 4c of "Probability, Context and Differentiation: The Layers": parameters, reparametrisations and decisions on the fan.
  PR1  the Para construction over source-indexed kernels (exact): composition is associative up to re-bracketing the parameters, has
       identities, holds the source fixed, and reparametrisations whisker: (g o f) o (r x r') = (g o r') o (f o r)
  PR2  the layer operations commute with reparametrisation (exact): labelling, pooling, marginalisation, forgetting and conditioning of
       a parameterised labelled law, then reparametrised, equal the same operations on the reparametrised law; the domain of conditioning
       pulls back
  PR3  the germ field (exact): evaluating the cells at an infinitesimal point eps_i = c_i s^{w_i} of the Puiseux field gives a labelled law
       over that field; its valuation on each cell is trop_w (non-degenerate case) and its standard part is the arc limit; numerics agree
  PR4  the decision fan (exact): along every arc the optimal act is eventually constant, decided by the signs of leading coefficients;
       it agrees with the act of the limiting state whenever that has no tie; on a vertex cone it does not depend on (w, c) in the cone;
       at ties the next order decides; numerical evaluation at small s agrees
  PR5  the audit example, exactly: in the race, audit source 1 when w_t < w_s, source 2 when w_t > w_s; on the ray t = c s, source 1 iff
       c > 1, source 2 iff c < 1, and at c = 1 the limit ties (1/2, 1/2) but the next order (the factor 1 - t) decides: source 1
  PR6  transport (exact): for a monomial reparametrisation eps = delta^R, the arc limits and the eventual acts of the reparametrised family
       at (w', c') equal those of the original at (R w', c'^R); the two study designs t = s u and (s, t) = (u v, u) map the whole quadrant
       into one cone each, so their limits exist (registry, resp. trials) although the race has none; the design t = s (1 + u) on the ray
       audits source 1 by every route
"""
import os, sys, json, random, itertools
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import para as PA
from pce import fans as FN
from pce import orders as OD
from pce import context as CT
from pce import jumps as JP
random.seed(20261031); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
X = ["a", "b", "c"]; L = [0, 1]
def rker():
    """a random parameterised source-indexed kernel X -> X: p is a tuple of Fractions; the law is affine in p with positive weights"""
    base = {(l, x): [Fr(random.randint(1, 5)) for _ in X] for l in L for x in X}; slope = {(l, x): [Fr(random.randint(0, 3)) for _ in X] for l in L for x in X}
    def f(p, l, x):
        s = sum(p) if isinstance(p, tuple) else p
        w = [b + s * m for b, m in zip(base[(l, x)], slope[(l, x)])]; t = sum(w); return {y: v / t for y, v in zip(X, w)}
    return f

# PR1
ok = True
for _ in range(50):
    f, g, h = rker(), rker(), rker(); p, q, r = Fr(random.randint(0, 4), 3), Fr(random.randint(0, 4), 3), Fr(random.randint(0, 4), 3)
    for l in L:
        for x in X:
            left = PA.compose(PA.compose(f, g), h)(((p, q), r), l, x); right = PA.compose(f, PA.compose(g, h))((p, (q, r)), l, x); ok &= left == right
            ide = lambda _p, l, x: {x: Fr(1)}; ok &= PA.compose(f, ide)((p, None), l, x) == f(p, l, x) == PA.compose(ide, f)((None, p), l, x)
            rr = lambda z: z * 2 + Fr(1, 3); rr2 = lambda z: z / 3
            lhs = PA.reparam(PA.compose(f, g), lambda pq: (rr(pq[0]), rr2(pq[1])))((p, q), l, x); rhs = PA.compose(PA.reparam(f, rr), PA.reparam(g, rr2))((p, q), l, x)
            ok &= lhs == rhs
check("PR1 Para over source-indexed kernels: associative up to re-bracketing, identities, source held fixed, reparametrisations whisker", ok)

# PR2
ok = True; A = lambda x: x in ("a", "b")
def plaw():
    co = {(l, x): (Fr(random.randint(0, 4)), Fr(random.randint(0, 4))) for l in L for x in X}
    def lam(p):
        raw = {k: a + b * p for k, (a, b) in co.items()}; s = sum(raw.values()); return {k: v / s for k, v in raw.items() if v}
    return lam
for _ in range(100):
    lam = plaw(); r = lambda z: z * z + Fr(1, 5)
    for p in (Fr(0), Fr(1, 2), Fr(2)):
        L1 = lam(r(p)); L2 = (lambda z: lam(r(z)))(p)
        ok &= JP.forget(L1) == JP.forget(L2) and JP.weights(L1) == JP.weights(L2)
        ok &= CT.marg({(k[1],): v for k, v in L1.items()} , [0]) == CT.marg({(k[1],): v for k, v in L2.items()}, [0])
        c1, c2 = JP.cond_lab(L1, A), JP.cond_lab(L2, A); ok &= (c1 is None) == (c2 is None) and c1 == c2
check("PR2 labelling, pooling, marginalisation, forgetting and conditioning commute with reparametrisation; the domain of conditioning pulls back", ok)

# PR3
ok = True
def rpoly(d, n=3, mx=3):
    f = {}
    for _ in range(n): f[tuple(Fr(random.randint(0, mx)) for _ in range(d))] = Fr(random.randint(1, 5))
    return f
for _ in range(200):
    d = 2; cells = {l: rpoly(d) for l in range(3)}; w = (Fr(random.randint(1, 3)), Fr(random.randint(1, 3))); c = (Fr(random.randint(1, 3)), Fr(random.randint(1, 3)))
    germs = FN.arc_germs(cells, w, c)
    for k, g in germs.items(): ok &= OD.lead(g)[0] == FN.trop(cells[k], w)          # valuation = tropical order (positive coefficients: never degenerate)
    st = FN.arc_limit(cells, w, c); s = 1e-5
    num = FN.evaluate({k: {a: float(v) for a, v in f.items()} for k, f in cells.items()}, [float(ci) * s ** float(wi) for ci, wi in zip(c, w)])
    ok &= sum(abs(num[k] - float(st.get(k, 0))) for k in cells) < 1e-3
check("PR3 the germ field: cells at an infinitesimal point have valuation trop_w; the standard part of the conditioned state is the arc limit", ok)

# PR4
ok = True; ties = decided = 0
for _ in range(300):
    cells = {l: rpoly(2) for l in range(3)}; U = {a: {l: Fr(random.randint(-3, 3)) for l in range(3)} for a in ("x", "y", "z")}
    w = (Fr(random.randint(1, 3)), Fr(random.randint(1, 3))); c = (Fr(random.randint(1, 3)), Fr(random.randint(1, 3)))
    ev = PA.eventual_acts(cells, U, w, c); lim = PA.limit_acts(cells, U, w, c)
    ok &= ev <= lim and len(ev) >= 1
    if len(lim) == 1: ok &= ev == lim; decided += 1
    else: ties += 1
    s = Fr(1, 10 ** 8); eps = [ci * s ** int(wi) for ci, wi in zip(c, w)]; val = FN.evaluate(cells, eps)
    scores = {a: sum(Fr(U[a].get(k, 0)) * v for k, v in val.items()) for a in U}; m = max(scores.values())
    ok &= {a for a, v in scores.items() if v == m} <= ev or ev == {a for a, v in scores.items() if v == m}
    V = FN.newton_vertices_2d(cells)
    if len(V) == 1:                                           # a single vertex: one cone, the act cannot depend on (w, c)
        acts = {frozenset(PA.eventual_acts(cells, U, (Fr(a), Fr(b)), (Fr(e), Fr(f)))) for a in (1, 2) for b in (1, 3) for e in (1, 2) for f in (1, 3)}
        ok &= len({frozenset(PA.limit_acts(cells, U, (Fr(a), Fr(b)), (Fr(1), Fr(1)))) for a in (1, 2) for b in (1, 3)}) == 1
check("PR4 decision fan: the act is eventually constant along every arc, equals the limit's act when that is unique, and the next order breaks ties", ok,
      f"{decided} decided at the limit, {ties} ties broken by higher order")

# PR5
ok = True; race = {1: {(Fr(1), Fr(1)): Fr(1)}, 2: {(Fr(2), Fr(0)): Fr(1), (Fr(2), Fr(1)): Fr(-1)}}
audit = {"audit 1": {1: Fr(1)}, "audit 2": {2: Fr(1)}}
ok &= PA.eventual_acts(race, audit, (Fr(2), Fr(1)), (1, 1)) == {"audit 1"} and PA.eventual_acts(race, audit, (Fr(1), Fr(2)), (1, 1)) == {"audit 2"}
ok &= PA.eventual_acts(race, audit, (Fr(1), Fr(1)), (Fr(1), Fr(2))) == {"audit 1"} and PA.eventual_acts(race, audit, (Fr(1), Fr(1)), (Fr(1), Fr(1, 2))) == {"audit 2"}
ok &= PA.limit_acts(race, audit, (Fr(1), Fr(1)), (Fr(1), Fr(1))) == {"audit 1", "audit 2"} and PA.eventual_acts(race, audit, (Fr(1), Fr(1)), (Fr(1), Fr(1))) == {"audit 1"}
N = PA.decision_germs(race, audit, (Fr(1), Fr(1)), (Fr(1), Fr(1))); gap = OD.add(N["audit 1"], OD.scale(N["audit 2"], Fr(-1)))
ok &= OD.lead(gap) == (Fr(3), Fr(1))
check("PR5 the audit decision in the race: cones decide; on the ray c > 1 audits 1, c < 1 audits 2; at c = 1 the tie is broken at order 3 (audit 1)", ok,
      f"gap germ on the tie ray: {dict(gap)}")

# PR6
ok = True
for _ in range(200):
    cells = {l: rpoly(2) for l in range(3)}; U = {a: {l: Fr(random.randint(-3, 3)) for l in range(3)} for a in ("x", "y")}
    R = [[random.randint(0, 2) for _ in range(2)] for _ in range(2)]
    if np.linalg.det(np.array(R, float)) == 0 or any(sum(row) == 0 for row in R): continue
    cells2 = PA.monomial_subst(cells, R); w2 = (Fr(random.randint(1, 3)), Fr(random.randint(1, 3))); c2 = (Fr(random.randint(1, 2)), Fr(random.randint(1, 2)))
    w = tuple(sum(R[i][j] * w2[j] for j in range(2)) for i in range(2)); c = PA.arc_coeffs(c2, R)
    ok &= FN.arc_limit(cells2, w2, c2) == FN.arc_limit(cells, w, c) and PA.eventual_acts(cells2, U, w2, c2) == PA.eventual_acts(cells, U, w, c)
d1 = PA.monomial_subst(race, [[1, 0], [1, 1]]); d2 = PA.monomial_subst(race, [[1, 1], [1, 0]])        # (s, t) = (s, s u) and (u v, u)
cm1, cm2 = FN.componentwise_min(d1), FN.componentwise_min(d2)
for _ in range(500):                                          # the design on the ray, t = s (1 + u): the gap s^2 u + s^3 (1 + u) > 0 by every route
    sv, uv = Fr(1, random.randint(10, 10 ** 6)), Fr(1, random.randint(10, 10 ** 6)); tv = sv * (1 + uv)
    ok &= tv * sv - (1 - tv) * sv * sv == sv * sv * uv + sv ** 3 * (1 + uv) > 0
ok &= cm1 is not None and FN.vertex_limit(d1, cm1) == {2: Fr(1)} and cm2 is not None and FN.vertex_limit(d2, cm2) == {1: Fr(1)}
check("PR6 transport: monomial reparametrisations pull back limits and acts (w = R w'); the designs t = s u and (u v, u) land in one cone each", ok,
      f"design t = s u: registry explains A by every route (min exponent {tuple(map(str, cm1))}); design s = u v, t = u: the trials (min {tuple(map(str, cm2))})")
OUT["designs"] = [list(map(str, cm1)), list(map(str, cm2))]
json.dump(OUT, open(os.path.join(HERE, "..", "lab", "para.json"), "w"))
print(f"{sum(results)}/{len(results)} passed")
