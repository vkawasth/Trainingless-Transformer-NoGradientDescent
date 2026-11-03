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
  DD6  an isolated degenerate direction, f1 = (y - x)^2 + (z - x)^2 against f2 = x^3: the degenerate face has one positive zero direction
       (x = y = z); after the weighted blow-up the cells pull back to t^6 (a^2 + b^2, 1) and tau^6 s'^2 (1 + b'^2, s') (sympy, exact); the
       residual generators have no common zero; every fibre point is approached from the positive orthant; the residual shares agree
       with exact arc limits (200 arcs), and their images (a^2+b^2)/(1+a^2+b^2) and (1+b'^2)/(1+b'^2+s') cover [0, 1]
  DD7  Example 63 near its degenerate plane: chart A of the blow-up of the z-axis, then four unimodular toric charts (both signs of u)
       compatible with the wall of (u^2 z, x^2): in every chart one residual generator is 1 or (1 + u)^2, so none share a zero for |u| < 1/2;
       the residual shares agree with exact arc limits along 400 lifted arcs; the fibre images cover [0, 1], and values strictly inside
       come only from the exceptional divisors of the wall rays (1, 1, 0) and (1, 0, 2)
  DD8  crossing degenerate planes, f1 = ((x - y)(x - z))^2 against f2 = x^5: the initial form vanishes on the two planes x = y and x = z, with
       multiplicity 2 on each and 4 where they cross; along each plane the critical exponent is single (share a^2 b^2/(1 + a^2 b^2)),
       while at the crossing (x = t^8) every pair with e1 + e2 = 20 is critical (exact arc limits); after one blow-up at the crossing the cells are
       x^4 (u^2 v^2, x): monomial, normal crossings
  DD9  tangent degenerate curves, f1 = ((z - x)((z - x) x^(k-1) - (y - x)^k))^2: in the blow-up chart the degenerate curves v = 0 and v = u^k
       have contact order k; after j < k - 1 point blow-ups the two branches through the origin are still tangent, and after exactly
       k - 1 the residual is u^(2(k-1)) v (v - u), two transverse branches: normal crossings (sympy, k = 1, ..., 4)
  DD10 Newton data do not see it: replacing every coefficient by its absolute value keeps the support, hence the Newton polyhedron, and
       removes every degenerate direction, for the crossing example and every k
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

# DD6
import sympy as sp
X, Y, Z, t, a, b, tau, s1, bb = sp.symbols("X Y Z t a b tau s1 bb", real=True)
F1 = (Y - X) ** 2 + (Z - X) ** 2; F2 = X ** 3
c1 = {X: t ** 2, Y: t ** 2 * (1 + t * a), Z: t ** 2 * (1 + t * b)}
ok = sp.expand(F1.subs(c1) - t ** 6 * (a ** 2 + b ** 2)) == 0 and sp.expand(F2.subs(c1) - t ** 6) == 0
for sg in (1, -1):
    c2 = {X: tau ** 2 * s1, Y: tau ** 2 * s1 * (1 + sg * tau), Z: tau ** 2 * s1 * (1 + tau * bb)}
    ok &= sp.expand(F1.subs(c2) - tau ** 6 * s1 ** 2 * (1 + bb ** 2)) == 0 and sp.expand(F2.subs(c2) - tau ** 6 * s1 ** 3) == 0
cellsI = {1: {(2, 0, 0): Fr(2), (1, 1, 0): Fr(-2), (0, 2, 0): Fr(1), (1, 0, 1): Fr(-2), (0, 0, 2): Fr(1)}, 2: {(3, 0, 0): Fr(1)}}
PI = total(cellsI); W = [(p, q, r) for p in range(1, 5) for q in range(1, 5) for r in range(1, 5)]
degI = [F for F in DD.faces(PI, W) if DD.degeneracy_gap(PI, F) < 1e-8]
ok &= len(degI) == 1 and set(degI[0]) == set(cellsI[1])
imgs = []; nar = 0
for _ in range(100):
    A, B = Fr(random.randint(-20, 20), random.randint(1, 5)), Fr(random.randint(-20, 20), random.randint(1, 5))
    lim = DD.arc_limit(cellsI, [{2: Fr(1)}, {2: Fr(1), 3: A}, {2: Fr(1), 3: B}])[1][1]; pred = (A * A + B * B) / (1 + A * A + B * B)
    ok &= lim == pred; imgs.append(float(pred)); nar += 1
    tt = 1e-3 / (1 + abs(float(A)) + abs(float(B))); ok &= tt ** 2 > 0 and 1 + tt * float(A) > 0 and 1 + tt * float(B) > 0   # approachable
for _ in range(100):
    S1, B2, sg = Fr(random.randint(0, 30), random.randint(1, 5)), Fr(random.randint(-20, 20), random.randint(1, 5)), random.choice([1, -1])
    if S1 == 0: imgs.append(1.0); continue
    lim = DD.arc_limit(cellsI, [{2: S1}, {2: S1, 3: sg * S1}, {2: S1, 3: B2 * S1}])[1][1]; pred = (1 + B2 * B2) / (1 + B2 * B2 + S1)
    ok &= lim == pred; imgs.append(float(pred)); nar += 1
u_ = np.linspace(0, 1, 400); imgs += list((u_ ** 2 * 2) / (1 + u_ ** 2 * 2)) + list(1 / (1 + np.geomspace(1e-3, 1e3, 400)))
v = sorted(imgs); gapI = max(q - p for p, q in zip(v, v[1:]))
check("DD6 an isolated degenerate direction: one weighted blow-up; residual generators (a^2+b^2, 1) and (1+b'^2, s') never vanish together; "
      "shares = exact arc limits; images cover [0, 1]", ok and gapI < 0.01 and v[0] == 0 and v[-1] == 1, f"{nar} arcs; largest gap {gapI:.4f}")

# DD7
ok = True; rng7 = np.random.default_rng(7); shares = []; inner_from_wall = True; nar = 0
f1o = {(2, 0, 1): Fr(1), (1, 1, 1): Fr(-2), (0, 2, 1): Fr(1)}; f2o = {(2, 2, 0): Fr(1)}; cells63 = {1: f1o, 2: f2o}
xs, us, zs, y1, y2, y3 = sp.symbols("x u z y1 y2 y3", positive=True)
for cone in DD.EX63_FAN:
    ok &= abs(round(np.linalg.det(np.array(cone, float)))) == 1
    d = [DD.wall_form(g) for g in cone]; ok &= all(x >= 0 for x in d) or all(x <= 0 for x in d)
    for sg in (1, -1):
        sub = {xs: y1 ** cone[0][0] * y2 ** cone[1][0] * y3 ** cone[2][0], us: sg * y1 ** cone[0][1] * y2 ** cone[1][1] * y3 ** cone[2][1],
               zs: y1 ** cone[0][2] * y2 ** cone[1][2] * y3 ** cone[2][2]}
        P1 = sp.factor((xs ** 2 * us ** 2 * zs).subs(sub)); P2 = sp.factor((xs ** 4 * (1 + us) ** 2).subs(sub))
        g = sp.gcd(P1, P2); r1, r2 = sp.simplify(P1 / g), sp.simplify(P2 / g)
        ok &= (r1 == 1 or sp.simplify(r2 / (1 + us.subs(sub)) ** 2).is_polynomial(y1, y2, y3)) and (r1 == 1 or r2 == (1 + us.subs(sub)) ** 2 or sp.expand(r2 - (1 + us.subs(sub)) ** 2) == 0)
        for _ in range(50):                                                   # lifted arcs y_j = c_j t^k_j, with x, z -> 0
            while True:
                k = [int(rng7.integers(0, 3)) for _ in range(3)]
                if any(k[j] > 0 and cone[j][0] > 0 for j in range(3)) and any(k[j] > 0 and cone[j][2] > 0 for j in range(3)): break
            cc = [Fr(int(rng7.integers(1, 6)), int(rng7.integers(1, 4))) for _ in range(3)]
            ex = [sum(k[j] * cone[j][i] for j in range(3)) for i in range(3)]; co = [np.prod([cc[j] ** cone[j][i] for j in range(3)], dtype=object) for i in range(3)]
            if ex[1] == 0 and abs(float(co[1])) >= 0.5: continue                  # stay in |u| < 1/2
            Xa = {ex[0]: co[0]}; Ya = dict(Xa); Ya[ex[0] + ex[1]] = Ya.get(ex[0] + ex[1], 0) + sg * co[0] * co[1]; Za = {ex[2]: co[2]}
            lim = DD.arc_limit(cells63, [Xa, Ya, Za])[1][1]
            ylim = [float(cc[j]) if k[j] == 0 else 0.0 for j in range(3)]; pred = DD.ex63_residual_share(cone, sg, ylim)
            ok &= abs(float(lim) - pred) < 1e-12; nar += 1
        for S in itertools.product((0, 1), repeat=3):                         # fibre points: coordinates in S set to 0
            if not (any(S[j] and cone[j][0] > 0 for j in range(3)) and any(S[j] and cone[j][2] > 0 for j in range(3))): continue
            for _ in range(60):
                yv = [0.0 if S[j] else float(np.exp(rng7.uniform(-4, 4))) for j in range(3)]
                if abs(DD.ex63_chart(cone, sg, yv)[0][1]) >= 0.5: continue
                sh = DD.ex63_residual_share(cone, sg, yv); shares.append(sh)
                if 1e-12 < sh < 1 - 1e-12: inner_from_wall &= all(DD.wall_form(cone[j]) == 0 for j in range(3) if S[j])
v = sorted(set([0.0, 1.0] + shares)); gap7 = max(q - p for p, q in zip(v, v[1:]))
check("DD7 Example 63: chart A then four unimodular toric charts, both signs of u; residual generators never vanish together for |u| < 1/2; "
      "shares = exact arc limits; fibre images cover [0, 1], interior values only on the wall-ray divisors",
      ok and inner_from_wall and gap7 < 0.02 and min(shares) < 1e-6 and max(shares) > 1 - 1e-6, f"{nar} lifted arcs; largest gap {gap7:.4f}")

# DD8
ok = True
x, y, z, u, v = sp.symbols("x y z u v")
F1 = sp.expand(((x - y) * (x - z)) ** 2)
def mult(f, pt):
    X_, Y_, Z_ = sp.symbols("X_ Y_ Z_"); g = sp.expand(f.subs({x: pt[0] + X_, y: pt[1] + Y_, z: pt[2] + Z_}))
    return min(sum(m) for m in sp.Poly(g, X_, Y_, Z_).as_dict())
ok &= mult(F1, (1, 1, 1)) == 4 and mult(F1, (1, 1, 2)) == 2 and mult(F1, (1, 2, 1)) == 2 and mult(F1, (1, 2, 3)) == 0
ok &= sp.expand(F1.subs({y: x * (1 + u), z: x * (1 + v)}) - x ** 4 * u ** 2 * v ** 2) == 0
cellsX = {1: {k: Fr(int(c)) for k, c in sp.Poly(F1, x, y, z).as_dict().items()}, 2: {(5, 0, 0): Fr(1)}}
for _ in range(40):
    A, B = Fr(random.randint(1, 9), random.randint(1, 4)), Fr(random.randint(1, 9), random.randint(1, 4))
    # along the plane x = y, away from x = z: x = t^2, y = t^2 + A t^3, z = t^2 (1 + B)
    lim = DD.arc_limit(cellsX, [{2: Fr(1)}, {2: Fr(1), 3: A}, {2: 1 + B}])[1][1]; ok &= lim == A * A * B * B / (1 + A * A * B * B)
    # at the crossing: x = t^8, y = t^8 + A t^e1, z = t^8 + B t^e2 with e1, e2 > 8: f1 = A^2 B^2 t^(2 e1 + 2 e2) against f2 = t^40
    for e1, e2 in ((9, 11), (10, 10), (11, 9)):
        lim = DD.arc_limit(cellsX, [{8: Fr(1)}, {8: Fr(1), e1: A}, {8: Fr(1), e2: B}])[1][1]; ok &= lim == A * A * B * B / (1 + A * A * B * B)
    ok &= DD.arc_limit(cellsX, [{8: Fr(1)}, {8: Fr(1), 9: A}, {8: Fr(1), 10: B}])[1][1] == 1                   # e1 + e2 < 20: source 1
    ok &= DD.arc_limit(cellsX, [{8: Fr(1)}, {8: Fr(1), 10: A}, {8: Fr(1), 11: B}])[1][2] == 1                  # e1 + e2 > 20: source 2
check("DD8 crossing planes: multiplicity 2 on each plane and 4 at the crossing; one critical exponent along a plane, a whole line "
      "e1 + e2 = 20 of them at the crossing; after one blow-up the cells are x^4 (u^2 v^2, x), normal crossings", ok)

# DD9
ok = True; counts = []
for k in (1, 2, 3, 4):
    G = sp.expand(((z - x) * ((z - x) * x ** (k - 1) - (y - x) ** k)) ** 2)
    pulled = sp.factor(G.subs({y: x * (1 + u), z: x * (1 + v)}))
    ok &= sp.expand(pulled - x ** (2 * k + 2) * (v * (v - u ** k)) ** 2) == 0
    cur = v * (v - u ** k); j = 0
    while True:
        # does the residual (after removing powers of u) still have two branches through the origin u = v = 0?
        red = sp.factor(sp.expand(cur))
        facs = [f for f, m in sp.factor_list(red)[1] if f.free_symbols & {v}]
        through = [f for f in facs if f.subs({u: 0, v: 0}) == 0]
        lin = [sp.Poly(f, u, v).as_dict() for f in through]
        tang = [(d.get((1, 0), 0), d.get((0, 1), 0)) for d in lin]                 # linear parts (coefficients of u and v)
        transverse = len(through) == 2 and all(any(tv) for tv in tang) and tang[0][0] * tang[1][1] - tang[0][1] * tang[1][0] != 0
        if len(through) <= 1 or transverse: break                                  # normal crossings
        cur = sp.expand(cur.subs(v, u * v)); j += 1                          # blow up the origin, chart v -> u v
    final = sp.factor(cur)
    ok &= j == k - 1 and sp.expand(final - u ** (2 * (k - 1)) * v * (v - u)) == 0; counts.append(j)
check("DD9 tangent degenerate curves of contact order k need exactly k - 1 point blow-ups before they cross transversally (normal crossings)", ok,
      f"blow-ups needed for k = 1..4: {counts}")

# DD10
ok = True
fams = [F1] + [sp.expand(((z - x) * ((z - x) * x ** (k - 1) - (y - x) ** k)) ** 2) for k in (1, 2, 3, 4)]
for Fk in fams:
    d = {kk: Fr(int(c)) for kk, c in sp.Poly(Fk, x, y, z).as_dict().items()}; dabs = {kk: abs(c) for kk, c in d.items()}
    W3 = [(1, 1, 1)]
    F_ = tuple(sorted(DD.initial_form(d, W3[0]))); ok &= DD.degeneracy_gap(d, F_) < 1e-8
    ok &= set(d) == set(dabs) and all(DD.degeneracy_gap(dabs, Fa) == 1.0 for Fa in DD.faces(dabs, [(a_, b_, c_) for a_ in range(1, 4) for b_ in range(1, 4) for c_ in range(1, 4)]))
check("DD10 Newton data do not see the degenerate locus: absolute-value coefficients keep the support and remove every degenerate direction", ok)
print(f"{sum(results)}/{len(results)} passed")
