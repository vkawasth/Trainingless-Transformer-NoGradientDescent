"""Checks for Part IV of the companion note: 2 x 4 coupling polytopes (module pce/cpl24.py).
  TF1  the Kahler chambers of 2 x 4 margins (sign vectors of f1 - g(J), J a nonempty proper set of columns): 148 are realised, each certified
       by an exact rational interior point (completeness rests on linear-programming infeasibility of the other sign vectors); every chamber
       polytope is simple and smooth (|det| = 1 at every vertex)
  TF2  the wall relations: on every edge of every chamber polytope, v_k + v_l + a v_i + b v_j = 0 with integers a, b, and the Kahler degree
       -lam_k - lam_l - a lam_i - b lam_j equals the lattice length of the edge (exact; an independent test of the relations)
  TF3  the toric Kleiman criterion chamber by chamber: -K.C = 2 + a + b on every invariant curve. Fano: CP^3 (2 chambers), Bl_pt CP^3 (8),
       (CP^1)^3 (4); weak Fano: 6, 7, 8 facets (24, 64, 46 chambers) with 1, 3, 6 curves of -K.C = 0; nothing else. The independent vertex test
       agrees on all 148; the normalised volume of conv{v} equals the number of vertices (= chi) in all 148; (-K)^3 = 64, 56, 48 for the Fano types
       and 48, 40, 32 for the weak Fano types with 6, 7, 8 facets
  TF4  the box criterion (exact, all chambers): Cpl(f, g) is a box iff some g_j > max(f1, f2); then the facet cells are the other three columns
       and the leximin levels are their half-masses g_k / 2 (example A: 1/40, 1/20, 1/10). The CP^3 chambers have one full row as facets, the
       Bl_pt CP^3 chambers one full row and one cell of the other row
  TF5  example A exactly: 8 critical points, y1 = +-T^(1/40), y3 = +-T^(1/20), P = +-T^(3/10), so all over the fibre (1/40, 9/40, 1/20); critical
       values 2(+-x +- x^2 +- x^4), x = T^(1/40); exactly one coincidence for T in (0, 1), between (+,-,-) and (-,+,+), at x^3 + x - 1 = 0
       (T* ~ 2.3e-7)
  TF6  example B exactly: the reduction gives -tau^19 (c^6 + tau^2 c^4 - tau^5), tau = T^(1/20), y2 = y3 = c, y1 = tau^3 / c^3; the cubic in c^2 has
       discriminant tau^10 (4 tau - 27), so 6 distinct critical points for T in (0, 1); its Newton polygon has one edge, so all six lie over the
       leximin fibre (1/40, 1/24, 1/24) with leading holonomies (zeta^-3, zeta, zeta), zeta^6 = 1; the critical values w = 2 y1 + 2 c are the roots
       of a sextic whose discriminant is 2^36 tau^19 (4 tau - 27)^6 (4 tau + 1)^2: distinct for every T in (0, 1)
  TF7  example C: the resultant has degree 10 in c (= the normalised volume); its Newton polygon in C = c^2 has edges of slopes 7/2 (four roots)
       and 1 (one root), so 8 critical points over the leximin fibre (1/20, 3/40, 7/80) and 2 over (1/40, 1/40, 9/40) (confirmed at L = 400);
       three invariant curves with -K.C = 0
  TF8  leximin exactly: A levels 1/40, 1/20, 1/10, drops (1, 1, 1); B levels 1/40, 1/24, drops (1, 2) (the residual row 1 spread over three
       cells); C levels 1/20, 3/40, 7/80, 13/80, drops (1, 1, 1, 0); tie-free random margins (600 samples) show only the drop patterns (3),
       (1, 2), (1, 1, 1) (+ trailing zeros): (3) alone on CP^3, (3) or (1, 2) on Bl_pt CP^3, (1, 1, 1) on the cube
  TF9  the level-drop spectral statement at the positive object: the Newton-polygon valuations of the eigenvalues of Q/W, estimated from
       L = 800 -> 1600, are 0 once and h_j - h_1 with multiplicity r_j (A: 1/40, 3/40; B: 1/60 twice; C: 1/40, 3/80; the non-dropping level
       13/80 contributes none), within 1e-5; and on C's second fibre the reduced rank is 3 = dim span of its leading set {e1, e2, -(1,1,1), -e3},
       with leading coefficients (i, i, i, -i) up to conjugation, not real of one sign; and the invariant-factor valuations e_k = d_k - d_(k-1)
       (d_k the least valuation of a k x k minor of Q/W, slopes L = 200 -> 400) equal the same list on every object over the leximin fibre
       (A: 8, B: 6, C: 8 objects)
"""
import os, sys, math, itertools
from fractions import Fraction as Fr
from collections import Counter
import numpy as np
import mpmath as mp
import sympy as sp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import cpl24 as C
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)

CH = C.chambers(); DATA = []
for signs, (f, g) in CH:
    cls, P, I = C.fano_class(f, g); DATA.append((signs, f, g, cls, P, I))

# TF1
ok = len(CH) == 148 and all(P["simple"] and P["smooth"] for _, _, _, _, P, _ in DATA)
ok &= all(all(s * (f[0] - sum(g[j] for j in J)) > 0 for s, J in zip(signs, C.SUBSETS)) for signs, f, g, _, _, _ in DATA)
check("TF1 148 Kahler chambers, each with an exact interior point; all simple and smooth", ok, f"{len(CH)} chambers")

# TF2
ne = sum(len(I) for *_, I in DATA); ok = all(c["a"] is not None and c["omega"] == c["length"] for *_, I in DATA for c in I)
check("TF2 integer wall relations on every edge; Kahler degree = lattice length of the edge (exact)", ok, f"{ne} edges")

# TF3
types = Counter(); okv = True; okchi = True; deg = {}
for signs, f, g, cls, P, I in DATA:
    z = sum(c["minusK"] == 0 for c in I); name = C.identify(f, g)[0]; types[(len(P["facets"]), cls, z, name)] += 1
    okv &= C.vertex_test(f, g) == cls; okchi &= round(C.normalized_volume(P)) == len(P["verts"])
    deg.setdefault(name or f"wF{len(P['facets'])}", set()).add(round(C.minus_K_cubed(P)[0]))
expect = {(4, "Fano", 0, "CP^3"): 2, (5, "Fano", 0, "P_(CP^2)(O + O(1)) = Bl_pt CP^3"): 8, (6, "Fano", 0, "CP^1 x CP^1 x CP^1"): 4,
          (6, "weak Fano", 1, None): 24, (7, "weak Fano", 3, None): 64, (8, "weak Fano", 6, None): 46}
ok = dict(types) == expect and okv and okchi and deg == {"CP^3": {64}, "P_(CP^2)(O + O(1)) = Bl_pt CP^3": {56}, "CP^1 x CP^1 x CP^1": {48}, "wF6": {48}, "wF7": {40}, "wF8": {32}}
check("TF3 toric Kleiman chamber by chamber: 14 Fano (CP^3 x2, Bl_pt CP^3 x8, (CP^1)^3 x4), 134 weak Fano (1, 3, 6 curves with -K.C = 0); vertex test agrees; vol = chi",
      ok, f"(-K)^3 {deg}")

# TF4
ok = all(C.is_box(f, g) == (C.identify(f, g)[0] == "CP^1 x CP^1 x CP^1") for _, f, g, *_ in DATA)
f, g = C.EX_A; lx = C.leximin_exact(f, g); P = C.polytope(f, g)
ok &= sorted(C.NAMES[k][1] for k in P["facets"]) == [0, 0, 2, 2, 3, 3] and lx["levels"] == sorted(g[k] / 2 for k in (0, 2, 3))
shapes = Counter()
for _, f_, g_, cls, P_, _ in DATA:
    nm = C.identify(f_, g_)[0]; rows = Counter(C.NAMES[k][0] for k in P_["facets"])
    if nm == "CP^3": shapes[(nm, tuple(sorted(rows.values())))] += 1
    if nm and nm.startswith("P_(CP^2)"): shapes[("Bl", tuple(sorted(rows.values())))] += 1
ok &= dict(shapes) == {("CP^3", (4,)): 2, ("Bl", (1, 4)): 8}
check("TF4 box iff some g_j > max(f1, f2), levels = half-masses of the other columns (A: 1/40, 1/20, 1/10); CP^3 facets = one row; Bl_pt CP^3 facets = one row + one cell",
      ok, f"A levels {[str(x) for x in lx['levels']]}; {dict(shapes)}")

# TF5
pts, b = C.critical_points_exact(f, g, 200.0); x = sp.Symbol("x")
uok = all(abs(-mp.log(abs(y[0])) / 200 - mp.mpf(1) / 40) < 1e-12 and abs(-mp.log(abs(y[2])) / 200 - mp.mpf(1) / 20) < 1e-12
          and abs(-mp.log(abs(y[0] * y[1] * y[2])) / 200 - mp.mpf(3) / 10) < 1e-12 for y in pts)
vals = {}; coinc = []
for sg in itertools.product((1, -1), repeat=3): vals[sg] = 2 * (sg[0] * x + sg[1] * x ** 2 + sg[2] * x ** 4)
for p, q in itertools.combinations(vals, 2):
    rr = [r for r in sp.real_roots(sp.Poly(sp.expand(vals[p] - vals[q]), x)) if 0 < r < 1]
    if rr: coinc.append((p, q, rr))
xs = sp.real_roots(sp.Poly(x ** 3 + x - 1, x)); ok = len(pts) == 8 == b and uok and len(coinc) == 1 and set(coinc[0][:2]) == {(1, -1, -1), (-1, 1, 1)}
ok &= len(xs) == 1 and coinc[0][2][0] == xs[0]
check("TF5 A: 8 critical points over (1/40, 9/40, 1/20); critical values 2(+-x +- x^2 +- x^4); one coincidence, (+,-,-) = (-,+,+) at x^3 + x - 1 = 0", ok,
      f"T* = {float(xs[0]) ** 40:.3e}")

# TF6
f, g = C.EX_B; D, E, Fp, R = C.reduction(f, g); t, c, w, Cc = sp.symbols("tau c w C")
ok = D == 20 and sp.expand(R + t ** 19 * (c ** 6 + t ** 2 * c ** 4 - t ** 5)) == 0
ok &= sp.factor(sp.discriminant(Cc ** 3 + t ** 2 * Cc ** 2 - t ** 5, Cc) - t ** 10 * (4 * t - 27)) == 0
Q = sp.factor_list(sp.resultant(sp.numer(sp.together(w - 2 * t ** 3 / c ** 3 - 2 * c)), c ** 6 + t ** 2 * c ** 4 - t ** 5, c))[1][-1][0]
dq = sp.factor(sp.discriminant(Q, w)); ok &= sp.expand(dq - 2 ** 36 * t ** 19 * (4 * t - 27) ** 6 * (4 * t + 1) ** 2) == 0
pts, b = C.critical_points_exact(f, g, 400.0); ok &= len(pts) == 6 == b
uok = all(abs(-mp.log(abs(y[0])) / 400 - mp.mpf(1) / 40) < 2e-3 and abs(-mp.log(abs(y[1])) / 400 - mp.mpf(1) / 24) < 2e-3 and abs(y[1] - y[2]) < abs(y[1]) * 1e-20 for y in pts)
zs = sorted(round(float(mp.arg(y[1]) / (mp.pi / 3))) % 6 for y in pts); hol = all(abs(mp.arg(y[0]) - mp.arg(y[1] ** -3)) % (2 * mp.pi) < 1e-2 or
                                                                            abs(abs(mp.arg(y[0]) - mp.arg(y[1] ** -3)) - 2 * mp.pi) < 1e-2 for y in pts)
ok &= uok and zs == [0, 1, 2, 3, 4, 5] and hol
check("TF6 B: c^6 + tau^2 c^4 - tau^5; 6 distinct points over (1/40, 1/24, 1/24), holonomies (zeta^-3, zeta, zeta); critical values distinct for all T in (0, 1)", ok,
      f"disc_w = {dq}")

# TF7
f, g = C.EX_C; D, E, Fp, R = C.reduction(f, g); num = sp.Poly(sp.numer(sp.together(R)), c)
coeffs = {m[0]: sp.Poly(co, t) for m, co in zip(num.monoms(), num.coeffs())}
pts2 = sorted((k // 2, min(mono[0] for mono in co.monoms())) for k, co in coeffs.items()); m0 = min(v for _, v in pts2); pts2 = [(k, v - m0) for k, v in pts2]
ok = num.degree() == 10 and all(k % 2 == 0 for k in coeffs) and pts2 == [(0, 15), (1, 13), (2, 8), (3, 5), (4, 1), (5, 0)]
pts, b = C.critical_points_exact(f, g, 400.0); fib = Counter()
for y in pts: fib[tuple(round(float(-mp.log(abs(v)) / 400) * 80) for v in y)] += 1
ok &= len(pts) == 10 == b and dict(fib) == {(4, 6, 7): 8, (2, 2, 18): 2}
z = sum(cc["minusK"] == 0 for cc in C.intersections(f, g)[1]); ok &= z == 3 and C.fano_class(f, g)[0] == "weak Fano"
check("TF7 C: resultant of degree 10; Newton polygon in c^2: slopes 7/2 (x4), 1 (x1); 8 points over (1/20, 3/40, 7/80), 2 over (1/40, 1/40, 9/40); 3 curves with -K.C = 0",
      ok, f"fibres x80 {dict(fib)}")

# TF8
LA = C.leximin_exact(*C.EX_A); LB = C.leximin_exact(*C.EX_B); LC = C.leximin_exact(*C.EX_C)
ok = (LA["levels"] == [Fr(1, 40), Fr(1, 20), Fr(1, 10)] and LA["drops"] == [1, 1, 1] and LB["levels"] == [Fr(1, 40), Fr(1, 24)] and LB["drops"] == [1, 2]
      and LC["levels"] == [Fr(1, 20), Fr(3, 40), Fr(7, 80), Fr(13, 80)] and LC["drops"] == [1, 1, 1, 0])
rng = np.random.default_rng(11); pats = Counter()
for _ in range(600):
    a = rng.dirichlet([2, 2]); bb = rng.dirichlet([2] * 4); f = (Fr(a[0]).limit_denominator(10 ** 6),); f = (f[0], 1 - f[0])
    g = tuple(Fr(x).limit_denominator(10 ** 6) for x in bb[:3]); g = g + (1 - sum(g),)
    if any(f[0] == sum(g[j] for j in J) for J in C.SUBSETS): continue
    cls, P, _ = C.fano_class(f, g); key = C.identify(f, g)[0] or f"wF{len(P['facets'])}"
    pats[(key, tuple(d for d in C.leximin_exact(f, g)["drops"] if d))] += 1
ok &= {p for _, p in pats} <= {(3,), (1, 2), (1, 1, 1)}
ok &= {p for k_, p in pats if k_ == "CP^3"} == {(3,)} and {p for k_, p in pats if k_.startswith("P_(")} <= {(3,), (1, 2)} and {p for k_, p in pats if k_.startswith("CP^1")} <= {(1, 1, 1)}
check("TF8 leximin exactly: A (1,1,1), B (1,2), C (1,1,1,0); census: only (3), (1,2), (1,1,1); CP^3 (3); Bl_pt CP^3 (3), (1,2); cube (1,1,1)", ok,
      f"census {dict(sorted(pats.items()))}")

# TF9
pred = {"A": [0, Fr(1, 40), Fr(3, 40)], "B": [0, Fr(1, 60), Fr(1, 60)], "C": [0, Fr(1, 40), Fr(3, 80)]}; ok = True; got = {}
for k, (f, g) in {"A": C.EX_A, "B": C.EX_B, "C": C.EX_C}.items():
    lv = {}
    for L in (800.0, 1600.0):
        y = C.positive_point(f, g, L); ev, W = C.clifford_spectrum(f, g, L, y); lv[L] = sorted(mp.log(abs(e)) for e in ev)[::-1]
    sl = sorted(float(-(lv[1600.0][i] - lv[800.0][i]) / 800) for i in range(3)); got[k] = [round(s, 7) for s in sl]
    ok &= all(abs(s - float(p)) < 1e-5 for s, p in zip(sl, pred[k]))
f, g = C.EX_C; pts, b = C.critical_points_exact(f, g, 200.0); pts4, _ = C.critical_points_exact(f, g, 400.0); ranks = []; leads = []
for y in pts:
    u = tuple(round(float(-mp.log(abs(v)) / 200) * 80) for v in y)
    if u != (2, 2, 18): continue
    y4 = min(pts4, key=lambda q: abs(mp.arg(q[0]) - mp.arg(y[0])) + abs(mp.log(abs(q[0])) / 400 - mp.log(abs(y[0])) / 200))
    e2, _ = C.clifford_spectrum(f, g, 200.0, y); e4, _ = C.clifford_spectrum(f, g, 400.0, y4)
    sl = [float(-(mp.log(abs(b4)) - mp.log(abs(b2))) / 200) for b2, b4 in zip(e2, e4)]; ranks.append(sum(abs(s) < 5e-3 for s in sl))
    parts = C._W_parts(C.potential(f, g), 400.0, y4); rho = [complex(p_ / mp.exp(-400.0 / 40)) for p_ in parts]
    K = [i for i, r in enumerate(rho) if abs(r) > 0.5]; leads.append(sorted((round(r.real, 2), round(r.imag, 2)) for r in (rho[i] for i in K)))
ok &= ranks == [3, 3] and len(leads) == 2 and all(len(l) == 4 and all(abs(a) < 0.05 for a, _ in l) for l in leads)
ok &= sorted(leads) == sorted([[(0.0, -1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], [(0.0, -1.0), (0.0, -1.0), (0.0, -1.0), (0.0, 1.0)]])
def minors_val(M, L):
    out = []
    for k in (1, 2, 3):
        best = max(abs(mp.det(mp.matrix([[M[i, j] for j in J] for i in I]))) for I in itertools.combinations(range(3), k) for J in itertools.combinations(range(3), k))
        out.append(mp.log(best))
    return out
sc = {}; okm = True
for k, (f, g), u0, want in (("A", C.EX_A, (1, 9, 2), [0, Fr(1, 40), Fr(3, 40)]), ("B", C.EX_B, (3, 5, 5), [0, Fr(1, 60), Fr(1, 60)]),
                             ("C", C.EX_C, (4, 6, 7), [0, Fr(1, 40), Fr(3, 80)])):
    den = {"A": 40, "B": 120, "C": 80}[k]; p2, _ = C.critical_points_exact(f, g, 200.0); p4, _ = C.critical_points_exact(f, g, 400.0); n = 0
    for y in p2:
        if tuple(round(float(-mp.log(abs(v)) / 200) * den) for v in y) != (u0 if k != "B" else (3, 5, 5)): continue
        y4 = min(p4, key=lambda q: sum(abs(mp.arg(q[i]) - mp.arg(y[i])) for i in range(3)))
        Ms = []
        for L, yy in ((200.0, y), (400.0, y4)):
            W, gr, H = C._grad_hess(C.potential(f, g), L, yy); Ms.append(minors_val(H / W, L))
        d = [float(-(Ms[1][i] - Ms[0][i]) / 200) for i in range(3)]; e = [d[0], d[1] - d[0], d[2] - d[1]]
        okm &= all(abs(a - float(b)) < 3e-3 for a, b in zip(e, want)); n += 1
    sc[k] = n
ok &= okm and sc == {"A": 8, "B": 6, "C": 8}
check("TF9 level-drop spectrum at the positive object: valuations 0 and h_j - h_1 with multiplicity r_j (A, B, C); C's second fibre: reduced rank 3 = span; invariant-factor valuations on every leximin-fibre object", ok,
      f"slopes {got}; objects {sc}")

print(f"{sum(results)}/{len(results)} passed")
