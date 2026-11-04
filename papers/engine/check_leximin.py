"""Checks for the selection theorem (companion note, section 11.6) and the column-binding example (modules pce/datawalls.py, pce/couplingfibre.py).
  LX1  the selection theorem: for 30 random positively spanning configurations (20 in R^2, 10 in R^3), the positive critical point u_L of
       sum_k e^(-L l_k(u)) (Newton's method, 300-digit arithmetic, L = 100, 200, 400) tends to the lexicographic max-min point, with
       L |u_L - u_lex| bounded
  LX2  the mechanism: at u_L the first-level terms drop out of the projected critical equation exactly, sum_{k not in S_1} w_k P v_k = 0, and
       L |l_k(u_L) - h_1| stays bounded for k in S_1
  LX3  consistency: the lexicographic max-min point is (1/3, 1/3) for CP^2 and for F_1 with alpha < 1/3, and ((1 + alpha)/4, (1 - alpha)/2) for alpha > 1/3
  LX4  coupling polytopes: the facet-leximin point equals the all-cell leximin coupling (300 random margins, shapes 2x3, 3x3, 2x4), and the
       positive critical point approaches it (20 random 2x3 margins, L = 100, 200)
  LX5  the column-binding example f = (2/5, 3/5), g = (1/2, 3/10, 1/5) (coupling-polytopes paper, Example 3.4(b)): the pentagon (1/5, 0), (2/5, 0),
       (1/10, 3/10), (0, 3/10), (0, 1/5), area 11/200; facets 11, 12, 13, 22, 23 (cell 21 never vanishes); level 1: h_1 = 1/10, S_1 = {13, 23}, the
       candidates the open segment x + y = 3/10, 1/10 < x < 1/5, whose endpoints have leading sets {11, 13, 22, 23} and {12, 13, 23} with no positive balance;
       level 2: h_2 = 3/20, S_2 = {11, 12, 22}, u* = (3/20, 3/20) (exact rationals)
  LX6  the finite-T correction: L (x_L - 3/20) -> ln 2 / 2, because cells 11 and 22 coincide on the segment and balance against cell 12 alone
  LX7  the five critical points (exact resultants, T = 10^-100, 10^-200): the positive one and a real one with y1, y2 < 0 over (3/20, 3/20), with
       L (x - 3/20) = ln 2 / 2 to 1e-3; the other three (one real, two complex) on the candidate segment, off the leximin point
  LX8  the thimble along the segment: the u1-marginal is uniform on [h, 3/10 - h] for h = 0.11, 0.13 and concentrates at 3/20 for h = 0.16
  LX9  the one-count example (g = (30, 44, 26)/100) is column-binding: levels (13/100: 13, 23), (27/200: 11, 12), and u* = (27/200, 27/200)
"""
import os, sys, random, math
from fractions import Fraction as Fr
import numpy as np
import mpmath as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import couplingfibre as CF, datawalls as DW, kahlerbridge as KB
random.seed(20261103); rng = np.random.default_rng(20261103); results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
mp.mp.dps = 300
def mp_crit(v, lam, L, u0):
    """Newton for grad (1/L) log sum e^(-L l_k) = 0 in multiprecision"""
    spread = float(np.ptp(np.array(v, float) @ np.array(u0, float) - np.array(lam, float))) + 1.0
    mp.mp.dps = int(L * spread / 2.3) + 80
    V = [[mp.mpf(x) for x in row] for row in v]; lm = [mp.mpf(x) for x in lam]; n = len(V[0]); u = [mp.mpf(x) for x in u0]; L = mp.mpf(L)
    for _ in range(200):
        z = [-L * (sum(V[k][i] * u[i] for i in range(n)) - lm[k]) for k in range(len(V))]; mx = max(z); w = [mp.e ** (zi - mx) for zi in z]; s = sum(w); w = [x / s for x in w]
        mu = [sum(w[k] * V[k][i] for k in range(len(V))) for i in range(n)]
        H = mp.matrix(n, n)
        for i in range(n):
            for j in range(n): H[i, j] = L * sum(w[k] * (V[k][i] - mu[i]) * (V[k][j] - mu[j]) for k in range(len(V)))
        d = mp.lu_solve(H, mp.matrix([m_ for m_ in mu])); step = mp.mpf(1)
        f0 = mx + mp.log(s)
        while True:                                   # backtracking on the convex objective
            un = [u[i] + step * d[i] for i in range(n)]; zn = [-L * (sum(V[k][i] * un[i] for i in range(n)) - lm[k]) for k in range(len(V))]; mxn = max(zn)
            if mxn + mp.log(sum(mp.e ** (zi - mxn) for zi in zn)) <= f0 or step < mp.mpf(10) ** -30: break
            step /= 2
        u = un
        if max(abs(x) for x in d) * step < mp.mpf(10) ** -60: break
    return np.array([float(x) for x in u]), u, w
def rconf(n):
    while True:
        m = random.randint(n + 2, n + 4); v = [tuple(random.randint(-2, 2) for _ in range(n)) for _ in range(m)]
        if any(all(x == 0 for x in r) for r in v) or len(set(v)) < m: continue
        V = np.array(v, float).T; from scipy.optimize import linprog
        r = linprog(np.zeros(m), A_eq=V, b_eq=np.zeros(n), bounds=[(1, None)] * m, method="highs")
        if r.status == 0 and np.linalg.matrix_rank(V) == n: return v, [round(x, 2) for x in rng.normal(0, 0.5, m)]

# LX1, LX2
ok1 = ok2 = True; rows = []
for n, cnt in ((2, 20), (3, 10)):
    for _ in range(cnt):
        v, lam = rconf(n); ul, lev = DW.lex_maxmin(v, lam); e = []; u0 = ul
        for L in (100, 200, 400):
            uf, umq, w = mp_crit(v, lam, L, u0); u0 = uf; e.append(np.abs(uf - ul).max() * L)
            h1, S1 = lev[0]; V = np.array(v, float); ok2 &= max(abs(float(V[k] @ uf - lam[k]) - h1) for k in S1) * L < 50
            R = [k for k in range(len(v)) if k not in S1]; B = V[list(S1)]
            P = np.eye(n) - (np.linalg.pinv(B) @ B if len(S1) else 0)
            proj = sum(float(w[k]) * (P @ V[k]) for k in R) if R else np.zeros(n); scale = max(float(w[k]) for k in R) if R else 1
            ok2 &= np.abs(proj).max() <= 1e-8 * max(scale, 1e-300)
        ok1 &= e[2] < 0.05 * 400 and e[2] <= 1.25 * e[1] + 1e-6 and e[1] <= 1.25 * e[0] + 1e-6; rows.append(round(e[2], 3))
check("LX1 the positive critical point tends to the lexicographic max-min point, L|u_L - u_lex| bounded (30 configurations)", ok1, f"L|err| at L = 400: max {max(rows)}")
check("LX2 the projected critical equation loses the first level exactly; first-level terms pinned to O(1/L)", ok2)

# LX3
ok = all(np.allclose(DW.lex_maxmin(*KB.F1(a))[0], (1 / 3, 1 / 3) if a < 1 / 3 else ((1 + a) / 4, (1 - a) / 2), atol=1e-9) for a in [k / 10 for k in range(1, 10) if k != 3])
ok &= np.allclose(DW.lex_maxmin(*KB.CP2)[0], (1 / 3, 1 / 3))
check("LX3 lexicographic max-min = the valuations of Theorem 3 and Example 9 on CP^2 and F_1", ok)

# LX4
ok = True; cnt = 0
def gen(fv, gv):
    n, m = len(fv), len(gv); d = (n - 1) * (m - 1); idx = {(i, j): i * (m - 1) + j for i in range(n - 1) for j in range(m - 1)}; v = {}; lam = {}
    for i in range(n - 1):
        for j in range(m - 1): e = np.zeros(d); e[idx[i, j]] = 1; v[i, j] = e; lam[i, j] = 0.0
        e = np.zeros(d); e[[idx[i, j] for j in range(m - 1)]] = -1; v[i, m - 1] = e; lam[i, m - 1] = -fv[i]
    for j in range(m - 1): e = np.zeros(d); e[[idx[i, j] for i in range(n - 1)]] = -1; v[n - 1, j] = e; lam[n - 1, j] = -gv[j]
    v[n - 1, m - 1] = np.ones(d); lam[n - 1, m - 1] = sum(fv[:n - 1]) - gv[m - 1]; return v, lam, idx
for n, m in ((2, 3), (3, 3), (2, 4)):
    for _ in range(100):
        fv = rng.dirichlet(np.ones(n) * 2); gv = rng.dirichlet(np.ones(m) * 2)
        if min(fv.min(), gv.min()) < 0.02: continue
        F = CF.facet_cells(fv, gv); vd, ld, idx = gen(list(fv), list(gv)); cells = [(c // m, c % m) for c in F]
        u, _ = DW.lex_maxmin([tuple(vd[c]) for c in cells], [ld[c] for c in cells]); A = CF.leximin(list(fv), list(gv))
        ok &= np.allclose([A[i, j] for (i, j) in sorted(idx, key=idx.get)], u, atol=1e-7); cnt += 1
done = 0
while done < 20:
    fv = [round(random.uniform(0.15, 0.85), 2)]; fv.append(round(1 - fv[0], 2)); gv = list(np.round(rng.dirichlet(np.ones(3) * 3), 2)); gv[2] = round(1 - gv[0] - gv[1], 2)
    if min(gv) < 0.05: continue
    F = CF.facet_cells(fv, gv); Vv, Ll = CF.cell_functions(fv[0], gv[0], gv[1]); vk = [Vv[k] for k in F]; lk = [Ll[k] for k in F]
    ul, _ = DW.lex_maxmin(vk, lk); e = []
    for L in (100, 200):
        uf, _, _ = mp_crit(vk, lk, L, ul); e.append(np.abs(uf - ul).max() * L)
    ok &= e[1] <= 1.25 * e[0] + 1e-6 and e[1] / 200 < 0.05; done += 1
check("LX4 coupling polytopes: facet-leximin = all-cell leximin coupling; the positive critical point approaches it", ok, f"{cnt} margins compared")

# LX5
f, g = (Fr(2, 5), Fr(3, 5)), (Fr(1, 2), Fr(3, 10), Fr(1, 5)); P, A = CF.polygon(f, g)
ok = set(P) == {(Fr(1, 5), Fr(0)), (Fr(2, 5), Fr(0)), (Fr(1, 10), Fr(3, 10)), (Fr(0), Fr(3, 10)), (Fr(0), Fr(1, 5))} and A == Fr(11, 200)
F = CF.facet_cells([0.4, 0.6], [0.5, 0.3, 0.2]); ok &= F == [0, 1, 2, 4, 5] and f[1] + g[0] == Fr(11, 10)
V, L = CF.cell_functions(f[0], g[0], g[1]); ell = lambda c, x, y: V[c][0] * x + V[c][1] * y - L[c]
h1, b1 = CF.maxmin_closed(f, g); ok &= h1 == Fr(1, 10) and b1 == [("col", 2)]
seg = [(x, Fr(3, 10) - x) for x in [Fr(1, 10) + Fr(k, 100) for k in range(0, 11)]]
ok &= all(ell(2, x, y) == Fr(1, 10) and ell(5, x, y) == Fr(1, 10) and all(ell(c, x, y) >= Fr(1, 10) for c in F) for x, y in seg)
vf = [V[c] for c in F]; lf = [float(L[c]) for c in F]
for (x, y), K in (((0.1, 0.2), (0, 2, 3, 4)), ((0.2, 0.1), (1, 2, 4))):
    ok &= KB.ties(vf, lf, (x, y), tol=1e-9) == K and not KB.balanced(vf, K)
lev2 = [min(ell(0, x, y), ell(1, x, y), ell(4, x, y)) for x, y in seg]; best = max(lev2); ok &= best == Fr(3, 20) and seg[lev2.index(best)] == (Fr(3, 20), Fr(3, 20))
ul, levels = DW.lex_maxmin(vf, lf); ok &= np.allclose(ul, (0.15, 0.15)) and [tuple(F[k] for k in S) for _, S in levels] == [(2, 5), (0, 1, 4)]
check("LX5 the column-binding example: pentagon, facets, level 1 (h = 1/10, cells 13, 23, a segment), level 2 (h = 3/20, cells 11, 12, 22), u* = (3/20, 3/20)", ok)

# LX6
ok = True; c6 = []
for Lg in (100, 200, 400):
    uf, _, _ = mp_crit(vf, lf, Lg, (0.15, 0.15)); c6.append(Lg * (uf[0] - 0.15))
ok &= abs(c6[2] - math.log(2) / 2) < 1e-6 and abs(c6[0] - math.log(2) / 2) < 5e-2
check("LX6 the finite-T correction L (x_L - 3/20) -> ln 2 / 2 (two coinciding cells against one)", ok, f"{[round(c, 6) for c in c6]} vs {math.log(2) / 2:.6f}")

# LX7
ok = True; info = []
for tau in ("1/10000000000", "1/100000000000000000000"):
    pts, D = CF.critical_points(f, g, tau, dps=500); T_ = mp.mpf(Fr(tau).numerator) / Fr(tau).denominator; Lg = float(-D * mp.log(T_)); ok &= len(pts) == 5
    at, off = [], []
    for y1, y2 in pts:
        u = np.array([float(mp.log(abs(y1)) / (D * mp.log(T_))), float(mp.log(abs(y2)) / (D * mp.log(T_)))]); real = abs(mp.im(y1)) < abs(y1) * 1e-30 and abs(mp.im(y2)) < abs(y2) * 1e-30
        sg = (int(np.sign(float(mp.re(y1)))), int(np.sign(float(mp.re(y2))))) if real else None
        (at if np.abs(u - 0.15).max() < 5e-3 else off).append((u, sg))
    ok &= len(at) == 2 and sorted(x[1] for x in at) == [(-1, -1), (1, 1)] and all(abs(Lg * (x[0][0] - 0.15) - math.log(2) / 2) < 1e-3 for x in at)
    ok &= len(off) == 3 and all(abs(x[0].sum() - 0.3) < 1e-5 and x[0][0] < 0.14 for x in off); info.append([tuple(np.round(x[0], 4)) for x in off][0])
check("LX7 five critical points: positive and (-,-) over (3/20, 3/20) with the ln 2 / 2 correction; three on the candidate segment, off the leximin", ok,
      f"the other three at {info}")

# LX8
ok = True
for h, lo, hi in ((0.11, 0.11, 0.19), (0.13, 0.13, 0.17)):
    U1, U2, w = KB.tropical_grid(vf, lf, -200.0, h, box=((0.05, 0.25), (0.05, 0.25)), n=801); m1 = w.sum(1); a1 = U1[:, 0]
    mean = (m1 * a1).sum(); var = (m1 * (a1 - mean) ** 2).sum(); ok &= abs(mean - 0.15) < 2e-3 and abs(var - (hi - lo) ** 2 / 12) < 3e-4
U1, U2, w = KB.tropical_grid(vf, lf, -200.0, 0.16, box=((0.05, 0.25), (0.05, 0.25)), n=801); ok &= w[np.abs(U1 - 0.15) < 0.01].sum() > 0.99
check("LX8 the thimble on the segment: uniform on [h, 3/10 - h] between the two levels, concentrated at 3/20 above", ok)

# LX9
g9 = (Fr(30, 100), Fr(44, 100), Fr(26, 100)); F9 = CF.facet_cells([0.4, 0.6], [0.3, 0.44, 0.26]); V9, L9 = CF.cell_functions(Fr(2, 5), g9[0], g9[1])
u9, lev9 = DW.lex_maxmin([V9[c] for c in F9], [float(L9[c]) for c in F9])
ok = np.allclose(u9, (0.135, 0.135)) and [(round(h, 6), tuple(CF.NAMES[F9[k]] for k in S)) for h, S in lev9][:2] == [(0.13, ("13", "23")), (0.135, ("11", "12"))]
check("LX9 the one-count example is column-binding: levels (13/100: 13, 23), (27/200: 11, 12); u* = (27/200, 27/200)", ok, f"{lev9}")
print(f"{sum(results)}/{len(results)} passed")
