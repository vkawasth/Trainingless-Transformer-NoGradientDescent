"""Checks for section 11 of the companion note, one coupling polytope through every layer (module pce/couplingfibre.py).
  CF1  the fibre: Cpl(f, g) for f = (2/5, 3/5), g = (3/10, 43/100, 27/100) is the pentagon with vertices (13/100, 0), (3/10, 0), (3/10, 1/10), (0, 2/5),
       (0, 13/100) and area 1331/20000 (exact); its facets are the cells 11, 12, 13, 21, 23; cell 22 never vanishes (a22 >= 3/100); consecutive
       normals have determinant 1 (Delzant)
  CF2  the tables: the integer tables with margins (40, 60), (30, 43, 27) are the 715 lattice points of 100 Cpl(f, g), and 715 = A + B/2 + 1 with A = 1331/2,
       B = 97 (Pick)
  CF3  Tier A on the fibre: with weights 1 / prod t_ij! the monomial law is Fisher's noncentral hypergeometric family; prod theta^t / t! restricted to
       the fibre is c_t e^(<u, (t11, t12)>) times a constant, with u the two log odds ratios (random theta); at u = 0 the mean is exactly
       100 f (x) g = (12, 86/5) (exact rationals); the Fisher information is positive definite; the law with mean 100 x leximin exists (Birch)
  CF4  the max-min cell: h* = min(min_i f_i / m, min_j g_j / n) against the facet-only linear programme (about 1000 random margins with every entry >= 1/100, shapes 2x3, 3x3,
       2x4, 3x4); the leading set is the binding row or column, every cell of which is a facet; the candidate dimension is (n-2)(m-1) for a binding
       row and (n-1)(m-2) for a binding column (2x3, 3x3); the example: h* = 2/15, binding row 1, exact dual certificate
  CF5  the critical points: W has 5 = 2 x area of its Newton polygon (Kushnirenko); exact resultants at T = 10^-100, 10^-200: three lie over the
       leximin coupling (2/15, 2/15), one of them positive; one real with y1, y2 < 0 over (13/100, 13/100), leading cells {11, 12, 23}, signed balance
       only; one real with y2 < 0 over (1/5, 1/10), leading cells {12, 13, 21}, signed balance only; valuations approach their limits as T decreases
  CF6  the positive valuation is the leximin coupling: for 20 random rational 2x3 margins, the positive critical point (T = 10^-60) sits over the
       (x, y) of the leximin coupling (tolerance 1e-2, margins with a gap of at least 1/50 between the two smallest ratios)
  CF7  the positive thimble, tropically: at h = 0 and h = 1/10 its mean and covariance tend to those of the uniform law on Q_h (log T = -200, -800);
       at h = h* the mass concentrates at the leximin coupling
  CF8  one count: moving one observation from column 3 to column 2 (g = (30, 44, 26)/100) makes column 3 bind: h* = 13/100, leading cells {13, 23},
       candidates a segment, leximin (27/200, 27/200); the example is 1/600 from that leximin wall and 3/100 from the nearest Kahler-chamber wall
       f1 = g2; the both-negative critical point moves onto the leximin fibre, so the distinct valuations go from 3 to 2 (exact resultants)
"""
import os, sys, random, itertools, math
from fractions import Fraction as Fr
import numpy as np
import mpmath as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import couplingfibre as CF, datawalls as DW, kahlerbridge as KB
random.seed(20261103); rng = np.random.default_rng(20261103); results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
f, g = CF.EX_F, CF.EX_G; ff, gf = [float(x) for x in f], [float(x) for x in g]

# CF1
P, A = CF.polygon(f, g); want = {(Fr(13, 100), Fr(0)), (Fr(3, 10), Fr(0)), (Fr(3, 10), Fr(1, 10)), (Fr(0), Fr(2, 5)), (Fr(0), Fr(13, 100))}
ok = set(P) == want and A == Fr(1331, 20000) and CF.facet_cells(ff, gf) == [0, 1, 2, 3, 5]
V, L = CF.cell_functions(f[0], g[0], g[1]); ok &= min(V[4][0] * x + V[4][1] * y - L[4] for x, y in P) == Fr(3, 100)
Fn = [V[k] for k in (0, 3, 2, 1, 5)]                       # cyclic order of the facet normals: 11 (1,0), 23 (1,1), 12 (0,1), 21 (-1,0), 13 (-1,-1)
Fn = sorted([V[k] for k in (0, 1, 2, 3, 5)], key=lambda w: math.atan2(w[1], w[0]))
ok &= all(abs(Fn[i][0] * Fn[(i + 1) % 5][1] - Fn[i][1] * Fn[(i + 1) % 5][0]) == 1 for i in range(5))
check("CF1 the pentagon: vertices, area 1331/20000, facets = cells 11,12,13,21,23, a22 >= 3/100, Delzant (exact)", ok, f"{sorted(P)}")

# CF2
Tb = CF.tables(100, f, g); Pi = [(int(100 * x), int(100 * y)) for x, y in P]
B = sum(math.gcd(abs(Pi[i][0] - Pi[(i + 1) % 5][0]), abs(Pi[i][1] - Pi[(i + 1) % 5][1])) for i in range(5))
ok = len(Tb) == 715 and B == 97 and Fr(len(Tb)) == A * 10000 + Fr(B, 2) + 1 and all(np.all(t >= 0) for _, t in Tb)
check("CF2 715 tables = lattice points of 100 Cpl = A + B/2 + 1 (Pick, B = 97)", ok)

# CF3
ok = True; S = [s for s, _ in Tb]; lw = np.array([CF.hypergeom_weights(t) for _, t in Tb]); c = np.exp(lw - lw.max())
for _ in range(5):
    th = rng.uniform(0.2, 2, (2, 3)); u = np.array([np.log(th[0, 0] * th[1, 2] / (th[0, 2] * th[1, 0])), np.log(th[0, 1] * th[1, 2] / (th[0, 2] * th[1, 1]))])
    full = np.array([np.sum(t * np.log(th)) + CF.hypergeom_weights(t) for _, t in Tb]); mono = np.log(c) + np.array(S) @ u
    ok &= np.allclose(full - mono, (full - mono)[0], atol=1e-8)
wex = [Fr(1, math.prod(math.factorial(int(x)) for x in np.ravel(t))) for _, t in Tb]; Z = sum(wex)
mean = (sum(w * s[0] for w, s in zip(wex, S)) / Z, sum(w * s[1] for w, s in zip(wex, S)) / Z); ok &= mean == (Fr(12), Fr(86, 5))
I0 = KB.fisher(S, c, (0.0, 0.0)); ok &= np.all(np.linalg.eigvalsh(I0) > 0)
lex = CF.leximin(ff, gf); psi, ulex, q = KB.legendre(S, c, 100 * lex[0, :2]); ok &= np.allclose(q @ np.array(S, float), 100 * lex[0, :2], atol=1e-6)
check("CF3 Tier A = Fisher's noncentral hypergeometric family; u = log odds ratios; mean at u = 0 is exactly (12, 86/5) = 100 f(x)g; Birch at the leximin", ok,
      f"Fisher at 0 = {np.round(I0, 3).tolist()}; odds ratios at the leximin mean = {np.round(np.exp(ulex), 4).tolist()}")

# CF4
def gen_config(fv, gv):
    n, m = len(fv), len(gv); d = (n - 1) * (m - 1); idx = {(i, j): i * (m - 1) + j for i in range(n - 1) for j in range(m - 1)}; v = {}; lam = {}
    for i in range(n - 1):
        for j in range(m - 1): e = np.zeros(d); e[idx[i, j]] = 1; v[i, j] = e; lam[i, j] = 0.0
        e = np.zeros(d); e[[idx[i, j] for j in range(m - 1)]] = -1; v[i, m - 1] = e; lam[i, m - 1] = -fv[i]
    for j in range(m - 1): e = np.zeros(d); e[[idx[i, j] for i in range(n - 1)]] = -1; v[n - 1, j] = e; lam[n - 1, j] = -gv[j]
    v[n - 1, m - 1] = np.ones(d); lam[n - 1, m - 1] = sum(fv[:n - 1]) - gv[m - 1]
    return v, lam
ok = True; nc = 0
for n, m in ((2, 3), (3, 3), (2, 4), (3, 4)):
    for _ in range(300):
        fv = rng.dirichlet(np.ones(n) * rng.uniform(0.3, 3)); gv = rng.dirichlet(np.ones(m) * rng.uniform(0.3, 3))
        if min(fv.min(), gv.min()) < 0.01: continue                       # keep the margins well conditioned (and nondegenerate)
        F = CF.facet_cells(fv, gv)
        h1, _ = CF.maxmin_lp(fv, gv, F); h0, bind = CF.maxmin_closed(list(fv), list(gv)); ok &= abs(h1 - h0) < 1e-9; nc += 1
        for kind, i in bind: ok &= all(((i * m + j) if kind == "row" else (r * m + i)) in F for j in range(m) for r in range(n))
for n, m in ((2, 3), (3, 3)):
    for _ in range(40):
        fv = rng.dirichlet(np.ones(n) * 2); gv = rng.dirichlet(np.ones(m) * 2); F = CF.facet_cells(fv, gv); h0, bind = CF.maxmin_closed(list(fv), list(gv))
        if len(bind) != 1: continue
        vd, ld = gen_config(list(fv), list(gv)); cells = [(c // m, c % m) for c in F]
        dim = DW.argmax_dim([tuple(vd[c]) for c in cells], [ld[c] for c in cells])
        ok &= dim == ((n - 2) * (m - 1) if bind[0][0] == "row" else (n - 1) * (m - 2))
h, bind = CF.maxmin_closed(f, g); ok &= h == Fr(2, 15) and bind == [("row", 0)]
ok &= np.allclose(lex[0], [2 / 15] * 3) and -(Fr(1, 3) * (L[0] + L[1] + L[2])) == Fr(2, 15)        # the balanced law (1/3,1/3,1/3) on row 1 certifies h* <= 2/15
check("CF4 h* = min(min f_i/m, min g_j/n) = facet LP (random margins); binding cells are facets; candidate dimensions (n-2)(m-1), (n-1)(m-2); example h* = 2/15",
      ok, f"{nc} random well-conditioned margins")

# CF5
ok = True; rows = []; prev = None
for tau in ("1/10", "1/100"):
    pts, D = CF.critical_points(f, g, tau, dps=300); T_ = mp.mpf(Fr(tau).numerator) / Fr(tau).denominator; ok &= len(pts) == 5 and D == 100
    info = []
    for y1, y2 in pts:
        u = np.array([float(mp.log(abs(y1)) / (D * mp.log(T_))), float(mp.log(abs(y2)) / (D * mp.log(T_)))])
        real = abs(mp.im(y1)) < abs(y1) * 1e-30 and abs(mp.im(y2)) < abs(y2) * 1e-30
        sg = (int(np.sign(float(mp.re(y1)))), int(np.sign(float(mp.re(y2))))) if real else None; info.append((u, sg))
    lexp = [x for x in info if np.allclose(x[0], [2 / 15] * 2, atol=3e-3)]; ok &= len(lexp) == 3 and sum(x[1] == (1, 1) for x in lexp) == 1
    nn = [x for x in info if x[1] == (-1, -1)]; ok &= len(nn) == 1 and np.allclose(nn[0][0], [0.13, 0.13], atol=2e-3)
    pm = [x for x in info if x[1] == (1, -1)]; ok &= len(pm) == 1 and np.allclose(pm[0][0], [0.2, 0.1], atol=1e-6)
    err = max(np.abs(x[0] - [2 / 15] * 2).max() for x in lexp) + np.abs(nn[0][0] - [0.13, 0.13]).max()
    if prev is not None: ok &= err < prev
    prev = err; rows.append((tau, f"{err:.1e}"))
vf = [V[k] for k in (0, 1, 2, 3, 5)]; lf = [float(L[k]) for k in (0, 1, 2, 3, 5)]
for u, K in (((0.13, 0.13), (0, 1, 4)), ((0.2, 0.1), (1, 2, 3))):
    ok &= KB.ties(vf, lf, u, tol=1e-9) == K and not KB.balanced(vf, K) and KB.signed_balanced(vf, K)
check("CF5 five critical points: three over the leximin (one positive), real (-,-) over (13/100,13/100), real (+,-) over (1/5,1/10); signed balance only", ok,
      f"distance to the limits {rows}")

# CF6
ok = True; done = 0
while done < 20:
    fv = sorted(rng.dirichlet(np.ones(2) * 2)); gv = list(rng.dirichlet(np.ones(3) * 2))
    fv = [round(x, 2) for x in fv]; fv[1] = round(1 - fv[0], 2); gv = [round(x, 2) for x in gv]; gv[2] = round(1 - gv[0] - gv[1], 2)
    if min(gv) <= 0.02 or min(fv) <= 0.02: continue
    ratios = sorted([x / 3 for x in fv] + [x / 2 for x in gv])
    if ratios[1] - ratios[0] < 0.02: continue
    F = CF.facet_cells(fv, gv); Vv, Ll = CF.cell_functions(fv[0], gv[0], gv[1]); vk = [Vv[k] for k in F]; lk = [Ll[k] for k in F]
    val = KB.valuation(vk, lk, [1e-60])[0]; lx = CF.leximin(fv, gv)[0, :2]; ok &= np.allclose(val, lx, atol=1e-2); done += 1
check("CF6 the positive critical point sits over the leximin coupling (20 random rational margins, T = 1e-60)", ok)

# CF7
ok = True; vk = [V[k] for k in (0, 1, 2, 3, 5)]; lk = [float(L[k]) for k in (0, 1, 2, 3, 5)]; box = ((-0.02, 0.32), (-0.02, 0.42)); er = {}
for h in (0.0, 0.1):
    me, Ce = KB.polygon_moments(vk, lk, h, (2 / 15, 2 / 15)); e = []
    for lt in (-200.0, -800.0):
        U1, U2, w = KB.tropical_grid(vk, lk, lt, h, box=box, n=1201); m = np.array([(w * U1).sum(), (w * U2).sum()]); d1, d2 = U1 - m[0], U2 - m[1]
        C = np.array([[(w * d1 * d1).sum(), (w * d1 * d2).sum()], [(w * d1 * d2).sum(), (w * d2 * d2).sum()]]); e.append(max(np.abs(m - me).max(), np.abs(C - Ce).max()))
    ok &= e[1] < e[0] and e[1] < 5e-4; er[h] = e
U1, U2, w = KB.tropical_grid(vk, lk, -800.0, 2 / 15, box=box, n=1201); ok &= w[(np.abs(U1 - 2 / 15) < 0.01) & (np.abs(U2 - 2 / 15) < 0.01)].sum() > 0.95
check("CF7 the thimble law: uniform on Q_h for h = 0, 1/10; concentrated at the leximin coupling at h*", ok, f"{ {k: [f'{x:.1e}' for x in v] for k, v in er.items()} }")

# CF8
ok = True; g2_ = (Fr(30, 100), Fr(44, 100), Fr(26, 100)); h2, b2 = CF.maxmin_closed(f, g2_); ok &= h2 == Fr(13, 100) and b2 == [("col", 2)]
lx2 = CF.leximin(ff, [float(x) for x in g2_]); ok &= np.allclose(lx2[0], [0.135, 0.135, 0.13])
F2 = CF.facet_cells(ff, [float(x) for x in g2_]); V2, L2 = CF.cell_functions(f[0], g2_[0], g2_[1]); v2 = [V2[k] for k in F2]; l2 = [float(L2[k]) for k in F2]
hh, _, K2 = DW.balance_lp(v2, l2); ok &= [CF.NAMES[F2[k]] for k in K2] == ["13", "23"] and DW.argmax_dim(v2, l2) == 1
ok &= Fr(27, 200) - Fr(2, 15) == Fr(1, 600) and g[1] - f[0] == Fr(3, 100)
dist = []
for tau in ("1/10", "1/1000"):
    pts, D = CF.critical_points(f, g2_, tau, dps=400); T_ = mp.mpf(Fr(tau).numerator) / Fr(tau).denominator
    U = [np.array([float(mp.log(abs(a)) / (D * mp.log(T_))), float(mp.log(abs(b)) / (D * mp.log(T_)))]) for a, b in pts]
    near = [u for u in U if np.abs(u - [0.135, 0.135]).max() < 5e-3]; ok &= len(pts) == 5 and len(near) == 4 and sum(np.allclose(u, [0.2, 0.1], atol=1e-4) for u in U) == 1
    dist.append(max(np.abs(u - [0.135, 0.135]).max() for u in near))
ok &= dist[1] < dist[0]
check("CF8 one count (g -> (30,44,26)/100): column 3 binds, candidates a segment; 1/600 from the leximin wall, 3/100 from f1 = g2; distinct valuations 3 -> 2",
      ok, f"distance of the four to (27/200, 27/200): {[f'{d:.1e}' for d in dist]}")
print(f"{sum(results)}/{len(results)} passed")
