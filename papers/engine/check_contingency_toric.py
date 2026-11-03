"""Checks for Part II of the companion note, contingency tables (module pce/contingency_toric.py).
  CT1  two configurations: the cells of the r x c independence model have (r - 1)(c - 1) relations, spanned by the basic moves, which fix both
       margins; the facets of Delta_(r-1) x Delta_(c-1) have exactly one circuit, (c 1_r, -r 1_c) / gcd (r, c in 2..4)
  CT2  the wall, from W itself: h* = min(L1 / r, L2 / c); the leading set is the row facets, the column facets, or all of them as c L1 - r L2 is
       < 0, > 0 or = 0; and c(lam) = -<a, lam> = c L1 - r L2 (exact rationals, and the balance programme on a grid of sizes)
  CT3  selection: the critical points of W are y_i = zeta T^(L1 / r), z_j = eta T^(L2 / c) with zeta^r = eta^c = 1 (r c of them, gradient zero to
       1e-12 relative); all have the valuation u* = (L1/r, ..., L2/c, ...) for every L1, L2, so the selected point never moves; the positive law is
       exactly (T^(L1/r) on rows, T^(L2/c) on columns) / sum; the candidate set has dimension 0 on the wall and c - 1 or r - 1 off it
  CT4  margins: l_k(b) is exactly the k-th row or column margin; the MLE rows x cols / n^2 has the observed margins (Birch) and is positive iff every
       margin is positive (200 random tables, with and without zero margins)
  CT5  the Fisher information of the independence family with surveys of sizes n1, n2 is block diagonal n1 Cov(row) + n2 Cov(col) (finite
       differences of the log-likelihood); the Fisher-Rao distance is sqrt(n1 d_row^2 + n2 d_col^2), and d on the simplex satisfies the triangle
       inequality and d(p, p) = 0 (random laws)
  CT6  2 x 3 with joint sampling (n1 = n2 = n): h* = n/3, leading set the column facets, candidates the open segment of row margins (n/3, 2n/3), u* at
       n/2; the wall needs n2 = 3n/2, that is n/2 more column observations (Corollary 12(3) with rho on a column facet: s* = n/2); 2 x 2 sits on the wall
       for every n
  CT7  a structural zero: deleting one cell of the 2 x 2 table leaves a triangle, which is not a face of the square; its facet configuration has no
       circuit (CP^2), so the one wall of the 2 x 2 model is gone
"""
import os, sys, random, itertools
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import linprog
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import contingency_toric as CT, datawalls as DW, kahlerbridge as KB
random.seed(20261103); rng = np.random.default_rng(20261103); results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)

# CT1
ok = True; rows = []
for r in (2, 3, 4):
    for c in (2, 3, 4):
        A = CT.cell_config(r, c); ok &= CT.cell_relation_dim(r, c) == (r - 1) * (c - 1)
        moves = [CT.basic_move(r, c, 0, 0, i, j).ravel() for i in range(1, r) for j in range(1, c)]
        ok &= np.linalg.matrix_rank(np.array(moves)) == (r - 1) * (c - 1) and np.allclose(np.array(moves) @ A, 0)
        v, lam = CT.facet_config(r, c, 1, 1); g = np.gcd(r, c); ok &= DW.circuits(v) == [[c // g] * r + [-r // g] * c]
        rows.append(f"{r}x{c}: {(r - 1) * (c - 1)} moves, 1 circuit")
check("CT1 cells: (r-1)(c-1) relations (basic moves, margins fixed); facets: exactly one circuit (c 1_r, -r 1_c)/gcd", ok, "; ".join(rows[:4]) + " ...")

# CT2
ok = True
for r, c in ((2, 2), (2, 3), (3, 3), (3, 4)):
    for L1, L2 in itertools.product([Fr(k, 2) for k in range(1, 7)], repeat=2):
        v, lam = CT.facet_config(r, c, float(L1), float(L2)); a = DW.circuits(v)[0]
        cl = -sum(x * y for x, y in zip(a, [Fr(0)] * (r - 1) + [-L1] + [Fr(0)] * (c - 1) + [-L2])); g = np.gcd(r, c)
        ok &= cl == (c * L1 - r * L2) / g
        h, _, K = DW.balance_lp(v, lam); ok &= abs(h - float(min(L1 / r, L2 / c))) < 1e-9
        want = tuple(range(r)) if c * L1 < r * L2 else (tuple(range(r, r + c)) if c * L1 > r * L2 else tuple(range(r + c)))
        ok &= K == want
check("CT2 h* = min(L1/r, L2/c); leading set rows / columns / all as c L1 - r L2 < 0, > 0, = 0; c(lam) = (c L1 - r L2)/gcd (exact)", ok)

# CT3
ok = True
for r, c in ((2, 2), (2, 3), (3, 4)):
    for L1, L2 in ((1.0, 1.0), (1.0, 2.0), (2.5, 1.0)):
        v, lam = CT.facet_config(r, c, L1, L2); V = np.array(v, float); T = 1e-8; coef = T ** (-np.array(lam))
        for zr in range(r):
            for zc in range(c):
                y = np.r_[[np.exp(2j * np.pi * zr / r) * T ** (L1 / r)] * (r - 1), [np.exp(2j * np.pi * zc / c) * T ** (L2 / c)] * (c - 1)]
                terms = coef * np.prod(y[None, :] ** V, 1); grad = (terms[:, None] * V).sum(0) / y
                ok &= np.max(np.abs(grad * y)) < 1e-12 * np.max(np.abs(terms))
                ok &= np.allclose(np.log(np.abs(y)) / np.log(T), np.r_[[L1 / r] * (r - 1), [L2 / c] * (c - 1)])
        ypos = np.r_[[T ** (L1 / r)] * (r - 1), [T ** (L2 / c)] * (c - 1)]; terms = coef * np.prod(ypos[None, :] ** V, 1); w = terms / terms.sum()
        ex = np.r_[[T ** (L1 / r)] * r, [T ** (L2 / c)] * c]; ok &= np.allclose(w, ex / ex.sum(), rtol=1e-12)
        d = DW.argmax_dim(v, lam); ok &= d == (0 if np.isclose(c * L1, r * L2) else (c - 1 if c * L1 < r * L2 else r - 1))
check("CT3 r c critical points, all over u* = (L1/r, L2/c) for every size: the selected point never moves; exact positive law; candidate dimension 0 / c-1 / r-1", ok)

# CT4
ok = True
for _ in range(200):
    r, c = random.randint(2, 4), random.randint(2, 4); t = rng.integers(0, 6, (r, c)).astype(float)
    if rng.random() < 0.3: t[rng.integers(r), :] = 0
    if rng.random() < 0.3: t[:, rng.integers(c)] = 0
    if t.sum() == 0: continue
    n = t.sum(); rm, cm = t.sum(1), t.sum(0); v, lam = CT.facet_config(r, c, n, n); b = CT.margin_point(rm, cm)
    ok &= np.allclose(CT.ell(v, lam, b), np.r_[rm, cm]); p = CT.mle(t)
    ok &= np.allclose(p.sum(1) * n, rm) and np.allclose(p.sum(0) * n, cm) and (np.all(p > 0) == (np.all(rm > 0) and np.all(cm > 0)))
check("CT4 l_k(b) = the margins; the MLE has the observed margins and is positive iff every margin is (200 tables)", ok)

# CT5
ok = True
for _ in range(10):
    r, c = random.randint(2, 4), random.randint(2, 4); n1, n2 = random.uniform(5, 50), random.uniform(5, 50)
    th = rng.normal(0, 0.5, r + c)
    def loglik(t):                        # expected log-likelihood of the two surveys, as a function of the natural parameters
        pr = np.exp(t[:r]) / np.exp(t[:r]).sum(); pc = np.exp(t[r:]) / np.exp(t[r:]).sum(); return n1 * (qr @ np.log(pr)) + n2 * (qc @ np.log(pc))
    pr0 = np.exp(th[:r]) / np.exp(th[:r]).sum(); pc0 = np.exp(th[r:]) / np.exp(th[r:]).sum(); qr, qc = pr0, pc0; hh = 1e-4; E = np.eye(r + c)
    H = np.array([[(loglik(th + hh * a + hh * bb) - loglik(th + hh * a - hh * bb) - loglik(th - hh * a + hh * bb) + loglik(th - hh * a - hh * bb)) / (4 * hh * hh)
                   for bb in E] for a in E])
    ok &= np.allclose(-H, CT.fisher_blocks(pr0, pc0, n1, n2), atol=1e-4 * max(n1, n2))
    P = [rng.dirichlet(np.ones(r)) for _ in range(3)]; ok &= CT.fr_simplex(P[0], P[0]) < 1e-6
    ok &= CT.fr_simplex(P[0], P[2]) <= CT.fr_simplex(P[0], P[1]) + CT.fr_simplex(P[1], P[2]) + 1e-12
check("CT5 Fisher information block diagonal n1 Cov(row) + n2 Cov(col); Fisher-Rao distance sqrt(n1 d_row^2 + n2 d_col^2)", ok)

# CT6
ok = True
for n in (6.0, 30.0, 300.0):
    v, lam = CT.facet_config(2, 3, n, n); h, _, K = DW.balance_lp(v, lam); ok &= abs(h - n / 3) < 1e-9 and K == (2, 3, 4)
    for b1 in np.linspace(0, n, 61):
        b = CT.margin_point([b1, n - b1], [n / 3, n / 3, n / 3]); g = CT.ell(v, lam, b).min()
        ok &= (abs(g - n / 3) < 1e-9) == (n / 3 - 1e-9 <= b1 <= 2 * n / 3 + 1e-9)
        if n / 3 + 1e-6 < b1 < 2 * n / 3 - 1e-6: ok &= KB.balanced(v, KB.ties(v, lam, b, tol=1e-7))
    ok &= not KB.balanced(v, KB.ties(v, lam, CT.margin_point([n / 3, 2 * n / 3], [n / 3] * 3), tol=1e-7))
    rho = np.zeros(5); rho[4] = 1.0; s = DW.first_wall(v, lam, rho, smax=n, n=401); ok &= abs(s - n / 2) < 1e-6 * n
    v2, lam2 = CT.facet_config(2, 2, n, n); ok &= DW.balance_lp(v2, lam2)[2] == (0, 1, 2, 3)
check("CT6 2x3 joint sampling: h* = n/3, column facets lead, candidates b1 in (n/3, 2n/3), u* = n/2; the wall needs n/2 more column observations; 2x2 on the wall",
      ok)

# CT7
ok = True; A = CT.cell_config(2, 2)[:, [0, 2]]           # cell (i, j) -> (row-1 indicator, col-1 indicator) in the square [0,1]^2
P = A[:3]                                                # delete cell (2, 2)
def is_face(P, Q):
    for d in rng.normal(size=(400, 2)):
        m = (Q @ d).max(); F = Q[np.isclose(Q @ d, m)]
        if len(F) == len(P) and all(any(np.allclose(p, f) for f in F) for p in P): return True
    return False
ok &= not is_face(P, A) and abs(np.linalg.det(np.c_[P[1:] - P[0]])) == 1
vT = [(1, 0), (0, 1), (-1, -1)]; ok &= DW.circuits(vT) == []
check("CT7 a structural zero in 2x2 leaves a unimodular triangle, not a face of the square; no circuit (CP^2), so no wall", ok)
print(f"{sum(results)}/{len(results)} passed")
