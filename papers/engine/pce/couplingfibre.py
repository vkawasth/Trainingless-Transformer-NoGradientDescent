"""Section 11 of the companion note: one coupling polytope through every layer. New module; it imports kahlerbridge and datawalls and
modifies neither. The running example is that of the architecture note, f = (2/5, 3/5), g = (3/10, 43/100, 27/100).

The fibre of the marginal map over (f, g) is the coupling polytope Cpl(f, g). For 2 x 3 write a coupling by x = a11, y = a12; then
  a13 = f1 - x - y,  a21 = g1 - x,  a22 = g2 - y,  a23 = x + y - (f1 + g1 + g2 - 1),
so cell c has facet function l_c = <v_c, (x, y)> - lam_c with the normals V6 below. The cells are the facets of the fibre.

  EX_F, EX_G                          the running example (Fractions)
  cell_functions(f1, g1, g2)          (V6, LAM6) for the six cells of a 2 x 3 coupling, in the coordinates (x, y)
  facet_cells(f, g)                   the cells whose zero set is a facet of Cpl(f, g) (linear programming; any n x m)
  polygon(f, g)                       exact vertices (Fractions) of the 2 x 3 coupling polygon, counter-clockwise, and its exact area
  maxmin_closed(f, g)                 h* = min(min_i f_i / m, min_j g_j / n) and the binding rows and columns (any n x m)
  maxmin_lp(f, g, cells)              max over couplings of the least cell among `cells` (any n x m)
  leximin(f, g)                       the leximin coupling (iterated max-min, fixing cells that are tight at every optimum)
  tables(N, f, g)                     the integer tables with margins N f, N g (2 x 3), as (t11, t12) and the full table
  hypergeom_weights(T)                log c_t = -sum log t_ij!, Fisher's (noncentral) hypergeometric base weights
  critical_points(f, g, tau, dps)     all critical points of W = sum_c T^(-lam_c) y^(v_c) over the facets, T = tau^D (D a common denominator),
                                      by an exact resultant; returns (y1, y2) as mpmath numbers
"""
from __future__ import annotations
import itertools, math
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import linprog

EX_F = (Fr(2, 5), Fr(3, 5)); EX_G = (Fr(3, 10), Fr(43, 100), Fr(27, 100))
V6 = [(1, 0), (0, 1), (-1, -1), (-1, 0), (0, -1), (1, 1)]; NAMES = ["11", "12", "13", "21", "22", "23"]


def cell_functions(f1, g1, g2):
    return V6, [0, 0, -f1, -g1, -g2, f1 + g1 + g2 - 1]


def _margin_rows(f, g):
    n, m = len(f), len(g); N = n * m; A = []; b = []
    for i in range(n):
        r = np.zeros(N); r[i * m:(i + 1) * m] = 1; A.append(r); b.append(float(f[i]))
    for j in range(m - 1):
        r = np.zeros(N); r[j:N:m] = 1; A.append(r); b.append(float(g[j]))
    return np.array(A), np.array(b)


def facet_cells(f, g, eps=1e-9):
    """c is a facet iff some coupling has a_c = 0 and every other cell > 0 (then {a_c = 0} has codimension one)"""
    n, m = len(f), len(g); N = n * m; A, b = _margin_rows(f, g); out = []
    for c in range(N):
        e = np.zeros(N); e[c] = 1
        Aub = [np.r_[-np.eye(N)[d], 1.0] for d in range(N) if d != c]
        r = linprog(np.r_[np.zeros(N), -1.0], A_ub=np.array(Aub), b_ub=np.zeros(N - 1), A_eq=np.r_[np.c_[A, np.zeros(len(A))], [np.r_[e, 0.0]]],
                    b_eq=np.r_[b, 0.0], bounds=[(None, None)] * N + [(None, 1)], method="highs")
        if r.status == 0 and -r.fun > eps: out.append(c)
    return out


def polygon(f, g):
    f1, g1, g2 = Fr(f[0]), Fr(g[0]), Fr(g[1]); V, L = cell_functions(f1, g1, g2); P = []
    for a, b in itertools.combinations(range(6), 2):
        (p, q), (r, s) = V[a], V[b]; d = p * s - q * r
        if d == 0: continue
        x = (L[a] * s - q * L[b]) / d; y = (p * L[b] - L[a] * r) / d
        if all(vx * x + vy * y - l >= 0 for (vx, vy), l in zip(V, L)) and (x, y) not in P: P.append((x, y))
    cx = sum(p[0] for p in P) / len(P); cy = sum(p[1] for p in P) / len(P)
    P.sort(key=lambda p: math.atan2(float(p[1] - cy), float(p[0] - cx)))
    A = sum(P[i][0] * P[(i + 1) % len(P)][1] - P[(i + 1) % len(P)][0] * P[i][1] for i in range(len(P))) / 2
    return P, A


def maxmin_closed(f, g):
    n, m = len(f), len(g); vals = [(Fr(fi) / m if isinstance(fi, Fr) else fi / m, "row", i) for i, fi in enumerate(f)] + \
                              [(Fr(gj) / n if isinstance(gj, Fr) else gj / n, "col", j) for j, gj in enumerate(g)]
    h = min(v[0] for v in vals); return h, [(k, i) for v, k, i in vals if v == h]


def maxmin_lp(f, g, cells=None):
    n, m = len(f), len(g); N = n * m; A, b = _margin_rows(f, g); cells = range(N) if cells is None else cells
    Aub = [np.r_[-np.eye(N)[c], 1.0] for c in cells]
    r = linprog(np.r_[np.zeros(N), -1.0], A_ub=np.array(Aub), b_ub=np.zeros(len(Aub)), A_eq=np.c_[A, np.zeros(len(A))], b_eq=b,
                bounds=[(0, None)] * N + [(None, None)], method="highs")
    return -r.fun, r.x[:N]


def leximin(f, g, tol=1e-9):
    n, m = len(f), len(g); N = n * m; A, b = _margin_rows(f, g); fixed = {}; free = set(range(N))
    while free:
        Ae = [np.r_[row, 0.0] for row in A]; be = list(b)
        for c, val in fixed.items(): Ae.append(np.r_[np.eye(N)[c], 0.0]); be.append(val)
        Aub = [np.r_[-np.eye(N)[c], 1.0] for c in free]
        r = linprog(np.r_[np.zeros(N), -1.0], A_ub=np.array(Aub), b_ub=np.zeros(len(Aub)), A_eq=np.array(Ae), b_eq=np.array(be),
                    bounds=[(0, None)] * N + [(None, None)], method="highs"); t = -r.fun
        tight = []
        for c in free:                     # c is tight at every optimum iff max a_c subject to the others >= t equals t
            Aub2 = [np.r_[-np.eye(N)[d], 0.0] for d in free if d != c]; bub2 = [-t] * len(Aub2)
            r2 = linprog(np.r_[-np.eye(N)[c], 0.0], A_ub=np.array(Aub2) if Aub2 else None, b_ub=np.array(bub2) if Aub2 else None, A_eq=np.array(Ae),
                         b_eq=np.array(be), bounds=[(0, None)] * N + [(0, 0)], method="highs")
            if -r2.fun <= t + tol: tight.append(c)
        if not tight: tight = list(free)
        for c in tight: fixed[c] = t; free.discard(c)
    return np.array([fixed[c] for c in range(N)]).reshape(n, m)


def tables(N, f, g):
    R = [int(N * x) for x in f]; C = [int(N * x) for x in g]; out = []
    for t11 in range(0, min(R[0], C[0]) + 1):
        for t12 in range(0, min(R[0] - t11, C[1]) + 1):
            t13 = R[0] - t11 - t12; t21 = C[0] - t11; t22 = C[1] - t12; t23 = C[2] - t13
            if min(t13, t21, t22, t23) >= 0: out.append(((t11, t12), np.array([[t11, t12, t13], [t21, t22, t23]])))
    return out


def hypergeom_weights(T): return -sum(math.lgamma(int(x) + 1) for x in np.ravel(T))


def critical_points(f, g, tau, dps=400):
    import sympy as sp, mpmath as mp
    f1, g1, g2 = Fr(f[0]), Fr(g[0]), Fr(g[1]); V, L = cell_functions(f1, g1, g2); F = facet_cells([float(f1), 1 - float(f1)], [float(g1), float(g2), float(1 - g1 - g2)])
    D = 1
    for k in F: D = D * Fr(L[k]).denominator // math.gcd(D, Fr(L[k]).denominator)
    t = sp.Rational(tau); y1, y2 = sp.symbols("y1 y2")
    W = sum(t ** int(-Fr(L[k]) * D) * y1 ** V[k][0] * y2 ** V[k][1] for k in F)
    E1 = sp.numer(sp.together(sp.expand(y1 * sp.diff(W, y1)))); E2 = sp.numer(sp.together(sp.expand(y2 * sp.diff(W, y2))))
    R = sp.Poly(sp.resultant(E1, E2, y1), y2); mp.mp.dps = dps
    coeffs = [mp.mpf(sp.Rational(c).p) / mp.mpf(sp.Rational(c).q) for c in R.all_coeffs()]
    while coeffs and coeffs[-1] == 0: coeffs.pop()
    roots = mp.polyroots(coeffs, maxsteps=800, extraprec=4 * dps); out = []
    P1 = sp.Poly(E1, y1); P2 = sp.Poly(E2, y1)
    for r2 in roots:
        if abs(r2) < mp.mpf(10) ** (-dps // 2): continue
        c2 = [complex(0)] and [mp.mpc(complex(sp.N(cf.subs(y2, 0), 5))) for cf in []]
        q2 = [sp.lambdify(y2, cf, "mpmath")(r2) for cf in P2.all_coeffs()]; cand = mp.polyroots(q2, maxsteps=400, extraprec=4 * dps)
        q1 = [sp.lambdify(y2, cf, "mpmath")(r2) for cf in P1.all_coeffs()]
        best = min(cand, key=lambda z: abs(mp.polyval(q1, z)) / (1 + abs(z)) ** len(q1)); out.append((best, r2))
    uniq = []
    for p in out:
        if all(abs(p[0] - q[0]) > abs(p[0]) * mp.mpf(10) ** (-dps // 4) or abs(p[1] - q[1]) > abs(p[1]) * mp.mpf(10) ** (-dps // 4) for q in uniq): uniq.append(p)
    return uniq, D
