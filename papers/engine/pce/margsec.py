"""Sections of the marginal projection, and the 4-cycle (CHSH) cover (companion note, sections 12.15-12.18). New module; it modifies
nothing and uses pce/selcomp.py and pce/selnat.py.

The 4-cycle cover: binary variables A0, A1, B0, B1 (a joint outcome is a tuple (a0, a1, b0, b1)), contexts (A0,B0), (A0,B1), (A1,B0), (A1,B1),
i.e. the cycle A0 - B0 - A1 - B1 - A0. Data coordinates (8): the 00 entries P(Ai = 0, Bj = 0) for (i, j) = 00, 01, 10, 11, then P(A0 = 0),
P(A1 = 0), P(B0 = 0), P(B1 = 0). Every data vector gives four tables that agree on the single marginals.
  CELLS, M                   the 16 joint outcomes and the 8 x 16 projection matrix pi
  local_ineqs()              the 16 inequalities of the local polytope L (all four tables nonnegative)
  facets_D()                 the facets of D = pi(J), exactly: hyperplanes through affinely independent images of vertices, one-sided
  faces_D(F)                 all faces of D (vertex sets), closed under intersection
  fibre_dim_face(V)          |V| - 1 - dim G  (dimension of the fibre over the relative interior of the face G with vertex set V)
  fibre(d)                   A, b of {x >= 0 : M x = d, sum x = 1}
  fibre_dim(d), cell_width(d), interval(d, w)   numerical: dimension (support + exact rank), largest range of a cell, range of w.x
  maxent(d)                  maximum-entropy point of the fibre (iterative proportional fitting from the uniform law)
  leximin_general(A, b)      facet-leximin point of {x >= 0, A x = b} (the recursion of selcomp.leximin, for any constraint system), exact
  table(d)                   the four 2 x 2 tables of a data vector
"""
from __future__ import annotations
import itertools
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import linprog
from . import selcomp as S

CELLS = list(itertools.product((0, 1), repeat=4))            # (a0, a1, b0, b1)
CTX = [(0, 2), (0, 3), (1, 2), (1, 3)]                       # (A0,B0), (A0,B1), (A1,B0), (A1,B1) as index pairs into the tuple
def _row(f): return [1 if f(c) else 0 for c in CELLS]
M = [_row(lambda c, i=i, j=j: c[i] == 0 and c[j] == 0) for (i, j) in CTX] + [_row(lambda c, k=k: c[k] == 0) for k in range(4)]
VERTS = [[Fr(M[r][k]) for r in range(8)] for k in range(16)]   # pi of the 16 deterministic joint laws


def table(d):
    """the four tables [[P(00), P(01)], [P(10), P(11)]] of contexts (Ai, Bj); singles d[4..7] = P(A0=0), P(A1=0), P(B0=0), P(B1=0)"""
    out = []
    for k, (i, j) in enumerate(CTX):
        a, b, t = d[4 + i], d[6 + (j - 2)], d[k]
        out.append([[t, a - t], [b - t, 1 - a - b + t]])
    return out


def local_ineqs():
    """rows (c, c0) meaning c . d + c0 >= 0: the 16 entries of the four tables"""
    out = []
    for k, (i, j) in enumerate(CTX):
        ia, ib = 4 + i, 6 + (j - 2)
        for sa in (0, 1):
            for sb in (0, 1):
                c = [Fr(0)] * 8; c0 = Fr(0)
                # entry P(Ai = sa, Bj = sb) = t, a - t, b - t, 1 - a - b + t
                c[k] = Fr(1) if sa == sb else Fr(-1)
                if (sa, sb) == (0, 1): c[ia] += 1
                if (sa, sb) == (1, 0): c[ib] += 1
                if (sa, sb) == (1, 1): c[ia] -= 1; c[ib] -= 1; c0 = Fr(1)
                out.append((tuple(c), c0))
    return out


def _affine_rank(P):
    if not P: return -1
    return S._rank([[x - y for x, y in zip(p, P[0])] for p in P[1:]] or [[0] * len(P[0])])


def _normal(sub):
    """the hyperplane c.d + c0 = 0 through the points of sub (affinely independent, 8 of them in R^8), primitive integer coefficients"""
    rows = [list(p) + [Fr(1)] for p in sub]
    # nullspace of the 8 x 9 matrix
    Mx = [r[:] for r in rows]; n = 9; piv = []; rk = 0
    for col in range(n):
        p = next((i for i in range(rk, len(Mx)) if Mx[i][col] != 0), None)
        if p is None: continue
        Mx[rk], Mx[p] = Mx[p], Mx[rk]; f = Mx[rk][col]; Mx[rk] = [x / f for x in Mx[rk]]
        for i in range(len(Mx)):
            if i != rk and Mx[i][col] != 0:
                g = Mx[i][col]; Mx[i] = [a - g * b for a, b in zip(Mx[i], Mx[rk])]
        piv.append(col); rk += 1
    free = [c for c in range(n) if c not in piv]
    if len(free) != 1: return None
    v = [Fr(0)] * n; v[free[0]] = Fr(1)
    for i, col in enumerate(piv): v[col] = -Mx[i][free[0]]
    from math import gcd
    den = 1
    for x in v: den = den * x.denominator // gcd(den, x.denominator)
    w = [int(x * den) for x in v]; g = 0
    for x in w: g = gcd(g, abs(x))
    return tuple(x // g for x in w)


def facets_D():
    """exact facet list of D: each is (c, c0, tight vertex indices) with c.d + c0 >= 0 on D"""
    found = {}
    for sub in itertools.combinations(range(16), 8):
        P = [VERTS[k] for k in sub]
        if _affine_rank(P) != 7: continue
        nv = _normal(P)
        if nv is None: continue
        vals = [sum(Fr(nv[r]) * VERTS[k][r] for r in range(8)) + nv[8] for k in range(16)]
        if all(v >= 0 for v in vals): sgn = 1
        elif all(v <= 0 for v in vals): sgn = -1
        else: continue
        key = tuple(sgn * x for x in nv)
        if key not in found: found[key] = tuple(k for k in range(16) if vals[k] == 0)
    return [(k[:8], k[8], v) for k, v in found.items()]


def faces_D(F):
    faces = {frozenset(range(16))}; frontier = [frozenset(f[2]) for f in F]
    for f in frontier: faces.add(f)
    changed = True
    while changed:
        changed = False; new = set()
        for a in faces:
            for f in frontier:
                c = a & f
                if c and c not in faces: new.add(c)
        if new: faces |= new; changed = True
    return faces


def face_dim(V): return _affine_rank([VERTS[k] for k in V])
def fibre_dim_face(V): return len(V) - 1 - face_dim(V)


def fibre(d):
    A = [r[:] for r in M] + [[1] * 16]; b = [Fr(x) for x in d] + [Fr(1)]; return A, b


def fibre_dim(d, tol=1e-9):
    A, b = fibre(d); Af = np.array(A, float); bf = np.array([float(x) for x in b])
    sup = S._support(Af, bf)
    if sup is None: return None
    zero = [k for k in range(16) if k not in sup]
    return 16 - S._rank(A + [[1 if k == z else 0 for k in range(16)] for z in zero])


def _lp(c, d, sense=1):
    A, b = fibre(d); r = linprog(sense * np.array(c, float), A_eq=np.array(A, float), b_eq=np.array([float(x) for x in b]), bounds=[(0, None)] * 16, method="highs")
    return None if r.status != 0 else sense * r.fun


def cell_width(d):
    w = 0.0
    for k in range(16):
        e = [0] * 16; e[k] = 1; lo = _lp(e, d); hi = _lp(e, d, -1)
        if lo is None: return None
        w = max(w, hi - lo)
    return w


def interval(d, w):
    lo = _lp(w, d); hi = _lp(w, d, -1); return (lo, hi) if lo is not None else None


def maxent(d, iters=20000, tol=1e-14):
    """iterative proportional fitting of the uniform law to the four pair tables (the I-projection of uniform = maximum entropy)"""
    T = [[[float(x) for x in r] for r in t] for t in table(d)]
    x = np.full(16, 1 / 16)
    idx = np.array(CELLS)
    for it in range(iters):
        err = 0.0
        for k, (i, j) in enumerate(CTX):
            for sa in (0, 1):
                for sb in (0, 1):
                    m = (idx[:, i] == sa) & (idx[:, j] == sb); cur = x[m].sum(); tgt = T[k][sa][sb]
                    err = max(err, abs(cur - tgt))
                    if cur > 0: x[m] *= tgt / cur
        if err < tol: break
    return x


def leximin_general(A, b, tol=1e-10):
    """the facet-leximin point of {x >= 0, A x = b}: selcomp.leximin's recursion with the constraint system given directly"""
    N = len(A[0]); Af = np.array(A, float); bf = np.array([float(x) for x in b])
    sup = S._support(Af, bf); zero0 = [d for d in range(N) if d not in sup]
    dim = N - S._rank(A + [[1 if k == d else 0 for k in range(N)] for d in zero0]); F = []
    for c in sup:
        sc = S._support(Af, bf, [c])
        if sc is None: continue
        z = [d for d in range(N) if d not in sc]
        if N - S._rank(A + [[1 if k == d else 0 for k in range(N)] for d in z]) == dim - 1: F.append(c)
    zero = [c for c in range(N) if c not in F and -linprog(-np.eye(N)[c], A_eq=Af, b_eq=bf, bounds=[(0, None)] * N, method="highs").fun <= tol]
    fixed = {}; free = set(F); levels = []
    while free:
        Ae = [np.r_[r, 0.0] for r in Af] + [np.r_[np.eye(N)[c], 0.0] for c in fixed]; be = list(bf) + [float(v) for v in fixed.values()]
        r = linprog(np.r_[np.zeros(N), -1.0], A_ub=[np.r_[-np.eye(N)[c], 1.0] for c in free], b_ub=np.zeros(len(free)), A_eq=Ae, b_eq=be,
                    bounds=[(0, None)] * N + [(None, None)], method="highs"); t = -r.fun; tight = []
        for c in free:
            oth = [d for d in free if d != c]
            r2 = linprog(-np.r_[np.eye(N)[c], 0.0], A_ub=[np.r_[-np.eye(N)[d], 0.0] for d in oth] or None, b_ub=[-t] * len(oth) or None, A_eq=Ae, b_eq=be,
                         bounds=[(0, None)] * N + [(0, 0)], method="highs")
            if r2.status == 0 and -r2.fun <= t + tol: tight.append(c)
        if not tight: tight = sorted(free)
        tq = Fr(t).limit_denominator(10 ** 7)
        for c in tight: fixed[c] = tq; free.discard(c)
        levels.append((tq, tuple(tight)))
    nlev = len(levels); Aext = [list(r) + [0] * nlev for r in A]; rhs = list(b)
    for c in zero: row = [0] * (N + nlev); row[c] = 1; Aext.append(row); rhs.append(Fr(0))
    for j, (_, Sx) in enumerate(levels):
        for c in Sx: row = [0] * (N + nlev); row[c] = 1; row[N + j] = -1; Aext.append(row); rhs.append(Fr(0))
    sol = S._solve_exact(Aext, rhs); x = sol[:N]
    if min(x) < 0: raise ValueError("negative cell in exact re-solve")
    return x, levels


def leximin(d):
    A, b = fibre(d); return leximin_general(A, b)[0]


# the symmetric CHSH path: uniform singles, correlations E = lam on three contexts and -lam on (A1, B1); S = 4 lam
def chsh_path(lam):
    lam = Fr(lam); t = (1 + lam) / 4; u = (1 - lam) / 4
    return [t, t, t, u, Fr(1, 2), Fr(1, 2), Fr(1, 2), Fr(1, 2)]


W_A0A1 = [1 if (c[0] == 0 and c[1] == 0) else 0 for c in CELLS]     # P(A0 = 0, A1 = 0): never observed together


# ------------------------------------------------------------------ the n-cycle cover (section 12.19)
def ncycle(n):
    """binary X_0..X_{n-1}, contexts {X_i, X_{i+1 mod n}}. Data: P(X_i = 0, X_{i+1} = 0) for i = 0..n-1, then P(X_i = 0). Returns cells, M, verts."""
    cells = list(itertools.product((0, 1), repeat=n))
    M_ = [[1 if (c[i] == 0 and c[(i + 1) % n] == 0) else 0 for c in cells] for i in range(n)] + [[1 if c[i] == 0 else 0 for c in cells] for i in range(n)]
    verts = [[Fr(M_[r][k]) for r in range(2 * n)] for k in range(len(cells))]
    return cells, M_, verts


def ncycle_local(n):
    out = []
    for i in range(n):
        ia, ib = n + i, n + (i + 1) % n
        for sa in (0, 1):
            for sb in (0, 1):
                c = [Fr(0)] * (2 * n); c0 = Fr(0); c[i] = Fr(1) if sa == sb else Fr(-1)
                if (sa, sb) == (0, 1): c[ia] += 1
                if (sa, sb) == (1, 0): c[ib] += 1
                if (sa, sb) == (1, 1): c[ia] -= 1; c[ib] -= 1; c0 = Fr(1)
                out.append((tuple(c), c0))
    return out


def facets_hull(verts):
    """exact facets of conv(verts) (full-dimensional in R^m): Qhull proposes, each proposal is re-derived exactly from its tight
    vertices (affine rank m - 1, one-sided) and merged by its primitive normal"""
    from scipy.spatial import ConvexHull
    from math import gcd
    m = len(verts[0]); P = np.array([[float(x) for x in v] for v in verts]); H = ConvexHull(P)
    found = {}
    for eq_ in H.equations:
        vals = P @ eq_[:-1] + eq_[-1]; tight = [k for k in range(len(verts)) if abs(vals[k]) < 1e-9]
        Pt = [verts[k] for k in tight]
        if _affine_rank(Pt) != m - 1: continue
        # exact normal: nullspace of [v, 1] rows of the tight vertices
        rows = [list(v) + [Fr(1)] for v in Pt]; Mx = [r[:] for r in rows]; nn = m + 1; piv = []; rk = 0
        for col in range(nn):
            p = next((i for i in range(rk, len(Mx)) if Mx[i][col] != 0), None)
            if p is None: continue
            Mx[rk], Mx[p] = Mx[p], Mx[rk]; f = Mx[rk][col]; Mx[rk] = [x / f for x in Mx[rk]]
            for i in range(len(Mx)):
                if i != rk and Mx[i][col] != 0:
                    g = Mx[i][col]; Mx[i] = [a - g * b for a, b in zip(Mx[i], Mx[rk])]
            piv.append(col); rk += 1
        free = [c for c in range(nn) if c not in piv]
        if len(free) != 1: continue
        v = [Fr(0)] * nn; v[free[0]] = Fr(1)
        for i, col in enumerate(piv): v[col] = -Mx[i][free[0]]
        den = 1
        for x in v: den = den * x.denominator // gcd(den, x.denominator)
        w = [int(x * den) for x in v]; g = 0
        for x in w: g = gcd(g, abs(x))
        w = [x // g for x in w]
        ev = [sum(Fr(w[r]) * verts[k][r] for r in range(m)) + w[m] for k in range(len(verts))]
        if all(x >= 0 for x in ev): key = tuple(w)
        elif all(x <= 0 for x in ev): key = tuple(-x for x in w)
        else: continue
        found[key] = tuple(k for k in range(len(verts)) if ev[k] == 0)
    return [(k[:m], k[m], t) for k, t in found.items()]


# ------------------------------------------------------------------ the power family of sections (section 12.15, remark on three layers)
def phi_q(x, q):
    """concave cell functional: (x^q - x)/(q(1 - q)); q = 1 gives -x log x (Shannon), q = 0 gives log x (Burg), up to terms that are
    constant on a fibre; for q -> -infinity its maximiser tends to the leximin point of the cells (Cressie-Read power family)"""
    import mpmath as mp
    if q == 1: return -x * mp.log(x)
    if q == 0: return mp.log(x)
    return (x ** q - x) / (q * (1 - q))


def power_section(A, b, q, x0=None, dps=40, iters=400):
    """argmax of sum_c phi_q(x_c) over {x >= 0, A x = b} (cells forced to zero are dropped): damped Newton in null-space coordinates,
    multiprecision. Returns the point as a list of mpf (zeros on forced-zero cells)."""
    import mpmath as mp, scipy.linalg as sl
    mp.mp.dps = dps
    N = len(A[0]); Af = np.array(A, float); bf = np.array([float(v) for v in b])
    sup = S._support(Af, bf)
    if x0 is None:
        # a relative-interior start: the analytic-centre-like LP point, then pushed inward
        r = linprog(np.zeros(N), A_eq=Af, b_eq=bf, bounds=[(0, None)] * N, method="highs-ipm"); x0 = r.x
    As = Af[:, sup]; Ns = sl.null_space(As); k = Ns.shape[1]
    xs = mp.matrix([mp.mpf(float(x0[c])) for c in sup]); Nm = mp.matrix(Ns.tolist()) if k else None
    # exact feasibility in multiprecision: project the start onto A_s x = b
    Asm = mp.matrix(As.tolist()); bm = mp.matrix([mp.mpf(Fr(v).numerator) / Fr(v).denominator for v in b])
    res = bm - Asm * xs
    corr = mp.lu_solve(Asm.T * Asm + mp.eye(len(sup)) * mp.mpf(10) ** (-30), Asm.T * res) if len(sup) else 0
    xs = xs + corr
    def F(x): return mp.fsum(phi_q(x[i], q) for i in range(len(sup)))
    def d1(x):
        if q == 1: return -mp.log(x) - 1
        if q == 0: return 1 / x
        return (q * x ** (q - 1) - 1) / (q * (1 - q))
    def d2(x):
        if q == 1: return -1 / x
        if q == 0: return -1 / x ** 2
        return (q * (q - 1) * x ** (q - 2)) / (q * (1 - q))
    for _ in range(iters):
        if k == 0: break
        g = Nm.T * mp.matrix([d1(xs[i]) for i in range(len(sup))])
        Hd = [d2(xs[i]) for i in range(len(sup))]
        H = mp.matrix(k, k)
        for i in range(k):
            for j in range(i, k):
                v = mp.fsum(Hd[c] * Nm[c, i] * Nm[c, j] for c in range(len(sup))); H[i, j] = v; H[j, i] = v
        dz = mp.lu_solve(H, -g); dx = Nm * dz; t = mp.mpf(1)
        while any(xs[i] + t * dx[i] <= 0 for i in range(len(sup))): t /= 2
        f0 = F(xs)
        while F(xs + t * dx) < f0 + mp.mpf("1e-4") * t * (g.T * dz)[0] and t > mp.mpf(10) ** -30: t /= 2
        xs = xs + t * dx
        if mp.norm(t * dx) < mp.mpf(10) ** (-dps + 10): break
    out = [mp.mpf(0)] * N
    for i, c in enumerate(sup): out[c] = xs[i]
    return out


def markov_point(gs):
    """the Markov chain law of a chain of couplings, in the cell order of selcomp.fibre_constraints (strictly positive on the support)"""
    shape, cells, A, b = S.fibre_constraints(gs); out = []
    for c in cells:
        v = Fr(gs[0][c[0]][c[1]])
        for k in range(1, len(gs)):
            pk = sum(Fr(gs[k - 1][i][c[k]]) for i in range(shape[k - 1])); v = v * Fr(gs[k][c[k]][c[k + 1]]) / pk
        out.append(v)
    return out


def chain_power_section(gs, q, dps=40):
    shape, cells, A, b = S.fibre_constraints(gs); x0 = np.array([float(v) for v in markov_point(gs)])
    return cells, power_section(A, b, q, x0=x0, dps=dps)
