"""Composition of couplings: Markov gluing (associative) against composition by selection (not associative), and the associahedral
configuration of selected composites (companion note, section 12.9). New module; it modifies nothing.

A chain is a list of couplings g_i in Pi(p_i, p_(i+1)) (2-d arrays of Fractions with full-support marginals). Its fibre is the polytope of
multi-couplings x on X_a x ... x X_b with the consecutive pair marginals g_a, ..., g_(b-1).

  markov(g1, g2)            g1 D^-1 g2: conditional-independence gluing over the shared marginal (exact)
  markov_chain(gs)          the Markov multi-coupling of a chain (the maximum-entropy point of its fibre)
  fibre_constraints(gs)     the equality system of the fibre
  facet_cells(gs)           cells c whose zero set is a facet of the fibre
  leximin(gs)               the facet-leximin multi-coupling (LP levels, then the point solved exactly in rationals and checked)
  select(gs)                its marginal on (first, last): the composite of the block by selection
  dissections(n)            the faces of the associahedron K_n as dissections of the (n+1)-gon with vertices 0..n (sets of diagonals)
  composite(gs, D)          the composite of the chain along a dissection D: each cell of D is one selection in its own multi-table, whose
                            side couplings are inputs or composites of smaller cells
  softmin(gs, L)            the minimiser of sum over facet cells of exp(-L x_c) on the fibre (the positive critical point at T = e^-L, Theorem 27)
"""
from __future__ import annotations
import itertools, math
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import linprog
import mpmath as mp


def frac(a): return np.array([[Fr(x) for x in row] for row in a], dtype=object)


def markov(g1, g2):
    pB = [sum(g1[i][j] for i in range(len(g1))) for j in range(len(g1[0]))]
    return np.array([[sum(g1[a][b] * g2[b][c] / pB[b] for b in range(len(pB))) for c in range(len(g2[0]))] for a in range(len(g1))], dtype=object)


def markov_chain(gs):
    shape = [len(gs[0])] + [len(g[0]) for g in gs]; out = {}
    for idx in itertools.product(*[range(s) for s in shape]):
        v = gs[0][idx[0]][idx[1]]
        for k in range(1, len(gs)):
            pk = sum(gs[k - 1][i][idx[k]] for i in range(shape[k - 1])); v = v * gs[k][idx[k]][idx[k + 1]] / pk
        out[idx] = v
    return shape, out


def fibre_constraints(gs):
    shape = [len(gs[0])] + [len(g[0]) for g in gs]; cells = list(itertools.product(*[range(s) for s in shape])); A = []; b = []
    for k, g in enumerate(gs):
        for i in range(shape[k]):
            for j in range(shape[k + 1]):
                A.append([1 if (c[k] == i and c[k + 1] == j) else 0 for c in cells]); b.append(Fr(g[i][j]))
    return shape, cells, A, b


def _support(Af, bf, extra_zero=(), delta=1e-5):
    """cells that can be positive on {x >= 0, A x = b, x_c = 0 for c in extra_zero}: one LP for a relative-interior point"""
    N = Af.shape[1]; Ae = np.c_[Af, np.zeros((len(Af), N))]
    if extra_zero: Ae = np.vstack([Ae] + [np.r_[np.eye(N)[c], np.zeros(N)][None, :] for c in extra_zero])
    be = np.r_[bf, np.zeros(len(extra_zero))]
    Aub = np.c_[-np.eye(N), np.eye(N)]                                     # z_d <= x_d
    r = linprog(np.r_[np.zeros(N), -np.ones(N)], A_ub=Aub, b_ub=np.zeros(N), A_eq=Ae, b_eq=be, bounds=[(0, None)] * N + [(0, delta)] * N, method="highs")
    if r.status != 0: return None
    return [d for d in range(N) if r.x[N + d] > delta / 2]


def facet_cells(gs, eps=1e-10):
    """c is a facet cell iff {x in the fibre : x_c = 0} has dimension dim(fibre) - 1 (so tied cells sharing a facet all count)"""
    shape, cells, A, b = fibre_constraints(gs); N = len(cells); Af = np.array(A, float); bf = np.array([float(x) for x in b])
    sup = _support(Af, bf); zero = [d for d in range(N) if d not in sup]
    dim = N - _rank(A + [[1 if k == d else 0 for k in range(N)] for d in zero]); out = []
    for c in sup:
        sc = _support(Af, bf, [c])
        if sc is None: continue
        z = [d for d in range(N) if d not in sc]
        if N - _rank(A + [[1 if k == d else 0 for k in range(N)] for d in z]) == dim - 1: out.append(c)
    return out


def _rank(rows):
    M = [list(map(Fr, r)) for r in rows]; rk = 0; ncol = len(M[0]) if M else 0
    for col in range(ncol):
        p = next((i for i in range(rk, len(M)) if M[i][col] != 0), None)
        if p is None: continue
        M[rk], M[p] = M[p], M[rk]
        for i in range(len(M)):
            if i != rk and M[i][col] != 0:
                f = M[i][col] / M[rk][col]; M[i] = [a - f * b_ for a, b_ in zip(M[i], M[rk])]
        rk += 1
    return rk


def _solve_exact(A, b):
    """the unique solution of A x = b (rationals), by Gauss-Jordan; raises if not unique or inconsistent"""
    M = [list(map(Fr, r)) + [Fr(v)] for r, v in zip(A, b)]; n = len(A[0]); rk = 0; piv = []
    for col in range(n):
        p = next((i for i in range(rk, len(M)) if M[i][col] != 0), None)
        if p is None: continue
        M[rk], M[p] = M[p], M[rk]; f = M[rk][col]; M[rk] = [x / f for x in M[rk]]
        for i in range(len(M)):
            if i != rk and M[i][col] != 0:
                g = M[i][col]; M[i] = [a - g * b_ for a, b_ in zip(M[i], M[rk])]
        piv.append(col); rk += 1
    if rk < n: raise ValueError("not unique")
    if any(all(x == 0 for x in r[:-1]) and r[-1] != 0 for r in M): raise ValueError("inconsistent")
    x = [Fr(0)] * n
    for i, col in enumerate(piv): x[col] = M[i][-1]
    return x


def leximin(gs, tol=1e-10):
    shape, cells, A, b = fibre_constraints(gs); N = len(cells); F = facet_cells(gs); Af = np.array(A, float); bf = np.array([float(x) for x in b])
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
    # exact: margins plus fixed cells determine the point; re-solve each level value exactly from the system with that level as unknown
    rows = [list(r) for r in A]; rhs = list(b); nlev = len(levels)
    Aext = [r + [0] * nlev for r in rows]
    for c in zero: row = [0] * (N + nlev); row[c] = 1; Aext.append(row); rhs.append(Fr(0))
    for j, (_, S) in enumerate(levels):
        for c in S: row = [0] * (N + nlev); row[c] = 1; row[N + j] = -1; Aext.append(row); rhs.append(Fr(0))
    sol = _solve_exact(Aext, rhs); x = sol[:N]; lev = sol[N:]
    if min(x) < 0 or any(abs(float(l) - float(t)) > 1e-7 for l, (t, _) in zip(lev, levels)): raise ValueError("exact re-solve disagrees with the LP")
    return shape, cells, dict(zip(cells, x)), [(l, S) for l, (_, S) in zip(lev, levels)]


def select(gs):
    shape, cells, x, _ = leximin(gs); out = np.full((shape[0], shape[-1]), Fr(0), dtype=object)
    for c, v in x.items(): out[c[0], c[-1]] += v
    return out


def dissections(n):
    """faces of K_n: sets of pairwise non-crossing diagonals of the (n+1)-gon 0..n (a diagonal (i, j), j - i >= 2, (i, j) != (0, n))"""
    diags = [(i, j) for i in range(n + 1) for j in range(i + 2, n + 1) if (i, j) != (0, n)]
    def cross(d, e): (a, b), (c, d2) = d, e; return a < c < b < d2 or c < a < d2 < b
    out = []
    for r in range(0, n - 1):
        for S in itertools.combinations(diags, r):
            if all(not cross(d, e) for d, e in itertools.combinations(S, 2)): out.append(frozenset(S))
    return out


def composite(gs, D, rule=select, memo=None):
    """composite along dissection D of the chain gs (vertices 0..n); the cell under the root edge (0, n) is split recursively"""
    n = len(gs); memo = {} if memo is None else memo
    def edge(i, j):
        if j == i + 1: return gs[i]
        key = (i, j)
        if key in memo: return memo[key]
        inner = [d for d in D if i <= d[0] and d[1] <= j and d != (i, j)]
        maximal = [d for d in inner if not any(e != d and e[0] <= d[0] and d[1] <= e[1] for e in inner)]
        verts = [v for v in range(i, j + 1) if not any(d[0] < v < d[1] for d in maximal)]
        sides = [edge(a, b) for a, b in zip(verts, verts[1:])]
        res = rule(sides) if len(sides) > 1 else sides[0]; memo[key] = res; return res
    return edge(0, n)


def markov_rule(sides):
    out = sides[0]
    for g in sides[1:]: out = markov(out, g)
    return out


def softmin(gs, L, dps=None):
    """argmin over the fibre of sum_{c facet} exp(-L x_c) (damped Newton in multiprecision, from the Markov point, with continuation in L);
    gs may hold Fractions or mpf; returns the (first, last) marginal as mpf"""
    mp.mp.dps = dps or int(L / 2.3) + 50
    G = [[[mp.mpf(Fr(v).numerator) / Fr(v).denominator if isinstance(v, (Fr, int)) else mp.mpf(v) for v in row] for row in g] for g in gs]
    shape = [len(G[0])] + [len(g[0]) for g in G]; cells = list(itertools.product(*[range(s_) for s_ in shape])); N = len(cells)
    gf = [[[float(v) for v in row] for row in g] for g in G]; F = _facets_float(gf)
    x0 = []
    for idx in cells:
        v = G[0][idx[0]][idx[1]]
        for k in range(1, len(G)):
            pk = mp.fsum(G[k - 1][i_][idx[k]] for i_ in range(shape[k - 1])); v = v * G[k][idx[k]][idx[k + 1]] / pk
        x0.append(v)
    _, _, A, _ = fibre_constraints([[[1] * len(g[0]) for _ in g] for g in gf])
    import scipy.linalg as sl
    Ns = sl.null_space(np.array(A, float)); k = Ns.shape[1]; Nm = mp.matrix([[Ns[i_, j] for j in range(k)] for i_ in range(N)])
    def x_of(z): return [x0[i_] + mp.fsum(Nm[i_, j] * z[j] for j in range(k)) for i_ in range(N)]
    z = mp.matrix([0] * k)
    for Lc in [L * mp.mpf(2) ** (-j) for j in range(8, -1, -1)]:
        def Fv(z): x = x_of(z); return mp.fsum(mp.exp(-Lc * x[c]) for c in F)
        for _ in range(200):
            x = x_of(z); w = [mp.exp(-Lc * x[c]) for c in F]
            g = mp.matrix([-Lc * mp.fsum(w[t] * Nm[c, j] for t, c in enumerate(F)) for j in range(k)])
            H = mp.matrix([[Lc * Lc * mp.fsum(w[t] * Nm[c, i_] * Nm[c, j] for t, c in enumerate(F)) for j in range(k)] for i_ in range(k)])
            d = mp.lu_solve(H, -g); s_ = mp.mpf(1); f0 = Fv(z)
            while Fv(z + s_ * d) > f0 + mp.mpf("1e-4") * s_ * (g.T * d)[0] and s_ > mp.mpf(10) ** -30: s_ /= 2
            z = z + s_ * d
            if mp.norm(s_ * d) < mp.mpf(10) ** (-mp.mp.dps + 12): break
    x = x_of(z); out = [[mp.mpf(0)] * shape[-1] for _ in range(shape[0])]
    for c, v in zip(cells, x): out[c[0]][c[-1]] += v
    return out


def _facets_float(gf, eps=1e-10):
    shape, cells, A, b = fibre_constraints([[[1] * len(g[0]) for _ in g] for g in gf]); N = len(cells)
    b = []
    for k_, g in enumerate(gf):
        for i in range(len(g)):
            for j in range(len(g[0])): b.append(g[i][j])
    Af = np.array(A, float); bf = np.array(b); out = []
    for c in range(N):
        e = np.zeros(N); e[c] = 1
        r = linprog(np.r_[np.zeros(N), -1.0], A_ub=np.array([np.r_[-np.eye(N)[d], 1.0] for d in range(N) if d != c]), b_ub=np.zeros(N - 1),
                    A_eq=np.r_[np.c_[Af, np.zeros(len(Af))], [np.r_[e, 0.0]]], b_eq=np.r_[bf, 0.0], bounds=[(None, None)] * N + [(None, 1)], method="highs")
        if r.status == 0 and -r.fun > eps: out.append(c)
    return out


# ---------------------------------------------------------------- binary outer variables: selection is Frechet-midpoint gluing (section 12.10)
def frechet_mid(r, s, Q):
    """the midpoint of the Frechet-Hoeffding interval [max(0, r + s - Q), min(r, s)] of a 2 x 2 slice with margins (r, Q - r), (s, Q - s)"""
    return (max(Fr(0), r + s - Q) + min(r, s)) / 2


def select_binary(g1, g2):
    """selection for X0, X2 binary (X1 any): sum over b of the Frechet midpoint of the slice b; returns the 00 entry"""
    return sum(frechet_mid(g1[0][b], g2[b][0], g1[0][b] + g1[1][b]) for b in range(len(g2)))


def u_bin(p, q, r, t1, t2):
    """binary chain: couplings (p, q, t1) and (q, r, t2) given by their 00 entries; the 00 entry of the selected composite"""
    return frechet_mid(t1, t2, q) + frechet_mid(p - t1, r - t2, 1 - q)


def u_markov(p, q, r, t1, t2): return t1 * t2 / q + (p - t1) * (r - t2) / (1 - q)


def associator_bin(ps, ts):
    """A = sigma(sigma(g1, g2), g3) - sigma(g1, sigma(g2, g3)) (00 entries) for binary margins ps = (p0..p3) and 00 entries ts"""
    left = u_bin(ps[0], ps[2], ps[3], u_bin(ps[0], ps[1], ps[2], ts[0], ts[1]), ts[2])
    right = u_bin(ps[0], ps[1], ps[3], ts[0], u_bin(ps[1], ps[2], ps[3], ts[1], ts[2]))
    return left - right


# ---------------------------------------------------------------- exact certificates (section 12.10)
class _Aff:
    """an affine function c0 + c1 t1 + c2 t2 + c3 t3 with rational coefficients"""
    def __init__(self, c): self.c = [Fr(x) for x in c]
    def _o(self, o): return o if isinstance(o, _Aff) else _Aff([o, 0, 0, 0])
    def __add__(self, o): o = self._o(o); return _Aff([a + b for a, b in zip(self.c, o.c)])
    __radd__ = __add__
    def __sub__(self, o): o = self._o(o); return _Aff([a - b for a, b in zip(self.c, o.c)])
    def __rsub__(self, o): return self._o(o) - self
    def __truediv__(self, k): return _Aff([a / k for a in self.c])
    def val(self, t): return self.c[0] + sum(a * b for a, b in zip(self.c[1:], t))


def assoc_cell(ps, t):
    """at a rational point t = (t1, t2, t3) of a binary chain with margins ps: if every max/min of the Frechet-midpoint formula is strict, return
    (A as an affine function, the list of affine functions that must stay > 0 on the cell, including the open feasibility box); else None"""
    ineq = []; T = [_Aff([0, 1, 0, 0]), _Aff([0, 0, 1, 0]), _Aff([0, 0, 0, 1])]
    def mid(r, s, Q):
        d = (r + s - Q).val(t); g = (r - s).val(t)
        if d == 0 or g == 0: raise ValueError("tie")
        ineq.append((r + s - Q) if d > 0 else (Q - r - s)); ineq.append((s - r) if g < 0 else (r - s))
        return ((r + s - Q) if d > 0 else _Aff([0, 0, 0, 0])) + (r if g < 0 else s)
    def u(p, q, r, a, b): return (mid(a, b, _Aff([q, 0, 0, 0])) + mid(p - a, r - b, _Aff([1 - q, 0, 0, 0]))) / 2
    try:
        L = u(ps[0], ps[2], ps[3], u(ps[0], ps[1], ps[2], T[0], T[1]), T[2]); R = u(ps[0], ps[1], ps[3], T[0], u(ps[1], ps[2], ps[3], T[1], T[2]))
    except ValueError:
        return None
    for i in range(3):
        a, b = ps[i], ps[i + 1]; lo, hi = max(Fr(0), a + b - 1), min(a, b); e = [0, 0, 0, 0]; e[i + 1] = 1
        ineq += [_Aff(e) - lo, hi - _Aff(e)]
    return L - R, ineq


def _exact_rank(rows): return _rank(rows) if rows else 0


def _nullspace_exact(M, n):
    """a basis of {z : M z = 0} over the rationals (M: list of rows of length n)"""
    R = [list(map(Fr, r)) for r in M]; piv = []; rk = 0
    for col in range(n):
        p = next((i for i in range(rk, len(R)) if R[i][col] != 0), None)
        if p is None: continue
        R[rk], R[p] = R[p], R[rk]; f = R[rk][col]; R[rk] = [x / f for x in R[rk]]
        for i in range(len(R)):
            if i != rk and R[i][col] != 0: g = R[i][col]; R[i] = [a - g * b for a, b in zip(R[i], R[rk])]
        piv.append(col); rk += 1
    free = [c for c in range(n) if c not in piv]; basis = []
    for fcol in free:
        z = [Fr(0)] * n; z[fcol] = Fr(1)
        for i, pc in enumerate(piv): z[pc] = -R[i][fcol]
        basis.append(z)
    return basis


def _rational_point(Ae, be, zero, N, denom=10 ** 12):
    """a rational x >= 0 with Ae x = be, x_d = 0 (d in zero), as positive as possible elsewhere: LP, rounding, exact projection, exact check"""
    Af = np.array([[float(v) for v in r] for r in Ae]); bf = np.array([float(v) for v in be])
    sup = _support(Af, bf, list(zero))
    if sup is None: return None
    A2 = Ae + [[1 if k == d else 0 for k in range(N)] for d in zero]; b2 = list(be) + [Fr(0)] * len(zero)
    # interior point of the face: average of the per-cell maximisers is enough; use the support LP solution again with a larger box
    r = linprog(np.r_[np.zeros(N), -np.ones(N)], A_ub=np.c_[-np.eye(N), np.eye(N)], b_ub=np.zeros(N),
                A_eq=np.c_[np.array([[float(v) for v in row] for row in A2]), np.zeros((len(A2), N))], b_eq=np.array([float(v) for v in b2]),
                bounds=[(0, None)] * N + [(0, 1e-5)] * N, method="highs")
    x = [Fr(v).limit_denominator(denom) for v in r.x[:N]]
    # exact projection onto {A2 x = b2}
    B = [r_ for r_ in A2]; resid = [sum(a * xi for a, xi in zip(row, x)) - bb for row, bb in zip(B, b2)]
    if any(v != 0 for v in resid):
        # least-norm correction: solve (B B^T) w = resid on an independent subset of rows
        rows = []; idx = []
        for i, row in enumerate(B):
            if _rank(rows + [row]) > len(rows): rows.append(row); idx.append(i)
        G = [[sum(a * b for a, b in zip(ri, rj)) for rj in rows] for ri in rows]; w = _solve_exact(G, [resid[i] for i in idx])
        x = [xi - sum(w[k] * rows[k][j] for k in range(len(rows))) for j, xi in enumerate(x)]
    if any(sum(a * xi for a, xi in zip(row, x)) != bb for row, bb in zip(A2, b2)) or min(x) < 0: return None
    return x


def certify_leximin(gs):
    """an exact certificate that the facet-leximin point of the chain's fibre is what leximin() returns:
    (i) the point is feasible; (ii) the forced-zero cells and the fibre dimension; (iii) every facet cell has a witness of a face of dimension
    dim - 1; every other cell has an exact lower bound x_c >= m > 0 or forces extra zeros on {x_c = 0}; (iv) each level j has a balanced law
    q > 0 on S_j with sum_c q_c e_c in the row space of (margins, earlier fixed cells), which bounds min_{S_j} x_c by h_j and forces every
    c in S_j to h_j at every optimum; (v) the point has x_d > h_j for every cell freed after level j (levels strictly increase).
    Returns (ok, report)."""
    shape, cells, A, b = fibre_constraints(gs); N = len(cells); F = facet_cells(gs); _, _, xd, levels = leximin(gs); x = [xd[c] for c in cells]
    rep = {}; ok = all(sum(a * xi for a, xi in zip(row, x)) == bb for row, bb in zip(A, b)) and min(x) >= 0; rep["feasible"] = ok
    Af = np.array(A, float); bf = np.array([float(v) for v in b]); sup = _support(Af, bf); Z0 = [d for d in range(N) if d not in sup]
    # (ii) forced zeros: certificate y with A^T y >= e_d (componentwise) and y . b = 0 (then 0 <= x_d <= y^T A x = 0); found by LP, checked exactly
    for d in Z0:
        r = linprog(np.zeros(len(A)), A_ub=-Af.T, b_ub=-np.eye(N)[d], A_eq=bf[None, :], b_eq=[0.0], bounds=[(None, None)] * len(A), method="highs")
        y = [Fr(v).limit_denominator(10 ** 6) for v in r.x] if r.status == 0 else None
        g_ = y is not None and sum(yi * bb for yi, bb in zip(y, b)) == 0 and all(sum(y[i] * A[i][k] for i in range(len(A))) >= (1 if k == d else 0) for k in range(N)); ok &= g_
        if not g_: rep.setdefault('fail', []).append(('zero', d))
    x0 = _rational_point(A, b, Z0, N); ok &= x0 is not None and all((x0[k] == 0) == (k in Z0) for k in range(N))
    dim = N - _rank(A + [[1 if k == d else 0 for k in range(N)] for d in Z0]); rep["dim"] = dim
    # (iii) facets
    for c in range(N):
        if c in Z0: continue
        if c in F:
            w = _rational_point(A, b, [c] + Z0, N)
            if w is None: ok = False; continue
            Zc = [k for k in range(N) if w[k] == 0]; g_ = N - _rank(A + [[1 if k == d else 0 for k in range(N)] for d in Zc]) >= dim - 1; ok &= g_
            if not g_: rep.setdefault('fail', []).append(('facet', c))
        else:
            r = linprog(np.eye(N)[c], A_eq=Af, b_eq=bf, bounds=[(0, None)] * N, method="highs")
            if r.status == 0 and r.fun > 1e-9:
                rd = linprog(-bf, A_ub=Af.T, b_ub=np.eye(N)[c], bounds=[(None, None)] * len(A), method="highs")  # max y.b with A^T y <= e_c
                y = [Fr(v).limit_denominator(10 ** 6) for v in rd.x]
                g_ = all(sum(y[i] * A[i][k] for i in range(len(A))) <= (1 if k == c else 0) for k in range(N)) and sum(yi * bb for yi, bb in zip(y, b)) > 0; ok &= g_
                if not g_: rep.setdefault('fail', []).append(('lowerbound', c))
            else:
                w = _rational_point(A, b, [c] + Z0, N)
                if w is None: ok = False; continue
                Zc = [k for k in range(N) if w[k] == 0]
                # upper bound on the face dimension: the face lies in {x_k = 0, k in Zc'} with Zc' forced; certify forcing of each k in Zc \ {c}, Z0
                forced = [c] + Z0
                for k in Zc:
                    if k in forced: continue
                    # exact certificate that x_c = 0 forces x_k = 0: e_k <= A^T y + alpha e_c componentwise and y . b = 0 (so x_k <= alpha x_c)
                    nA = len(A); Mub = -np.c_[Af.T, np.eye(N)[c][:, None]]
                    r2 = linprog(np.zeros(nA + 1), A_ub=Mub, b_ub=-np.eye(N)[k], A_eq=np.r_[bf, 0.0][None, :], b_eq=[0.0],
                                 bounds=[(None, None)] * nA + [(0, None)], method="highs")
                    if r2.status != 0: continue
                    y = [Fr(v).limit_denominator(10 ** 8) for v in r2.x[:nA]]; al = Fr(r2.x[nA]).limit_denominator(10 ** 8)
                    if sum(yi * bb for yi, bb in zip(y, b)) == 0 and all(sum(y[i] * A[i][m] for i in range(nA)) + (al if m == c else 0) >= (1 if m == k else 0) for m in range(N)):
                        forced.append(k); rep["forced_certs"] = rep.get("forced_certs", 0) + 1
                g_ = N - _rank(A + [[1 if k == d else 0 for k in range(N)] for d in forced]) < dim - 1; ok &= g_
                if not g_: rep.setdefault('fail', []).append(('nonfacet', c))
    # (iv) balanced laws, (v) strict levels
    fixed = list(Z0)
    for j, (h, S) in enumerate(levels):
        M = A + [[1 if k == d else 0 for k in range(N)] for d in fixed]; NS = _nullspace_exact(M, N)
        E = [[z[c] for c in S] for z in NS] + [[Fr(1)] * len(S)]; f = [Fr(0)] * len(NS) + [Fr(1)]
        Ef = np.array([[float(v) for v in r] for r in E]); ff = np.array([float(v) for v in f]); k = len(S)
        r = linprog(np.r_[np.zeros(k), -1.0], A_ub=np.c_[-np.eye(k), np.ones(k)], b_ub=np.zeros(k), A_eq=np.c_[Ef, np.zeros(len(Ef))], b_eq=ff,
                    bounds=[(0, None)] * k + [(None, 1)], method="highs")
        if r.status != 0 or -r.fun <= 1e-12: ok = False; rep[f"level{j}"] = "no balanced law"; break
        q = [Fr(v).limit_denominator(10 ** 9) for v in r.x[:k]]
        rows = []; idx = []
        for i, row in enumerate(E):
            if _rank(rows + [row]) > len(rows): rows.append(row); idx.append(i)
        res = [sum(a * qi for a, qi in zip(row, q)) - ff_ for row, ff_ in zip(E, f)]
        if any(v != 0 for v in res):
            G = [[sum(a * b_ for a, b_ in zip(ri, rj)) for rj in rows] for ri in rows]; w = _solve_exact(G, [res[i] for i in idx])
            q = [qi - sum(w[t] * rows[t][jj] for t in range(len(rows))) for jj, qi in enumerate(q)]
        good = all(sum(a * qi for a, qi in zip(row, q)) == ff_ for row, ff_ in zip(E, f)) and min(q) > 0 and all(x[c] == h for c in S)
        ok &= good; fixed += list(S)
        if not good: rep.setdefault('fail', []).append(('level', j))
        later = [c for c in F if c not in fixed]
        g_ = all(x[c] > h for c in later); ok &= g_
        if not g_: rep.setdefault('fail', []).append(('strict', j))
    rep["levels"] = len(levels)
    return bool(ok), rep


def delzant(gs, cap=200000):
    """exact test whether the fibre polytope of a chain is simple and Delzant in the lattice of integer tables (directions in ker A with integer
    cells), and whether the toric variety is Fano: traverse the vertex graph by exact pivoting; at each vertex with tight facets S (one cell per
    facet) the edge vector z^(c) (z_s = 0 for s in S \\ c, z_c = 1) must be integral; the edge along c that reaches facet k is an invariant curve C
    with -K.C = 2 - sum_{s in S, s != c} z^(s)_k (the wall relation read in cell coordinates). Returns (status, vertices, min -K.C) with status
    'delzant', 'not simple', 'not unimodular' or 'cap'"""
    shape, cells, A, b = fibre_constraints(gs); N = len(cells); F = facet_cells(gs)
    Af = np.array(A, float); bf = np.array([float(x) for x in b]); sup = _support(Af, bf); Z0 = [d for d in range(N) if d not in sup]
    rows = A + [[1 if k == d else 0 for k in range(N)] for d in Z0]; rhs = list(b) + [Fr(0)] * len(Z0)
    reps = []
    for c in F:
        if not any(_rank(rows + [[(1 if k == c else 0) - (1 if k == r else 0) for k in range(N)]]) == _rank(rows) for r in reps): reps.append(c)
    dim = N - _rank(rows)
    r = linprog(np.random.default_rng(0).uniform(1, 2, N), A_eq=Af, b_eq=bf, bounds=[(0, None)] * N, method="highs")
    tight = [c for c in reps if r.x[c] < 1e-9]
    if len(tight) != dim: return "not simple", 0, None
    def vertex(S):
        return _solve_exact(rows + [[1 if k == c else 0 for k in range(N)] for c in S], rhs + [Fr(0)] * len(S))
    start = frozenset(tight); seen = {start}; stack = [start]; status = "delzant"; minK = None
    while stack:
        S = stack.pop(); v = vertex(sorted(S))
        if min(v) < 0: return "not simple", len(seen), minK
        Z = {}
        for c in S:
            others = sorted(S - {c})
            Z[c] = _solve_exact(rows + [[1 if k == s else 0 for k in range(N)] for s in others] + [[1 if k == c else 0 for k in range(N)]],
                                [Fr(0)] * len(rows) + [Fr(0)] * len(others) + [Fr(1)])
            if any(zz.denominator != 1 for zz in Z[c]): status = "not unimodular"
        for c in S:
            z = Z[c]; ts = [(v[k] / (-z[k]), k) for k in reps if k not in S and z[k] < 0]
            if not ts: continue
            tmin = min(t for t, _ in ts); hit = [k for t, k in ts if t == tmin]
            if len(hit) > 1: return "not simple", len(seen), minK
            k = hit[0]; KC = 2 - sum(Z[s][k] for s in S if s != c); minK = KC if minK is None else min(minK, KC)
            S2 = frozenset(sorted(S - {c}) + [k])
            if S2 not in seen:
                seen.add(S2); stack.append(S2)
                if len(seen) > cap: return "cap", len(seen), minK
    return status, len(seen), minK


def binary_slices(g1, g2):
    """the nondegenerate slices of a binary chain: (b, L_b, U_b, r_b, s_b, q_b) with L_b < U_b"""
    out = []
    for b in range(len(g2)):
        r, s, q = g1[0][b], g2[b][0], g1[0][b] + g1[1][b]; L, U = max(Fr(0), r + s - q), min(r, s)
        if L < U: out.append((b, L, U, r, s, q))
    return out
