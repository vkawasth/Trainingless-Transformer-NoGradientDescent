"""Toric strata: AMB outcomes as boundary strata of a log-linear model, and the AMB toggle radius.

A hierarchical log-linear (toric) model on a table with cells j has a design matrix A (columns a_j = indicator
vectors of the margins that cell j contributes to). Its closure is a union of strata, one for each FACIAL SET S
(a set of cells that is the zero set of a supporting hyperplane of cone(A): h.a_j = 0 on S, h.a_j > 0 off S);
the stratum of S is the log-linear model restricted to the cells in S (Geiger-Meek-Sturmfels 2006; Fienberg-Rinaldo
2012). The support of a distribution in the closure -- which cells are possible -- is the AMB (possibilistic)
outcome; the interior is the stratum S = all cells.

Proposition (AMB toggle radius). For p in the interior and a facial set S, the closest point of the S-stratum in
reverse KL is p restricted to S and renormalised, at distance KL(p|_S || p) = -log p(S). Hence the smallest
change of the probability layer that makes cell c impossible, while staying in the model, is
    r(c) = min { -log(1 - p(F)) : c in F, complement of F facial }.
F is the set of cells that must vanish TOGETHER with c. For the saturated model F = {c}; for the independence model
F is the row or the column of c; the more the model constrains (the more it glues), the more mass must move.
We bracket r(c): lower bound -log(1 - p_c); upper bound from a co-facial set F found by the linear program
    min sum_j p_j (h.a_j)  s.t.  h.a_j >= 0 for all j,  h.a_c >= 1,
whose optimal h is a supporting hyperplane (its zero set is facial), so F = {j : h.a_j > 0} is achievable.
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import linprog


def design(shape, margins):
    """rows: one per margin cell; columns: table cells (C order). margins: list of axis tuples, e.g. [(0,1),(0,2),(1,2)]."""
    cells = list(itertools.product(*[range(n) for n in shape]))
    rows = []
    for m in margins:
        for key in itertools.product(*[range(shape[a]) for a in m]):
            rows.append([1.0 if tuple(c[a] for a in m) == key else 0.0 for c in cells])
    return np.array(rows), cells


def ipf(N, margins, iters=500, tol=1e-10):
    M = np.full(N.shape, N.sum() / N.size, float)
    for _ in range(iters):
        old = M.copy()
        for m in margins:
            other = tuple(a for a in range(N.ndim) if a not in m)
            tgt = N.sum(axis=other, keepdims=True); cur = M.sum(axis=other, keepdims=True)
            M = M * np.where(cur > 0, tgt / np.maximum(cur, 1e-300), 0.0)
        if np.abs(M - old).max() < tol:
            break
    return M


def cofacial_set(A, p, c, tol=1e-9):
    """LP upper-bound co-facial set containing cell index c (see module doc). Returns (F, h) or (None, None)."""
    d, n = A.shape
    res = linprog(c=A @ p, A_ub=np.vstack([-A.T, -A[:, c][None, :]]), b_ub=np.concatenate([np.zeros(n), [-1.0]]),
                  bounds=[(None, None)] * d, method="highs")
    if not res.success:
        return None, None
    v = A.T @ res.x
    return np.where(v > tol * max(1.0, v.max()))[0], res.x


def is_facial(A, S, tol=1e-9):
    """S facial iff exists h with h.a_j = 0 on S, h.a_j >= 1 off S."""
    d, n = A.shape; off = [j for j in range(n) if j not in set(S)]
    if not off:
        return True
    res = linprog(c=np.zeros(d), A_eq=A[:, list(S)].T if len(S) else None, b_eq=np.zeros(len(S)) if len(S) else None,
                  A_ub=-A[:, off].T, b_ub=-np.ones(len(off)), bounds=[(None, None)] * d, method="highs")
    return bool(res.success)


def toggle_radius_amb(A, p, c):
    F, _ = cofacial_set(A, p, c)
    lo = -np.log1p(-p[c])
    hi = -np.log1p(-p[F].sum()) if F is not None and p[F].sum() < 1 else np.inf
    return dict(cell=int(c), lower=float(lo), upper=float(hi), forced=[int(j) for j in F] if F is not None else None,
                mass_forced=float(p[F].sum()) if F is not None else None)


def exact_toggle(A, p, c, K=1e3, time_limit=60):
    """Exact AMB toggle radius by MILP: min sum_j p_j z_j  s.t.  h.a_j >= 0,  h.a_j <= M z_j,  h.a_c >= 1, z binary.
    The optimal F = {j : z_j = 1, h.a_j > 0} is a minimum-mass co-facial set containing c (checked facial)."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    d, n = A.shape; M = 3 * K * A.sum(0).max()
    # variables: h (d, free in [-K, K]), z (n, binary)
    cost = np.concatenate([np.zeros(d), p])
    G1 = np.hstack([A.T, np.zeros((n, n))])                     # h.a_j >= 0
    G2 = np.hstack([A.T, -M * np.eye(n)])                       # h.a_j - M z_j <= 0
    G3 = np.concatenate([A[:, c], np.zeros(n)])[None, :]        # h.a_c >= 1
    cons = [LinearConstraint(G1, 0, np.inf), LinearConstraint(G2, -np.inf, 0), LinearConstraint(G3, 1, np.inf)]
    integrality = np.concatenate([np.zeros(d), np.ones(n)])
    bounds = Bounds(np.concatenate([-K * np.ones(d), np.zeros(n)]), np.concatenate([K * np.ones(d), np.ones(n)]))
    res = milp(cost, constraints=cons, integrality=integrality, bounds=bounds, options=dict(time_limit=time_limit))
    if res.x is None:
        return None
    h = res.x[:d]; v = A.T @ h
    F = np.where(v > 1e-7)[0]
    return dict(cell=int(c), forced=[int(j) for j in F], mass=float(p[F].sum()), radius=float(-np.log1p(-p[F].sum())),
                optimal=bool(res.status == 0))
