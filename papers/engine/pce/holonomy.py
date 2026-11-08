"""History-indexed selection and its holonomy (companion note, section 12.26). New module; it modifies nothing and uses pce/margsec.py.
A history is a sequence of snapshots (contexts observed so far, their tables). An updating rule carries the previously selected joint law to
the current fibre. The rule is path independent when the law it reaches depends only on the final snapshot; the holonomy of two histories
with the same final snapshot is the total-variation distance between the laws they reach.
  ctx_rows(k)                     the four indicator rows of context k of margsec.CTX (cells of margsec.CELLS)
  tables(x)                       the four context tables of a joint law
  constraints(ks, T)              (A, b) for the fibre of the contexts ks with tables T (plus normalisation)
  kl_update(q, ks, T)             I-projection of q onto the fibre (iterative proportional fitting started at q)
  euclid_update(q, ks, T)         Euclidean nearest point of the fibre (exact active-set QP, written out here; no solver dependency)
  tv_update(q, ks, T)             total-variation nearest point, ties broken by the least value of a fixed linear order (two LPs)
  run(rule, history, start)       the selected laws along a history [(ks, T), ...]
  holonomy(rule, H1, H2, start)   TV distance between the final laws of two histories
  glues(ks, T)                    whether the tables of the contexts ks have a common joint law (LP feasibility)
  tv_diameter(ks, T)              the TV diameter of the fibre (largest TV distance over pairs of vertices, by enumeration of the LP
                                  vertices; small fibres only)
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import linprog
from . import margsec as G

C = np.array(G.CELLS); CTX = G.CTX; N = 16


def ctx_rows(k):
    i, j = CTX[k]; return np.array([[1.0 if (c[i] == a and c[j] == b) else 0.0 for c in C] for a in (0, 1) for b in (0, 1)])


def tables(x):
    return [ctx_rows(k) @ np.asarray(x, float) for k in range(4)]


def constraints(ks, T):
    A = np.vstack([ctx_rows(k) for k in ks] + [np.ones((1, N))]); b = np.concatenate([np.asarray(T[k], float) for k in ks] + [[1.0]])
    return A, b


def _independent(A, b, tol=1e-10):
    rows, basis = [], np.zeros((0, A.shape[1]))
    for i, r in enumerate(A):
        if np.linalg.matrix_rank(np.vstack([basis, r]), tol) > basis.shape[0]: rows.append(i); basis = np.vstack([basis, r])
    return A[rows], b[rows]


def kl_update(q, ks, T, iters=50000, tol=1e-14):
    x = np.asarray(q, float).copy()
    for _ in range(iters):
        err = 0.0
        for k in ks:
            for r, t in zip(ctx_rows(k), T[k]):
                m = r > 0; cur = x[m].sum(); err = max(err, abs(cur - t))
                if cur > 0: x[m] *= t / cur
        if err < tol: break
    return x


def euclid_update(q, ks, T, tol=1e-12):
    """min |x - q|^2 s.t. A x = b, x >= 0, by a primal active-set method on the zero set Z (x_i = 0 for i in Z): each step solves the
    equality-constrained least-squares problem exactly and checks the multipliers (KKT)."""
    A, b = _independent(*constraints(ks, T)); q = np.asarray(q, float)
    def solve(Z):
        free = [i for i in range(N) if i not in Z]; Af = A[:, free]
        y = np.linalg.lstsq(Af @ Af.T, Af @ q[free] - b, rcond=None)[0]                 # x_free = q_free - Af^T y
        x = np.zeros(N); x[free] = q[free] - Af.T @ y; return x, y
    # start from a feasible point (LP) and keep feasibility (classical primal active set for strictly convex QP)
    r = linprog(np.zeros(N), A_eq=A, b_eq=b, bounds=[(0, None)] * N, method="highs")
    if r.status != 0: return None
    x = r.x; Z = set(i for i in range(N) if x[i] < tol)
    for _ in range(500):
        xs, y = solve(Z); p = xs - x
        if np.abs(p).max() < 1e-13:
            mu = {i: (x - q + A.T @ y)[i] for i in Z}                                  # multiplier of x_i >= 0
            neg = [i for i in Z if mu[i] < -1e-11]
            if not neg: return np.where(x < 0, 0.0, x)
            Z.discard(min(neg, key=lambda i: mu[i])); continue
        alpha, block = 1.0, None
        for i in range(N):
            if i not in Z and p[i] < -1e-15:
                a_ = -x[i] / p[i]
                if a_ < alpha: alpha, block = a_, i
        x = x + alpha * p
        if block is not None: Z.add(block); x[block] = 0.0
    raise RuntimeError("active set did not converge")


def tv_update(q, ks, T):
    A, b = constraints(ks, T); q = np.asarray(q, float)
    Aub = np.block([[np.eye(N), -np.eye(N)], [-np.eye(N), -np.eye(N)]]); bub = np.r_[q, -q]; Aeq = np.c_[A, np.zeros((A.shape[0], N))]
    r = linprog(np.r_[np.zeros(N), np.ones(N)], A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=b, bounds=[(0, None)] * (2 * N), method="highs")
    if r.status != 0: return None
    r2 = linprog(np.r_[np.arange(1, N + 1) / N, np.zeros(N)], A_ub=np.vstack([Aub, np.r_[np.zeros(N), np.ones(N)]]), b_ub=np.r_[bub, r.fun + 1e-11],
                 A_eq=Aeq, b_eq=b, bounds=[(0, None)] * (2 * N), method="highs")
    return r2.x[:N]


RULES = {"KL": kl_update, "Euclid": euclid_update, "TV": tv_update}


def run(rule, history, start=None):
    q = np.full(N, 1 / N) if start is None else np.asarray(start, float); out = []
    for ks, T in history:
        q = rule(q, ks, T); out.append(q)
    return out


def tv(p, q): return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def holonomy(rule, H1, H2, start=None):
    return tv(run(rule, H1, start)[-1], run(rule, H2, start)[-1])


def glues(ks, T):
    A, b = constraints(ks, T); return linprog(np.zeros(N), A_eq=A, b_eq=b, bounds=[(0, None)] * N, method="highs").status == 0


def interval(ks, T, w):
    A, b = constraints(ks, T); w = np.asarray(w, float)
    lo = linprog(w, A_eq=A, b_eq=b, bounds=[(0, None)] * N, method="highs"); hi = linprog(-w, A_eq=A, b_eq=b, bounds=[(0, None)] * N, method="highs")
    return (lo.fun, -hi.fun) if lo.status == 0 else None


def fibre_dim(ks, T, tol=1e-9):
    """dimension of the fibre: 16 minus the rank of the constraints together with the cells forced to zero"""
    A, b = constraints(ks, T); zero = []
    for k in range(N):
        r = linprog(-np.eye(N)[k], A_eq=A, b_eq=b, bounds=[(0, None)] * N, method="highs")
        if r.status != 0: return None
        if -r.fun < tol: zero.append(k)
    return N - np.linalg.matrix_rank(np.vstack([A] + [np.eye(N)[z][None] for z in zero]), 1e-9)


def tv_diameter(ks, T, tol=1e-10):
    """largest TV distance between two laws of the fibre: max over pairs of vertices (a convex function attains its maximum over a polytope
    at vertices). Vertices are the basic feasible solutions, enumerated over supports of size rank(A)."""
    A, b = _independent(*constraints(ks, T)); r = A.shape[0]; V = []
    for S in itertools.combinations(range(N), r):
        B = A[:, S]
        if abs(np.linalg.det(B)) < 1e-12: continue
        xs = np.linalg.solve(B, b)
        if xs.min() < -tol: continue
        x = np.zeros(N); x[list(S)] = np.clip(xs, 0, None)
        if not any(np.abs(x - v).max() < 1e-9 for v in V): V.append(x)
    return max((tv(u, v) for u, v in itertools.combinations(V, 2)), default=0.0), V
