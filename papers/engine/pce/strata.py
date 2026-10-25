"""Walls and chambers: where geometry, probability, context and outcome move together, and where they jump.

Fix the cover (the windows' variable sets, so the marginal matrix A is fixed) and let the window tables t vary. Everything the engine
computes is then a function of t, and the functions are of three kinds:
  * combinatorial invariants (support of the fibre, its face lattice, which cells of the gluability complex are filled, the place
    layer, the decision): piecewise constant, jump on walls;
  * polyhedral quantities (vertices of the fibre, lam*, the robust value of an action): piecewise affine, continuous, kink on walls;
  * the maximum-entropy merge and its decision: continuous, real-analytic inside each chamber, not affine.

  A_full(domains, windows)          the marginal matrix with the normalisation row, and the stacked tables
  support(domains, windows)         S(t): the atoms that some gluing charges (the support of the relative interior of the fibre); None if empty
  tangent_complex(domains, windows) the two-term deformation complex  T0 = R^S --A_S--> T1 = im A:  H0 = ker A_S (directions inside the
                                    fibre), H1 = im A / im A_S (table directions that cannot be followed without changing the support),
                                    chi = |S| - rank A = dim H0 - dim H1
  vertices(domains, windows)        the vertices of the fibre, each with its basis (small families: enumerates bases)
  chamber_type(domains, windows)    the combinatorial type: (support, set of vertex bases)
  robust_value(domains, windows, g) R(t) = min over gluings of <g, q> (one LP); convex in t, affine on each chamber
  lam_gradient(domains, windows)    the optimal dual y of lam* = max{1'b : A b <= t, b >= 0}; lam* = t'y and, where y is unique, grad lam* = y
                                    (the witness: across a wall the gradient jumps from one witness to another)
  walls_on_path(f, key, n, tol)     points s in (0, 1) where key(f(s)) changes, located by bisection
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import linprog
from .coherence import _constraints


def A_full(domains, windows):
    A, t, atoms, index = _constraints(domains, windows)
    return np.vstack([A, np.ones(A.shape[1])]), np.concatenate([t, [1.0]]), atoms


def support(domains, windows, tol=1e-9):
    A, t, atoms = A_full(domains, windows); n = A.shape[1]; S = []
    for k in range(n):
        e = np.zeros(n); e[k] = -1
        r = linprog(e, A_eq=A, b_eq=t, bounds=(0, None), method="highs")
        if r.status != 0: return None
        if -r.fun > tol: S.append(k)
    return tuple(S)


def _rank(M, tol=1e-9):
    return int(np.linalg.matrix_rank(M, tol)) if M.size else 0


def tangent_complex(domains, windows):
    A, t, atoms = A_full(domains, windows); S = support(domains, windows)
    rA = _rank(A)
    if S is None: return {"empty": True, "rankA": rA}
    rS = _rank(A[:, list(S)])
    return {"empty": False, "S": S, "H0": len(S) - rS, "H1": rA - rS, "chi": len(S) - rA, "rankA": rA}


def _row_basis(A, tol=1e-9):
    rows = []
    for i in range(A.shape[0]):
        if _rank(A[rows + [i]], tol) > len(rows): rows.append(i)
    return rows


def vertices(domains, windows, tol=1e-9):
    A, t, atoms = A_full(domains, windows); R = _row_basis(A); A, t = A[R], t[R]
    m, n = A.shape; out = []
    for B in itertools.combinations(range(n), m):
        M = A[:, B]
        if abs(np.linalg.det(M)) < 1e-12: continue
        x = np.linalg.solve(M, t)
        if np.all(x >= -tol):
            q = np.zeros(n); q[list(B)] = np.clip(x, 0, None); out.append((B, q))
    uniq = []
    for B, q in out:                                                      # degenerate vertices have several bases: keep one point
        if not any(np.allclose(q, q2, atol=1e-9) for _, q2 in uniq): uniq.append((B, q))
    return uniq


def chamber_type(domains, windows, tol=1e-9):
    """(support, frozenset of the supports of the vertices): constant on each open chamber."""
    S = support(domains, windows)
    if S is None: return None
    V = vertices(domains, windows, tol)
    return S, frozenset(tuple(np.nonzero(q > tol)[0]) for _, q in V)


def robust_value(domains, windows, g):
    A, t, atoms = A_full(domains, windows)
    r = linprog(np.asarray(g, float), A_eq=A, b_eq=t, bounds=(0, None), method="highs")
    return float(r.fun) if r.status == 0 else float("nan")


def lam_gradient(domains, windows):
    A, t, atoms, index = _constraints(domains, windows)
    d = linprog(t, A_ub=-A.T, b_ub=-np.ones(A.shape[1]), bounds=(0, None), method="highs")
    return float(d.fun), d.x


def walls_on_path(f, key, n=200, tol=1e-6):
    s = np.linspace(0, 1, n + 1); k = [key(f(x)) for x in s]; out = []
    for i in range(n):
        if k[i] == k[i + 1]: continue
        lo, hi, klo = s[i], s[i + 1], k[i]
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if key(f(mid)) == klo: lo = mid
            else: hi = mid
        out.append(0.5 * (lo + hi))
    return out
