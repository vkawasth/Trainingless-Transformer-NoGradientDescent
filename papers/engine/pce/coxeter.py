"""Representation schemes of right-angled Coxeter groups: the observable algebra of a compatibility graph.

For a graph G (vertices = +-1 observables, edges = commuting pairs) the observables generate the right-angled Coxeter group W_G
(x_v^2 = 1, x_u x_v = x_v x_u on edges). Quantum models in dimension n are points of Rep_n(W_G) = {(X_v) in M_n^V : X_v^2 = I,
X_u X_v = X_v X_u on edges}. Group theory: W_G is virtually free iff G is chordal, and in characteristic 0 the group algebra Q[W_G] is
formally smooth (quasi-free) iff W_G is virtually free; formal smoothness forces every Rep_n to be smooth. This module tests the other
direction on matrices: at points of Rep_n, does some tangent vector fail to extend to second order (a certificate of a singular point)?

  linearisation(G, X)          the linear map L(Y) = (X_v Y_v + Y_v X_v ; [X_u, Y_v] + [Y_u, X_v] on edges) as a matrix
  quadratic(G, Y)              Q(Y) = (Y_v^2 ; [Y_u, Y_v] on edges); the second-order equation is L(Z) = -Q(Y)
  analyse(G, X, trials, rng)   tangent dimension, orbit dimension (n^2 - dim of the commutant), H^1 = tangent - orbit, and whether a
                               random tangent vector is obstructed at second order (numerical, with a rank test)
  exact_obstruction(G, X, Y)   exact rational check (sympy) that L(Z) = -Q(Y) has no solution for the given rational tangent vector Y
  diagonal_points(G, n)        all points that are direct sums of characters (X_v diagonal +-1), up to simultaneous reordering of the
                               coordinates; these are the most reducible points, where singularities concentrate
  is_chordal(G)                every cycle of length >= 4 has a chord (perfect elimination ordering)
"""
from __future__ import annotations
import itertools
import numpy as np


def _vec(Ms):
    return np.concatenate([M.ravel() for M in Ms])


def linearisation(G, X):
    V, E = G; n = X[0].shape[0]; N = len(V) * n * n; cols = []
    for k in range(N):
        y = np.zeros(N); y[k] = 1.0; Y = [y[i * n * n:(i + 1) * n * n].reshape(n, n) for i in range(len(V))]
        cols.append(_vec([X[i] @ Y[i] + Y[i] @ X[i] for i in range(len(V))] + [X[u] @ Y[v] - Y[v] @ X[u] + Y[u] @ X[v] - X[v] @ Y[u] for u, v in E]))
    return np.array(cols).T


def quadratic(G, Y):
    V, E = G
    return _vec([Y[i] @ Y[i] for i in range(len(V))] + [Y[u] @ Y[v] - Y[v] @ Y[u] for u, v in E])


def _null(M, tol=1e-9):
    u, s, vt = np.linalg.svd(M); r = int(np.sum(s > tol * max(1, s.max()))); return vt[r:].T, r


def analyse(G, X, trials=3, rng=None):
    V, E = G; n = X[0].shape[0]; rng = rng or np.random.default_rng(0)
    L = linearisation(G, X); K, r = _null(L); tdim = K.shape[1]
    C = np.vstack([np.kron(Xv, np.eye(n)) - np.kron(np.eye(n), Xv.T) for Xv in X]); comm = _null(C)[0].shape[1]
    orbit = n * n - comm; obstructed = False; worst = 0.0
    for _ in range(trials):
        y = K @ rng.normal(size=tdim) if tdim else np.zeros(L.shape[1])
        Y = [y[i * n * n:(i + 1) * n * n].reshape(n, n) for i in range(len(V))]; q = -quadratic(G, Y)
        z, *_ = np.linalg.lstsq(L, q, rcond=None); res = np.linalg.norm(L @ z - q) / max(1.0, np.linalg.norm(q))
        worst = max(worst, res); obstructed |= res > 1e-8
    return {"tangent": tdim, "orbit": orbit, "H1": tdim - orbit, "obstructed": bool(obstructed), "residual": float(worst)}


def exact_obstruction(G, X, Yrat):
    import sympy as sp
    V, E = G; n = X[0].shape[0]
    Ls = sp.Matrix(np.rint(linearisation(G, X)).astype(int))
    Ys = [sp.Matrix(Y) for Y in Yrat]
    assert all(v == 0 for v in (Ls * sp.Matrix([e for Y in Ys for e in Y]))), "Y is not a tangent vector"
    q = []
    for i in range(len(V)): q += list(-(Ys[i] * Ys[i]))
    for u, v in E: q += list(-(Ys[u] * Ys[v] - Ys[v] * Ys[u]))
    q = sp.Matrix(q)
    return Ls.rank() < Ls.row_join(q).rank()          # True: no Z with L Z = -Q(Y), so Y is obstructed


def diagonal_points(G, n):
    V, E = G; seen = set()
    for signs in itertools.product((1, -1), repeat=len(V) * n):
        S = np.array(signs).reshape(len(V), n); key = tuple(sorted(tuple(S[:, j]) for j in range(n)))
        if key in seen: continue
        seen.add(key); yield [np.diag(S[i]).astype(float) for i in range(len(V))], key


def is_chordal(G):
    V, E = G; adj = {v: set() for v in range(len(V))}
    for u, v in E: adj[u].add(v); adj[v].add(u)
    rem = set(adj)
    while rem:
        simp = [v for v in rem if all(b in adj[a] for a, b in itertools.combinations(adj[v] & rem, 2))]
        if not simp: return False
        rem.discard(simp[0])
    return True
