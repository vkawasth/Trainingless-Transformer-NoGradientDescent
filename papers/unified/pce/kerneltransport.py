"""Transport when a revision changes the variables: stochastic kernels between versions. New module; it modifies nothing.

Fine version: X in {a, b, c, d} (0..3) and outcome Y in {0, 1}; outcomes indexed 2x + y (8 states). Coarse version: X' in {A, B}
(A = {a, b}, B = {c, d}) and Y; outcomes 2c + y (4 states). Laws are row vectors; a kernel is a row-stochastic matrix, P -> P K.

Data are tables on contexts; a context is given by a function from outcomes to labels (a 'statistic'), and a table assigns each label its
probability. The fibre of the data is a polytope, enumerated by its vertices.
  FINE, COARSE                 outcome lists
  merge()                      the coarsening kernel (deterministic)
  split(theta)                 a reconstruction kernel: (A, y) -> (a, y) w.p. theta[(0, y)], (b, y) otherwise; (B, y) -> (c, y) w.p.
                               theta[(1, y)], (d, y) otherwise. An outcome-independent split has theta[(k, 0)] = theta[(k, 1)]
  ctx_X, ctx_Xc_Y, ctx_X_Y, ctx_c, ctx_cY   statistics on fine (and coarse) outcomes
  table(P, ctx, outs)          the table of P on a context
  constraints(tables, outs)    equality rows
  vertices(tables, outs)       vertices of the fibre
  refit(Q, tables, outs)       iterative proportional fitting (the I-projection of Q onto the fibre, for positive Q)
  hol(Ka, Kb, V)               max over the vertices V of TV(P Ka, P Kb): the representation holonomy of two paths over the fibre
  tv(P, Q)
"""
from __future__ import annotations
import itertools
import numpy as np

FINE = [(x, y) for x in range(4) for y in range(2)]
COARSE = [(c, y) for c in range(2) for y in range(2)]
cls = lambda x: 0 if x < 2 else 1


def merge():
    K = np.zeros((8, 4))
    for i, (x, y) in enumerate(FINE): K[i, COARSE.index((cls(x), y))] = 1.0
    return K


def split(theta):
    K = np.zeros((4, 8))
    for j, (c, y) in enumerate(COARSE):
        lo, hi = (0, 1) if c == 0 else (2, 3); t = theta[(c, y)]
        K[j, FINE.index((lo, y))] = t; K[j, FINE.index((hi, y))] = 1 - t
    return K


ctx_X = lambda w: w[0]
ctx_Y = lambda w: w[1]
ctx_Xc_Y = lambda w: (cls(w[0]), w[1])
ctx_X_Y = lambda w: (w[0], w[1])
ctx_c = lambda w: w[0]                       # on coarse outcomes
ctx_cY = lambda w: (w[0], w[1])


def table(P, ctx, outs):
    t = {}
    for w, p in zip(outs, P): t[ctx(w)] = t.get(ctx(w), 0.0) + p
    return t


def constraints(tables, outs):
    A = [np.ones(len(outs))]; b = [1.0]
    for ctx, tab in tables:
        for lab, v in tab.items():
            A.append(np.array([1.0 if ctx(w) == lab else 0.0 for w in outs])); b.append(v)
    return np.array(A), np.array(b)


def vertices(tables, outs):
    A, b = constraints(tables, outs); r = np.linalg.matrix_rank(A); n = len(outs); V = []
    for basis in itertools.combinations(range(n), r):
        B = A[:, basis]
        if np.linalg.matrix_rank(B) < r: continue
        x = np.linalg.lstsq(B, b, rcond=None)[0]
        if np.abs(B @ x - b).max() > 1e-9 or x.min() < -1e-12: continue
        P = np.zeros(n); P[list(basis)] = np.clip(x, 0, None)
        if not any(np.abs(P - v).max() < 1e-10 for v in V): V.append(P)
    return np.array(V)


def jeffrey(P, ctx, tab, outs):
    m = table(P, ctx, outs); return np.array([p * tab[ctx(w)] / m[ctx(w)] for w, p in zip(outs, P)])


def refit(Q, tables, outs, sweeps=3000):
    for _ in range(sweeps):
        for ctx, tab in tables: Q = jeffrey(Q, ctx, tab, outs)
    return Q


def tv(P, Q): return 0.5 * float(np.abs(np.asarray(P) - np.asarray(Q)).sum())


def hol(Ka, Kb, V): return max(tv(P @ Ka, P @ Kb) for P in V)


def condY(P, x):
    """P(Y = 1 | X = x) on fine outcomes"""
    return P[FINE.index((x, 1))] / (P[FINE.index((x, 0))] + P[FINE.index((x, 1))])
