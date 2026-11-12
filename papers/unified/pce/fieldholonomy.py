"""Transport of the conditional field along revisions, and its holonomy. New module; it modifies nothing.

Three binary variables: two pathways R, S and the outcome Y; a law is a vector over the 8 outcomes (r, s, y). Revisions of the record are
edits of context tables. Two kinds of decision maker:
  recomputer   keeps no state: at every version its credal set is the fibre of the current data (path independent by construction);
  updater      carries a law forward: each revision is a Jeffrey update (I-projection onto the revised table of one context), and at the
               end it refits the current data (iterative proportional fitting).
  OUT                          the outcome tuples
  marg(P, ctx)                 the table of P on a context (tuple of variable indices 0 = R, 1 = S, 2 = Y), as a dict
  jeffrey(P, ctx, table)       P(w) * table[w_ctx] / P_ctx(w_ctx)
  refit(P, tables, sweeps)     iterative proportional fitting to all current tables
  bayes(P, L)                  likelihood reweighting
  interaction(P)               the conditional log odds ratios of (R, S) given Y = 0 and Y = 1 (the (R,S) and (R,S,Y) log-linear terms)
  pair_field(P)                (P(Y=1 | R=1), P(Y=1 | S=1)): determined by the data on {R,Y}, {S,Y}
  joint_field(P)               (P(Y=1 | R=1, S=0), P(Y=1 | R=0, S=1)): 'pathway alone' reaction functions, not determined by those data
  fibre_vertices(tables)       vertices of the fibre of the data (enumeration of basic feasible solutions)
  fibre_range(tables, fn, m)   min and max of a function over the fibre (vertices plus a dense sample of convex combinations)
  tv(P, Q)                     total variation
"""
from __future__ import annotations
import itertools
import numpy as np

OUT = list(itertools.product((0, 1), repeat=3))


def marg(P, ctx):
    t = {}
    for w, pw in zip(OUT, P): key = tuple(w[c] for c in ctx); t[key] = t.get(key, 0.0) + pw
    return t


def jeffrey(P, ctx, table):
    m = marg(P, ctx); return np.array([pw * table[tuple(w[c] for c in ctx)] / m[tuple(w[c] for c in ctx)] for w, pw in zip(OUT, P)])


def refit(P, tables, sweeps=2000):
    for _ in range(sweeps):
        for ctx, tab in tables: P = jeffrey(P, ctx, tab)
    return P


def bayes(P, L): Q = P * L; return Q / Q.sum()


def interaction(P):
    d = dict(zip(OUT, P))
    return tuple(np.log(d[(1, 1, y)] * d[(0, 0, y)] / (d[(1, 0, y)] * d[(0, 1, y)])) for y in (0, 1))


def _cond(P, num, den):
    a = sum(pw for w, pw in zip(OUT, P) if num(w)); b = sum(pw for w, pw in zip(OUT, P) if den(w)); return a / b


def pair_field(P):
    return (_cond(P, lambda w: w[0] == 1 and w[2] == 1, lambda w: w[0] == 1), _cond(P, lambda w: w[1] == 1 and w[2] == 1, lambda w: w[1] == 1))


def joint_field(P):
    return (_cond(P, lambda w: w[0] == 1 and w[1] == 0 and w[2] == 1, lambda w: w[0] == 1 and w[1] == 0),
            _cond(P, lambda w: w[0] == 0 and w[1] == 1 and w[2] == 1, lambda w: w[0] == 0 and w[1] == 1))


def tv(P, Q): return 0.5 * float(np.abs(np.asarray(P) - np.asarray(Q)).sum())


def _rows(tables):
    A = [np.ones(8)]; b = [1.0]
    for ctx, tab in tables:
        for key, v in tab.items():
            A.append(np.array([1.0 if tuple(w[c] for c in ctx) == key else 0.0 for w in OUT])); b.append(v)
    return np.array(A), np.array(b)


def fibre_vertices(tables):
    A, b = _rows(tables); r = np.linalg.matrix_rank(A); V = []
    for basis in itertools.combinations(range(8), r):
        B = A[:, basis]
        if np.linalg.matrix_rank(B) < r: continue
        x, res, *_ = np.linalg.lstsq(B, b, rcond=None)
        if np.abs(B @ x - b).max() > 1e-9 or x.min() < -1e-12: continue
        P = np.zeros(8); P[list(basis)] = np.clip(x, 0, None)
        if not any(np.abs(P - v).max() < 1e-10 for v in V): V.append(P)
    return V


def fibre_range(tables, fn, m=4000, rng=None):
    rng = rng or np.random.default_rng(0); V = np.array(fibre_vertices(tables)); vals = []
    with np.errstate(all="ignore"):
        for P in list(V) + list(rng.dirichlet(np.ones(len(V)) * 0.3, m) @ V):
            try: v = fn(P)
            except ZeroDivisionError: continue
            if np.isfinite(v): vals.append(float(v))
    return min(vals), max(vals)
