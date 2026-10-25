"""Dimension-free quantum rung: positivity on truncated free modules (the NPA hierarchy) for bipartite, binary-outcome Bell scenarios.

Alice has observables A_x (x = 0..mA-1), Bob B_y (y = 0..mB-1), all with eigenvalues +-1 (A_x^2 = B_y^2 = 1) and [A_x, B_y] = 0. Words are
reduced to a normal form (Alice letters, then Bob letters, no repeated adjacent letter). A level is a list of words; the moment matrix
Gamma[u, v] = <u^* v> is real symmetric (enough for real-valued behaviours), positive semidefinite, with entries identified whenever the
reduced words u^* v agree. This is the GNS module of the state truncated to the words of the level: no dimension is ever fixed.

  level_words(mA, mB, level)   level in {"1", "1+AB", "2"}
  behaviour(c)                 p(a, b | x, y) from correlators: c["A"][x], c["B"][y], c["AB"][x][y]
  npa_value(coeffs, level)     max of sum_xy coeffs[x][y] <A_x B_y> over the level (e.g. CHSH: 2 sqrt 2)
  npa_fraction(P, level)       max lambda such that lambda q <= P cellwise with q in the level-k quantum set (a semidefinite program)
  local_fraction(P)            the classical lam* (the engine's linear program) for comparison
"""
from __future__ import annotations
import itertools
import numpy as np
import cvxpy as cp


def _reduce(word):
    a = [l for l in word if l[0] == "A"]; b = [l for l in word if l[0] == "B"]
    def red(seq):
        out = []
        for l in seq:
            if out and out[-1] == l: out.pop()
            else: out.append(l)
        return out
    return tuple(red(a) + red(b))


def _adj(word):
    return tuple(reversed(word))


def level_words(mA, mB, level):
    A = [("A", x) for x in range(mA)]; B = [("B", y) for y in range(mB)]
    W = [()] + [(a,) for a in A] + [(b,) for b in B]
    if level in ("1+AB", "2"): W += [(a, b) for a in A for b in B]
    if level == "2": W += [(a1, a2) for a1 in A for a2 in A if a1 != a2] + [(b1, b2) for b1 in B for b2 in B if b1 != b2]
    out = []
    for w in W:
        r = _reduce(w)
        if r not in out: out.append(r)
    return out


def _moment(words, scale=None):
    """symmetric moment matrix with entries tied by reduced words; returns (Gamma, value-of-word function, constraints)."""
    n = len(words); Gm = cp.Variable((n, n), symmetric=True); key = {}; cons = []
    for i in range(n):
        for j in range(i, n):
            w = _reduce(_adj(words[i]) + words[j]); w = min(w, _reduce(_adj(w)))          # real part: identify w with w^*
            if w in key: cons.append(Gm[i, j] == Gm[key[w]])
            else: key[w] = (i, j)
    cons.append(Gm >> 0)
    def val(w):
        w = _reduce(w); w = min(w, _reduce(_adj(w))); return Gm[key[w]] if w in key else None
    return Gm, val, cons


def behaviour(c):
    mA, mB = len(c["A"]), len(c["B"]); P = np.zeros((mA, mB, 2, 2))
    for x, y in itertools.product(range(mA), range(mB)):
        for ia, a in enumerate((1, -1)):
            for ib, b in enumerate((1, -1)):
                P[x, y, ia, ib] = (1 + a * c["A"][x] + b * c["B"][y] + a * b * c["AB"][x][y]) / 4
    return P


def npa_value(coeffs, level="1+AB"):
    mA, mB = len(coeffs), len(coeffs[0]); words = level_words(mA, mB, level); Gm, val, cons = _moment(words)
    cons.append(val(()) == 1); obj = sum(coeffs[x][y] * val((("A", x), ("B", y))) for x in range(mA) for y in range(mB))
    pr = cp.Problem(cp.Maximize(obj), cons); pr.solve(solver="SCS", eps=1e-9, max_iters=200000); return float(pr.value)


def npa_fraction(P, level="1+AB"):
    mA, mB = P.shape[0], P.shape[1]; words = level_words(mA, mB, level); Gm, val, cons = _moment(words); lam = val(())
    for x, y in itertools.product(range(mA), range(mB)):
        for ia, a in enumerate((1, -1)):
            for ib, b in enumerate((1, -1)):
                mu = (lam + a * val((("A", x),)) + b * val((("B", y),)) + a * b * val((("A", x), ("B", y)))) / 4
                cons += [mu <= P[x, y, ia, ib], mu >= 0]
    pr = cp.Problem(cp.Maximize(lam), cons); pr.solve(solver="SCS", eps=1e-9, max_iters=200000); return float(pr.value)


def local_fraction(P):
    from . import Window
    from .integrate import gluable_fraction
    mA, mB = P.shape[0], P.shape[1]; dom = {**{f"A{x}": 2 for x in range(mA)}, **{f"B{y}": 2 for y in range(mB)}}
    W = [Window((f"A{x}", f"B{y}"), P[x, y]) for x in range(mA) for y in range(mB)]
    return gluable_fraction(dom, W).lam
