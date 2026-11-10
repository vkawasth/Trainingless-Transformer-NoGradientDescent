"""Sufficient boundary statistics for gluing across a cut, and cut-invariant attribution (companion note, prop:boundary). New module; it
modifies nothing.
For a function f of resolutions e in {+1, -1}^n and a cut (S, S^c) of the crossings:
  classes(f, S, n)          the row classes of the cut: x_S ~ x'_S iff f(x_S, y) = f(x'_S, y) for every y (minimal sufficient statistic of
                            side S), and the column classes of S^c; returned as dicts assignment -> class index
  class_law(P, f, S, n)     the joint law of (T_S, T_Sc) under a law P on {+1, -1}^n (P a dict assignment -> mass)
  glue(P, S, n, stat)       the law obtained by gluing the two sides of P conditionally independently given a cut state b = stat(e)
                            recorded with each side: P(x_S, y_Sc) = sum_b P(x_S, b) P(y_Sc, b) / P(b)
  moebius(T)                for a dict arm -> value on two mechanisms {(0,0), (1,0), (0,1), (1,1)}: base, effects, interaction, Shapley
"""
from __future__ import annotations
import itertools
import numpy as np


def assignments(n): return list(itertools.product((1, -1), repeat=n))


def split(e, S):
    return tuple(e[i] for i in S), tuple(e[i] for i in range(len(e)) if i not in S)


def classes(f, S, n):
    Sc = [i for i in range(n) if i not in S]; xs = list(itertools.product((1, -1), repeat=len(S))); ys = list(itertools.product((1, -1), repeat=len(Sc)))
    def full(x, y):
        e = [0] * n
        for i, v in zip(S, x): e[i] = v
        for i, v in zip(Sc, y): e[i] = v
        return tuple(e)
    rows = {x: tuple(round(f(full(x, y)), 12) for y in ys) for x in xs}; cols = {y: tuple(round(f(full(x, y)), 12) for x in xs) for y in ys}
    rk = {r: i for i, r in enumerate(dict.fromkeys(rows.values()))}; ck = {c: i for i, c in enumerate(dict.fromkeys(cols.values()))}
    return {x: rk[rows[x]] for x in xs}, {y: ck[cols[y]] for y in ys}


def class_law(P, f, S, n):
    TS, TC = classes(f, S, n); out = {}
    for e, p in P.items():
        x, y = split(e, S); k = (TS[x], TC[y]); out[k] = out.get(k, 0.0) + p
    return out


def glue(P, S, n, stat):
    """glue the two sides conditionally independently given a cut state b = stat(e) recorded with each side:
    P(x, y) = sum_b P(x, b) P(y, b) / P(b)"""
    beta = stat; pb, pxb, pyb = {}, {}, {}
    for e, p in P.items():
        x, y = split(e, S); b = beta(e)
        pb[b] = pb.get(b, 0.0) + p; pxb[(x, b)] = pxb.get((x, b), 0.0) + p; pyb[(y, b)] = pyb.get((y, b), 0.0) + p
    out = {}
    for e in assignments(n):
        x, y = split(e, S); out[e] = sum(pxb.get((x, b), 0.0) * pyb.get((y, b), 0.0) / pb[b] for b in pb if pb[b] > 0)
    return out


def moebius(T):
    base = T[(0, 0)]; dC = T[(1, 0)] - base; dV = T[(0, 1)] - base; I = T[(1, 1)] - T[(1, 0)] - T[(0, 1)] + base
    return dict(base=base, context=dC, voi=dV, interaction=I, shapley_context=dC + I / 2, shapley_voi=dV + I / 2)
