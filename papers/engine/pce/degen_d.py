"""Degenerate directions in several parameters (layers paper, Step 4f, Theorem "several parameters"). New module; pce/degen.py (d = 2)
is unchanged. Exact arithmetic (Fraction) on polynomials in d variables and on arcs with integer exponents.

  compose(cell, arc)            the univariate polynomial cell(x(t)) for an arc x_i(t) = sum_k a_ik t^k (dict exponent -> coefficient)
  lead(p)                       (order, leading coefficient) of a univariate polynomial
  arc_limit(cells, arc)         the conditioned state along the arc: normalised leading coefficients of the least-order cells
  truncate(arc, K)              the arc with every term of exponent > K removed
  initial_form(poly, w)         the part of a d-variable polynomial on the face minimising <w, alpha>
  faces(poly, weights)          the distinct leading faces seen over a set of positive weights (the bounded faces of the Newton polyhedron)
  degeneracy_gap(poly, face)    min over the positive torus of in_F P / (sum of |monomials|) on a grid in log coordinates (0 = degenerate)
"""
from __future__ import annotations
import itertools
from fractions import Fraction as Fr
import numpy as np


def _pmul(p, q):
    out = {}
    for a, x in p.items():
        for b, y in q.items(): out[a + b] = out.get(a + b, 0) + x * y
    return {k: v for k, v in out.items() if v != 0}


def _ppow(p, n):
    out = {0: Fr(1)}
    for _ in range(n): out = _pmul(out, p)
    return out


def compose(cell, arc):
    tot = {}
    for alpha, c in cell.items():
        term = {0: Fr(c)}
        for i, e in enumerate(alpha): term = _pmul(term, _ppow(arc[i], e))
        for k, v in term.items(): tot[k] = tot.get(k, 0) + v
    return {k: v for k, v in tot.items() if v != 0}


def lead(p):
    if not p: return None, Fr(0)
    k = min(p); return k, p[k]


def arc_limit(cells, arc):
    L = {c: lead(compose(f, arc)) for c, f in cells.items()}; orders = [o for o, _ in L.values() if o is not None]
    if not orders: return None, None
    m = min(orders); tot = sum(v for o, v in L.values() if o == m)
    return m, {c: (v / tot if o == m else Fr(0)) for c, (o, v) in L.items()}


def evaluate(cells, arc, t):
    vals = {c: sum(Fr(co) * np.prod([sum(Fr(a) * Fr(t) ** k for k, a in arc[i].items()) ** e for i, e in enumerate(al)], dtype=object)
                   for al, co in f.items()) for c, f in cells.items()}
    tot = sum(vals.values()); return {c: v / tot for c, v in vals.items()}


def truncate(arc, K): return [{k: a for k, a in x.items() if k <= K} for x in arc]


def initial_form(poly, w):
    m = min(sum(wi * ai for wi, ai in zip(w, a)) for a in poly)
    return {a: c for a, c in poly.items() if sum(wi * ai for wi, ai in zip(w, a)) == m}


def faces(poly, weights):
    return {tuple(sorted(initial_form(poly, w))) for w in weights}


def degeneracy_gap(poly, face, box=6.0):
    """numerical test for a zero of in_F P on the positive torus: minimise in_F P / (sum of |monomials|) over log coordinates in a box
    (one coordinate fixed by the torus action). A face with only positive coefficients returns 1 without search."""
    from scipy.optimize import minimize
    F = {a: float(c) for a, c in poly.items() if a in face}; d = len(next(iter(F)))
    if all(c > 0 for c in F.values()): return 1.0
    def r(u):
        x = np.exp(np.r_[0.0, np.clip(u, -box, box)]); mon = [c * np.prod(x ** np.array(a)) for a, c in F.items()]
        return sum(mon) / sum(abs(v) for v in mon)
    grid = np.linspace(-box, box, 25); best = min(itertools.product(grid, repeat=d - 1), key=lambda u: r(np.array(u)))
    res = minimize(r, np.array(best, float), method="Nelder-Mead", options=dict(xatol=1e-12, fatol=1e-14, maxiter=4000))
    return float(res.fun)
