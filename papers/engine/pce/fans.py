"""Step 4a-4b of "Probability, Context and Differentiation: The Layers": multi-parameter orders, the order fan, iterated limits. New
module. Exact arithmetic (Fraction).

A family depends on eps = (eps_1, ..., eps_d) -> 0 in the open orthant. A cell is a generalised polynomial {alpha: coefficient}, alpha a
tuple of non-negative Fractions; `cells` maps a cell key (l, x) on L x A to its polynomial.

  trop(f, w), initial(f, w)       min <w, alpha> over the support, and the monomials attaining it
  arc_germs(cells, w, c)          the one-parameter germs f(c_1 s^w_1, ..., c_d s^w_d) as power sums in s (for pce.orders)
  arc_limit(cells, w, c)          the limit of the conditioned state along that monomial arc (Theorem 32 of Step 3)
  newton_vertices_2d(cells)       vertices of the Newton polyhedron conv(union of supports) + R^2_{>=0} (d = 2), ordered by alpha_1
  componentwise_min(cells)        the exponent <= every other exponent in every coordinate, if it exists
  vertex_limit(cells, alpha)      normalised coefficients of eps^alpha over the cells
  iterated_limit(cells, order)    send eps_order[0] -> 0 first, then eps_order[1], ...: the normalised coefficients at the lexicographic minimum
  evaluate(cells, eps)            the conditioned state at a parameter value (exact or float)
"""
from __future__ import annotations
from fractions import Fraction as Fr
from . import orders as OD


def trop(f, w): return min(sum(Fr(wi) * ai for wi, ai in zip(w, a)) for a, v in f.items() if v != 0)


def initial(f, w):
    t = trop(f, w); return {a: v for a, v in f.items() if v != 0 and sum(Fr(wi) * ai for wi, ai in zip(w, a)) == t}


def arc_germs(cells, w, c):
    out = {}
    for k, f in cells.items():
        g = {}
        for a, v in f.items():
            if v == 0: continue
            e = sum(Fr(wi) * ai for wi, ai in zip(w, a)); coef = v
            for ci, ai in zip(c, a):
                if ai.denominator != 1: raise ValueError("use integer exponents for exact arcs")
                coef *= Fr(ci) ** int(ai)
            g[e] = g.get(e, Fr(0)) + coef
        out[k] = OD.clean(g)
    return out


def arc_limit(cells, w, c):
    germs = arc_germs(cells, w, c); r = OD.cond_limit({(k, "A"): g for k, g in germs.items()}, lambda x: True)
    return {k[0]: v for k, v in r["limit"].items()}


def evaluate(cells, eps):
    vals = {}
    for k, f in cells.items():
        tot = 0
        for a, v in f.items():
            term = v
            for e, ai in zip(eps, a): term = term * (e ** int(ai) if Fr(ai).denominator == 1 else float(e) ** float(ai))
            tot += term
        vals[k] = tot
    s = sum(vals.values()); return {k: v / s for k, v in vals.items()}


def all_exponents(cells): return sorted({a for f in cells.values() for a, v in f.items() if v != 0})


def componentwise_min(cells):
    ex = all_exponents(cells); m = tuple(min(a[i] for a in ex) for i in range(len(ex[0])))
    return m if m in ex else None


def newton_vertices_2d(cells):
    ex = all_exponents(cells)
    # minimal points (no other point <= in both coordinates), then the lower convex hull of the staircase
    mins = sorted({a for a in ex if not any(b != a and b[0] <= a[0] and b[1] <= a[1] for b in ex)})
    hull = []
    for p in mins:                                            # increasing alpha_1, decreasing alpha_2
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) <= 0: hull.pop()     # keep convex (lower-left) chain
            else: break
        hull.append(p)
    return hull


def vertex_limit(cells, alpha):
    co = {k: f.get(alpha, Fr(0)) for k, f in cells.items()}; co = {k: v for k, v in co.items() if v != 0}; s = sum(co.values())
    return {k: v / s for k, v in co.items()}


def iterated_limit(cells, order):
    ex = all_exponents(cells)
    for i in order:
        m = min(a[i] for a in ex); ex = [a for a in ex if a[i] == m]
    return vertex_limit(cells, ex[0]), ex[0]
