"""Step 4c of "Probability, Context and Differentiation: The Layers": parameters as an object (the Para construction over source-indexed
kernels), reparametrisations, and decisions on the order fan. New module. Exact arithmetic (Fraction).

  compose(f, g)                     (Q, g) o (P, f): (p, q, l, x) -> sum_y f(p, l, x)(y) g(q, l, y); parameters concatenate, the source is held
  reparam(f, r)                     the 2-cell action: f o (r x id)
  monomial_subst(cells, R)          a cell polynomial in eps composed with the monomial map eps_i = prod_j delta_j^{R[i][j]}
  arc_coeffs(c_new, R)              the leading coefficients of eps along the image of the arc delta_j = c_new[j] s^{w_j}
  decision_germs(cells, U, w, c)    for each act a, the germ N_a(s) = sum_cells U[a][cell] f_cell(eps(s)) along the arc (power sums in s)
  eventual_acts(cells, U, w, c)     the acts optimal for all small s along the arc (decided by the signs of leading coefficients)
  limit_acts(cells, U, w, c)        the acts optimal for the limiting conditioned state (may tie where eventual_acts does not)
"""
from __future__ import annotations
from fractions import Fraction as Fr
from . import orders as OD
from . import fans as FN


def compose(f, g):
    def h(pq, l, x):
        p, q = pq; out = {}
        for y, v in f(p, l, x).items():
            for z, u in g(q, l, y).items(): out[z] = out.get(z, Fr(0)) + v * u
        return {z: v for z, v in out.items() if v != 0}
    return h


def reparam(f, r): return lambda p, l, x: f(r(p), l, x)


def monomial_subst(cells, R):
    d_new = len(R[0]); out = {}
    for k, f in cells.items():
        g = {}
        for a, v in f.items():
            b = tuple(sum(Fr(R[i][j]) * a[i] for i in range(len(a))) for j in range(d_new))
            g[b] = g.get(b, Fr(0)) + v
        out[k] = {b: v for b, v in g.items() if v != 0}
    return out


def arc_coeffs(c_new, R):
    out = []
    for i in range(len(R)):
        v = Fr(1)
        for j in range(len(c_new)): v *= Fr(c_new[j]) ** int(R[i][j])
        out.append(v)
    return out


def decision_germs(cells, U, w, c):
    germs = FN.arc_germs(cells, w, c); out = {}
    for a, u in U.items():
        g = {}
        for k, gk in germs.items(): g = OD.add(g, OD.scale(gk, Fr(u.get(k, 0))))
        out[a] = g
    return out


def eventual_acts(cells, U, w, c):
    N = decision_germs(cells, U, w, c); best = []
    for a in U:
        if all(OD.lead(OD.add(N[a], OD.scale(N[b], Fr(-1))))[1] >= 0 for b in U if b != a): best.append(a)
    return set(best)


def limit_acts(cells, U, w, c):
    lim = FN.arc_limit(cells, w, c); val = {a: sum(Fr(u.get(k, 0)) * v for k, v in lim.items()) for a, u in U.items()}
    m = max(val.values()); return {a for a, v in val.items() if v == m}
