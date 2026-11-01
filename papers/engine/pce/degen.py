"""Step 4f of "Probability, Context and Differentiation: The Layers": the recursive refinement of degenerate initial forms (two parameters).
New module. Exact arithmetic (Fraction; sympy for the roots of edge polynomials).

Cells are polynomials in (x, y), dicts {(a, b): Fraction} with integer exponents, non-negative near 0 (for y > 0 at the top level, for y of
either sign at refined levels). An arc is y = sum_i c_i x^{e_i} (finite Puiseux polynomial, increasing positive exponents, c_1 != 0).

  total(cells)                       P_A = sum of the cells (the event's total mass)
  edges(f)                           the bounded edges of the lower-left Newton polygon of f: (slope w, (a1, b1), (a2, b2)) with w = (a2-a1)/(b1-b2)
  edge_poly(f, w)                    {b: coefficient} of the edge polynomial: the monomials of least weight a + w b, with x = 1
  degenerate(cells, top)             the pairs (w, c*, multiplicity): w an edge slope of P_A, c* a non-zero rational root of P_A's edge
                                     polynomial (positive at the top level); irrational real roots are reported separately
  substitute(f, w, cstar)            f(t^q, t^p (c* + y1)) with w = p/q, as a polynomial in (t, y1)
  arc_direct(cells, arc)             the limit of the conditioned state along the arc, by direct expansion (ground truth)
  arc_recursive(cells, arc, top)     the same limit by the recursion of the theorem: stop when the leading cells' edge polynomials do not all
                                     vanish at the arc's coefficient, otherwise substitute and continue with the arc's tail
  tree(cells, depth)                 the refinement tree, as a list of (path, multiplicity)
"""
from __future__ import annotations
from fractions import Fraction as Fr
from math import gcd, comb
import sympy as sp


def clean(f): return {k: Fr(v) for k, v in f.items() if v != 0}


def total(cells):
    out = {}
    for f in cells.values():
        for k, v in f.items(): out[k] = out.get(k, Fr(0)) + v
    return clean(out)


def _lower_hull(pts):
    mins = sorted({p for p in pts if not any(q != p and q[0] <= p[0] and q[1] <= p[1] for q in pts)})
    hull = []
    for p in mins:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) <= 0: hull.pop()
            else: break
        hull.append(p)
    return hull


def edges(f):
    h = _lower_hull(list(clean(f).keys()))
    return [(Fr(a2 - a1, b1 - b2), (a1, b1), (a2, b2)) for (a1, b1), (a2, b2) in zip(h, h[1:])]


def weight_min(f, w): return min(a + w * b for (a, b) in clean(f))


def edge_poly(f, w):
    mu = weight_min(f, w); return {b: v for (a, b), v in clean(f).items() if a + w * b == mu}


def _sympy_poly(ep):
    c = sp.Symbol("c"); return sp.Poly(sum(sp.Rational(v.numerator, v.denominator) * c ** b for b, v in ep.items()), c), c


def degenerate(cells, top=True):
    P = total(cells); out = []; irr = []
    for w, _, _ in edges(P):
        poly, c = _sympy_poly(edge_poly(P, w))
        for r, m in sp.roots(poly, multiple=False).items() if poly.degree() > 0 else []:
            if not r.is_real or r == 0: continue
            if top and not r > 0: continue
            if r.is_rational: out.append((w, Fr(int(r.p), int(r.q)), m))
            else: irr.append((w, r, m))
    return out, irr


def _pmul(f, g):
    out = {}
    for (a, b), v in f.items():
        for (c, d), u in g.items(): out[(a + c, b + d)] = out.get((a + c, b + d), Fr(0)) + v * u
    return clean(out)


def substitute(f, w, cstar):
    p, q = w.numerator, w.denominator; out = {}
    for (a, b), v in clean(f).items():
        base = q * a + p * b                                            # t exponent from x^a y^b
        for j in range(b + 1):                                           # (c* + y1)^b
            key = (base, j); out[key] = out.get(key, Fr(0)) + v * comb(b, j) * Fr(cstar) ** (b - j)
    return clean(out)


def _lcm(a, b): return a * b // gcd(a, b)


def arc_direct(cells, arc):
    Q = 1
    for e, _ in arc: Q = _lcm(Q, Fr(e).denominator)
    ypoly = {}
    for e, cf in arc: ypoly[int(Fr(e) * Q)] = ypoly.get(int(Fr(e) * Q), Fr(0)) + Fr(cf)
    lead = {}
    for k, f in cells.items():
        tot = {}
        for (a, b), v in clean(f).items():
            term = {Q * a: v}
            for _ in range(b):
                nt = {}
                for i, u in term.items():
                    for j, z in ypoly.items(): nt[i + j] = nt.get(i + j, Fr(0)) + u * z
                term = nt
            for i, u in term.items(): tot[i] = tot.get(i, Fr(0)) + u
        tot = {i: u for i, u in tot.items() if u != 0}
        if tot: m = min(tot); lead[k] = (m, tot[m])
    if not lead: return None
    m = min(o for o, _ in lead.values()); vals = {k: c for k, (o, c) in lead.items() if o == m}; s = sum(vals.values())
    return {k: v / s for k, v in vals.items()}


def arc_recursive(cells, arc, top=True, depth=0, trace=None):
    if trace is not None: trace.append(depth)
    if not arc:                                                         # exact branch: y1 = 0, a corner; send t -> 0 with y1 = 0
        lead = {}
        for k, f in cells.items():
            row = {a: v for (a, b), v in clean(f).items() if b == 0}
            if row: m = min(row); lead[k] = (m, row[m])
        if not lead: return None
        m = min(o for o, _ in lead.values()); vals = {k: c for k, (o, c) in lead.items() if o == m}; s = sum(vals.values())
        return {k: v / s for k, v in vals.items()}
    w, c = Fr(arc[0][0]), Fr(arc[0][1]); P = total(cells); mu = weight_min(P, w)
    vals = {}
    for k, f in cells.items():
        f = clean(f)
        if not f or weight_min(f, w) != mu: continue
        val = sum(v * c ** b for b, v in edge_poly(f, w).items())
        if val != 0: vals[k] = val
    if vals:
        s = sum(vals.values()); return {k: v / s for k, v in vals.items()}
    p, q = w.numerator, w.denominator
    tail = [(Fr(e) * q - p, cf) for e, cf in arc[1:]]                   # y1 = (y / t^p) - c*, in t = x^(1/q)
    tail = [(e / 1, cf) for e, cf in tail]
    new = {k: substitute(f, w, c) for k, f in cells.items()}
    return arc_recursive(new, tail, top=False, depth=depth + 1, trace=trace)


def tree(cells, depth=6, top=True, path=()):
    out = []
    if depth == 0: return out
    for w, cs, m in degenerate(cells, top)[0]:
        node = path + ((w, cs),); out.append((node, m))
        out += tree({k: substitute(f, w, cs) for k, f in cells.items()}, depth - 1, False, node)
    return out
