"""Cone-by-cone statements (layers paper: Lemma "the spread order" and Theorem "poison order against tie order on the fan"). New module.
Exact arithmetic on polynomials in d variables along arcs with integer exponents (pce.degen_d supplies composition).

  spread_order(cells, gaps, w)     the order rho of the conditional variance of the gap along the weight w, from fan data (Lemma (2)-(3))
  spread_exact(cells, gaps, arc, t) the conditional variance of the gap at t, exactly
  trop(poly, w)                    min <w, alpha> over the support
  init_val(poly, w, c)             in_w poly evaluated at the arc coefficient c
  poisoned_sign(G, E, w, c)        the predicted sign of G + E along an arc with leading term c t^w (None if the initial forms cancel)
  critical_share(G, H, w, k)       the fragility exponent p*(w) = (trop_w G - trop_w H) / k: a share e t^p flips the act iff p < p* (opposite signs)
"""
from __future__ import annotations
from fractions import Fraction as Fr
from . import degen_d as DD


def trop(poly, w): return min(sum(Fr(wi) * ai for wi, ai in zip(w, a)) for a in poly)


def init_val(poly, w, c):
    m = trop(poly, w); tot = Fr(0)
    for a, co in poly.items():
        if sum(Fr(wi) * ai for wi, ai in zip(w, a)) == m:
            term = Fr(co)
            for ci, ai in zip(c, a): term *= Fr(ci) ** ai
            tot += term
    return tot


def _cell_order_coef(f, w, c):
    return trop(f, w), init_val(f, w, c)


def spread_order(cells, gaps, w, c):
    """rho from fan data. Leading cells: least trop. If the leading cells (with non-zero initial value) carry two different gaps,
    rho = 0. Otherwise, with g0 their common gap, rho = min over cells with gap != g0 of (trop_w f - v(P)); infinity if none."""
    oc = {k: _cell_order_coef(f, w, c) for k, f in cells.items()}
    vP = min(o for o, v in oc.values() if v != 0)
    lead_gaps = {gaps[k] for k, (o, v) in oc.items() if o == vP and v != 0}
    if len(lead_gaps) > 1: return Fr(0)
    g0 = next(iter(lead_gaps)); rest = [o - vP for k, (o, v) in oc.items() if gaps[k] != g0 and v != 0]
    return min(rest) if rest else None


def spread_exact(cells, gaps, arc, t):
    vals = {k: sum(Fr(co) * _prod(arc, a, t) for a, co in f.items()) for k, f in cells.items()}
    P = sum(vals.values()); D = sum(gaps[k] * v for k, v in vals.items()) / P
    return sum(v * (gaps[k] - D) ** 2 for k, v in vals.items()) / P


def _prod(arc, a, t):
    out = Fr(1)
    for i, e in enumerate(a):
        xi = sum(Fr(co) * Fr(t) ** k for k, co in arc[i].items()); out *= xi ** e
    return out


def poisoned_sign(G, E, w, c):
    tg, te = trop(G, w), trop(E, w); ig, ie = init_val(G, w, c), init_val(E, w, c)
    if ig == 0 or ie == 0: return None
    if tg < te: return 1 if ig > 0 else -1
    if te < tg: return 1 if ie > 0 else -1
    s = ig + ie; return None if s == 0 else (1 if s > 0 else -1)


def critical_share(G, H, w, k=1): return (trop(G, w) - trop(H, w)) / k
