"""The column-binding example, classified exactly (companion note, section 12.7). f = (2/5, 3/5), g = (1/2, 3/10, 1/5); facets 11, 12, 13, 22, 23;
with tau = T^(1/10),
    W = y1 + y2 + tau^4 / (y1 y2) + tau^3 / y2 + y1 y2 / tau^2 .
New module; it imports floermorph and modifies nothing.

  symbolic()                 sympy: the reduction y1 = s - tau^3 / s (s = y2), the quintic q(s) = s^5 + tau^2 s^4 - 2 tau^3 s^3 - 2 tau^5 s^2 + tau^8,
                             its discriminant, the Hessian resultant, W(s), the critical-value polynomial R(w) and its discriminant
  newton_polygon()           lower hull of the quintic's (degree, tau-valuation) points, the edge slopes and the edge polynomials
  branches(tau, dps)         the five critical points at a given tau (mpmath), as (y1, y2)
  exact_branches()           the exact leading data of the five branches: valuation in T, leading coefficients, leading critical value, leading det(Q/W)
  V, LAM                     the facet normals and exponents (in powers of T) for floermorph.objects
"""
from __future__ import annotations
from fractions import Fraction as Fr
import mpmath as mp
import sympy as sp

V = [(1, 0), (0, 1), (-1, -1), (0, -1), (1, 1)]; NAMES = ["11", "12", "13", "22", "23"]
LAM = [Fr(0), Fr(0), Fr(-2, 5), Fr(-3, 10), Fr(1, 5)]


def symbolic():
    t, s, y1, y2, w = sp.symbols("tau s y1 y2 w")
    W = y1 + y2 + t ** 4 / (y1 * y2) + t ** 3 / y2 + y1 * y2 / t ** 2
    diff = sp.simplify(sp.expand(y1 * sp.diff(W, y1) - y2 * sp.diff(W, y2)))          # = y1 - y2 + tau^3 / y2
    q = sp.expand(s ** 5 + t ** 2 * s ** 4 - 2 * t ** 3 * s ** 3 - 2 * t ** 5 * s ** 2 + t ** 8)
    E1 = sp.numer(sp.together((y1 * sp.diff(W, y1)).subs(y1, s - t ** 3 / s).subs(y2, s)))
    c = [y1, y2, t ** 4 / (y1 * y2), t ** 3 / y2, y1 * y2 / t ** 2]
    Q = sp.Matrix(2, 2, lambda i, j: sum(ck * vk[i] * vk[j] for ck, vk in zip(c, V)))
    detQ = sp.together(Q.det().subs(y1, s - t ** 3 / s).subs(y2, s)); Ws = sp.together(W.subs(y1, s - t ** 3 / s).subs(y2, s))
    Rw = sp.resultant(q, sp.expand(sp.numer(Ws) - w * sp.denom(Ws)), s)
    return dict(t=t, s=s, w=w, W=W, diff=diff, q=q, E1=sp.expand(E1), disc=sp.factor(sp.discriminant(q, s)),
                hess_res=sp.factor(sp.resultant(q, sp.numer(detQ), s)), detQ=detQ, Ws=Ws, Rw=sp.factor(Rw), disc_w=sp.factor(sp.discriminant(Rw, w)))


def newton_polygon():
    pts = [(5, 0), (4, 2), (3, 3), (2, 5), (0, 8)]                                  # (degree in s, tau-valuation of the coefficient)
    hull = [(0, 8)]
    while hull[-1][0] < 5:
        x0, y0 = hull[-1]; nxt = min((p for p in pts if p[0] > x0), key=lambda p: (Fr(p[1] - y0, p[0] - x0), -p[0])); hull.append(nxt)
    edges = [(a, b, -Fr(b[1] - a[1], b[0] - a[0])) for a, b in zip(hull, hull[1:])]   # root valuation = minus the slope
    return pts, hull, edges


def branches(tau, dps=400):
    mp.mp.dps = dps; tau = mp.mpf(tau)
    roots = mp.polyroots([1, tau ** 2, -2 * tau ** 3, -2 * tau ** 5, 0, tau ** 8], maxsteps=2000, extraprec=6 * dps)
    return [(s - tau ** 3 / s, s) for s in roots]


def exact_branches():
    """valuations in T (tau = T^(1/10)); y = coef * T^val to leading order"""
    sq2 = mp.sqrt(2); om = [mp.mpc(1), mp.exp(2j * mp.pi / 3), mp.exp(-2j * mp.pi / 3)]
    A = [dict(name=("positive" if e > 0 else "real (-,-)"), val=(Fr(3, 20), Fr(3, 20)), coef=(e * sq2 / 2, e * sq2), W_lead=f"2 tau {'+' if e > 0 else '-'} 2 sqrt2 tau^(3/2)",
              det_lead=(e * sq2, Fr(1, 2))) for e in (1, -1)]
    B = [dict(name=("real (-,+)" if k == 0 else "complex"), val=(Fr(2, 15), Fr(1, 6)), coef=(-mp.mpf(2) ** (mp.mpf(1) / 3) / o, mp.mpf(2) ** (-mp.mpf(1) / 3) * o),
              W_lead="-2 tau + 2^(2/3) omega tau^(5/3)", det_lead=(-3 * mp.mpf(2) ** (-mp.mpf(4) / 3) / o ** 2, Fr(2, 3)), omega=o) for k, o in enumerate(om)]
    return A + B
