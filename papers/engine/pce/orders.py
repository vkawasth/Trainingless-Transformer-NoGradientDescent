"""Step 3 of "Probability, Context and Differentiation: The Layers": the order of a probability transition. New module.

A germ at s = 0+ is a finite generalised power sum g(s) = sum_i c_i s^{a_i} (exponents Fractions >= 0, coefficients Fractions), stored as a
dict {exponent: coefficient}. A germ is non-negative if g(s) >= 0 for small s > 0, i.e. its leading coefficient is positive (or g = 0).

  clean(g), add(g, h), mul(g, h), scale(g, c)     arithmetic
  lead(g)                                          (order, leading coefficient); (inf, 0) for the zero germ
  ev(g, s)                                         value at s (exact if s and the exponents allow it, else float)
  pool_orders(w, ps)                               leading data of the labelled cells w_l(s) p_{l,s}(x) and of the pooled cells
  cond_limit(Lam, A)                               for a labelled path {(l, x): germ}: the order m of P_s(A), the limit of the labelled
                                                   conditional, the surviving sources, each source's extinction order m_l - m, and the
                                                   convergence exponent delta (the gap from m to the next exponent on L x A)
  jet_truncate(Lam, k)                             the k-jet of an analytic (integer-exponent) path: terms of exponent <= k
"""
from __future__ import annotations
from fractions import Fraction as Fr
import math

INF = float("inf")


def clean(g): return {Fr(a): Fr(c) for a, c in g.items() if c != 0}


def add(g, h):
    out = dict(clean(g))
    for a, c in clean(h).items(): out[a] = out.get(a, Fr(0)) + c
    return clean(out)


def mul(g, h):
    out = {}
    for a, c in clean(g).items():
        for b, d in clean(h).items(): out[a + b] = out.get(a + b, Fr(0)) + c * d
    return clean(out)


def scale(g, c): return clean({a: c * v for a, v in g.items()})


def lead(g):
    g = clean(g)
    if not g: return INF, Fr(0)
    a = min(g); return a, g[a]


def nonneg(g): return lead(g)[1] >= 0


def ev(g, s):
    tot = 0
    for a, c in clean(g).items():
        if Fr(a).denominator == 1 and isinstance(s, Fr): tot += c * s ** int(a)
        else: tot += float(c) * float(s) ** float(a)
    return tot


def total(Lam, pred):
    out = {}
    for k, g in Lam.items():
        if pred(k): out = add(out, g)
    return out


def cond_limit(Lam, A):
    """Lam: {(l, x): germ}; A: predicate on x"""
    onA = {k: clean(g) for k, g in Lam.items() if A(k[1]) and clean(g)}
    if not onA: return None
    m = min(lead(g)[0] for g in onA.values())
    lc = {k: lead(g)[1] for k, g in onA.items() if lead(g)[0] == m}; s = sum(lc.values())
    limit = {k: v / s for k, v in lc.items()}
    srcs = {k[0] for k in onA}; ml = {l: lead(total(onA, lambda k, l=l: k[0] == l))[0] for l in srcs}
    exps = sorted({a for g in onA.values() for a in g})
    delta = (exps[exps.index(m) + 1] - m) if exps.index(m) + 1 < len(exps) else INF
    return dict(m=m, limit=limit, survivors={l for l in ml if ml[l] == m}, extinction={l: ml[l] - m for l in ml}, delta=delta)


def jet_truncate(Lam, k): return {key: clean({a: c for a, c in g.items() if a <= k}) for key, g in Lam.items()}
