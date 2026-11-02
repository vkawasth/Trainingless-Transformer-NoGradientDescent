"""The tower over parameter space: fractures and blind spots (layers paper, Section "Fractures and blind spots"). New module. Exact
arithmetic (Fraction) on germs from pce.orders.

  poison_wins(G, E)                    the clean decision gap G(s) and the poison's effect E(s) as germs; the poisoned gap G + E has the sign of
                                       the lower-order term (sum of leading coefficients on a tie of orders); returns (flips, order of G,
                                       order of E)
  cov(P)                               the covariance of two binary variables, P on cells (00, 01, 10, 11): P11 - (P10 + P11)(P01 + P11)
  cov_parts(P, U)                      Cov(P + e U) = Cov(P) + e D(U) + e^2 Q(U): returns (D(U), Q(U)) exactly
  limit_set_sample(cells, weights, cs) arc limits over a grid of weights and coefficients (pce.fans): a singleton means no order-0 fracture
  pairwise_glues(eps)                  the three-token example of the main paper: a joint law with all pairwise agreements eps exists iff eps >= 1/3
"""
from __future__ import annotations
from fractions import Fraction as Fr
from . import orders as OD
from . import fans as FN


def poison_wins(G, E):
    (mg, cg), (me, ce) = OD.lead(G), OD.lead(E); tot = OD.add(G, E)
    if not tot: return True, mg, me                                    # an exact tie created: the act is no longer strictly preferred
    m, c = OD.lead(tot); return (c > 0) != (cg > 0), mg, me


def cov(P): return P[3] - (P[2] + P[3]) * (P[1] + P[3])


def cov_parts(P, U):
    a = P[2] + P[3]; b = P[1] + P[3]; ua = U[2] + U[3]; ub = U[1] + U[3]
    return U[3] - a * ub - ua * b, -ua * ub


def limit_set_sample(cells, weights, cs):
    out = set()
    for w in weights:
        for c in cs: out.add(tuple(sorted(FN.arc_limit(cells, w, c).items())))
    return out


def pairwise_glues(eps):
    """three binary tokens, every pair agreeing with probability eps (uniform margins): the 3-cycle inequality Phi = 3(1 - eps) <= 2"""
    return 3 * (1 - Fr(eps)) <= 2


# ------------------------------------------------------------------------------------------------ fragility, reach, observational fractures
def flip_share_line(F, b, u, lo=0.0, hi=1.0, tol=1e-14):
    """the least share e in (0, hi] with F(b + e u) <= 0 when F(b) > 0 (bisection on the first sign change found on a fine grid)"""
    import numpy as np
    grid = np.geomspace(1e-12, hi, 4000); prev = 0.0
    for e in grid:
        if F(b + e * u) <= 0:
            a, c = prev, e
            while c - a > tol * max(1.0, c):
                mid = (a + c) / 2
                if F(b + mid * u) <= 0: c = mid
                else: a = mid
            return c
        prev = e
    return None


def race_outcomes(c):
    """on the ray t = c s of the race: limiting Bayes weight of source 1, the audit act, and the value of learning the source label"""
    p1 = Fr(c) / (1 + Fr(c)); act = "audit 1" if p1 > Fr(1, 2) else ("audit 2" if p1 < Fr(1, 2) else "tie")
    return p1, act, 1 - max(p1, 1 - p1)
