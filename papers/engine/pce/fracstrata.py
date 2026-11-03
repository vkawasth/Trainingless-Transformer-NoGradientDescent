"""One stratification for all fractures (layers paper, Section "Fracture strata"). New module. Exact arithmetic (Fraction) on the
blueprint's flag world over the closed quadrant (s, t) >= 0 near 0.

  cells(s, t)                     the four labelled flag cells (source, mass) and their gaps (pce.blueprint.FLAG_CELLS)
  invariants(s, t)                (support pattern J, sign of N_ab, sign of the value of learning the source label)
  locus(kind, b, h)               True if the invariant `kind` is not locally constant at b (it takes another value within distance h)
  eps_act(b)                      the act's fragility over all perturbations of the parameters: the distance to its wall t = s/2
  reach(b, h)                     the set of kinds whose fracture locus contains b
"""
from __future__ import annotations
from fractions import Fraction as Fr
import math

KINDS = ("definedness", "act", "value of information")


def cells(s, t):
    s, t = Fr(s), Fr(t)
    return [(1, s ** 3, 1), (1, Fr(7, 4) * s * s * t, -1), (2, t * s, 0), (2, t ** 3, -1)]   # (source, mass on A, gap u_a - u_b)


def _sign(x): return (x > 0) - (x < 0)


def invariants(s, t):
    C = cells(s, t); J = frozenset(l for l, m, g in C if m > 0)
    N = sum(m * g for l, m, g in C)
    G = {l: sum(m * g for l2, m, g in C if l2 == l) for l in (1, 2)}         # each source's gap mass
    V = sum(max(Fr(0), G[l]) for l in G) - max(Fr(0), sum(G.values()))      # value of learning the label, times P(A)
    return {"definedness": J, "act": _sign(N) if J else None, "value of information": _sign(V) if J else None}


def _neighbours(b, h, k=24):
    s, t = b; out = []
    for i in range(k):
        a = 2 * math.pi * i / k; ds, dt = Fr(math.cos(a)).limit_denominator(10 ** 6) * h, Fr(math.sin(a)).limit_denominator(10 ** 6) * h
        if s + ds >= 0 and t + dt >= 0: out.append((s + ds, t + dt))
    for f in (Fr(1, 3), Fr(2, 3)):
        for q in list(out): out.append((s + (q[0] - s) * f, t + (q[1] - t) * f))
    return out


def locus(kind, b, h=Fr(1, 10 ** 9)):
    v = invariants(*b)[kind]; return any(invariants(*q)[kind] != v for q in _neighbours(b, h))


def reach(b, h=Fr(1, 10 ** 9)): return frozenset(k for k in KINDS if locus(k, b, h))


def eps_act(b):
    """distance from b to the act's wall t = s/2 (both coordinates free; the wall lies inside the quadrant)"""
    s, t = map(float, b); return abs(t - s / 2) / math.sqrt(1.25)
