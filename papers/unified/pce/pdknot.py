"""Pseudo-diagrams given by planar-diagram (PD) codes: resolutions, Jones polynomials, knot types and Walsh supports. New module; it
modifies nothing (pce/pseudoknot.py treats the 2-strand closures; this one treats any PD code).
PD convention (KnotAtlas): X[a, b, c, d] lists the four edge labels counterclockwise starting from the incoming under-strand; edge labels
increase along the orientation (mod 2n). The crossing is positive iff the over-strand runs d -> b. The bracket rule is
<X[a,b,c,d]> = A <P[a,b] P[c,d]> + A^-1 <P[a,d] P[b,c]>; changing the crossing swaps A and A^-1 and flips its sign.
  sign(x, m)                     the sign of crossing x (2n = m edge labels)
  bracket(pd, flips)             Kauffman bracket (normalised so the unknot is 1) with the crossings in `flips` changed
  jones(pd, flips)               V = (-A^3)^(-w) <K>, t = A^-4
  resolve(pd, e)                 the resolved diagram for crossing signs e (e_i = +-1): flips = {i : e_i != sign_i}
  KNOWN                          Jones polynomials of the unknot, both trefoils and the figure-eight
  classify(V)                    the name of V in KNOWN, or None
  walsh(f, n)                    Walsh coefficients over {+1, -1}^n
"""
from __future__ import annotations
import itertools
import numpy as np
import sympy as sp

A = sp.Symbol("A"); t = sp.Symbol("t")
FIG8 = [(4, 2, 5, 1), (8, 6, 1, 5), (6, 3, 7, 4), (2, 7, 3, 8)]
TREFOIL = [(1, 5, 2, 4), (3, 1, 4, 6), (5, 3, 6, 2)]


def _succ(a, b, m):
    return (b - a) % m == 1


def sign(x, m):
    a, b, c, d = x
    if _succ(d, b, m): return 1
    if _succ(b, d, m): return -1
    raise ValueError("over-strand labels not consecutive")


def _loops(pairs, labels):
    parent = {l: l for l in labels}
    def find(u):
        while parent[u] != u: parent[u] = parent[parent[u]]; u = parent[u]
        return u
    for u, v in pairs: parent[find(u)] = find(v)
    return len(set(find(l) for l in labels))


def bracket(pd, flips=()):
    labels = sorted(set(l for x in pd for l in x)); d = -A ** 2 - A ** -2; total = sp.Integer(0)
    for state in itertools.product((0, 1), repeat=len(pd)):           # 0: the A-smoothing of the (possibly changed) crossing
        pairs, power = [], 0
        for i, (x, s) in enumerate(zip(pd, state)):
            a, b, c, dd = x; Apair = [(a, b), (c, dd)]; Bpair = [(a, dd), (b, c)]
            if i in flips: Apair, Bpair = Bpair, Apair
            pairs += Apair if s == 0 else Bpair; power += 1 if s == 0 else -1
        total += A ** power * d ** (_loops(pairs, labels) - 1)
    return sp.expand(total)


def jones(pd, flips=()):
    m = 2 * len(pd); w = sum(-sign(x, m) if i in flips else sign(x, m) for i, x in enumerate(pd))
    V = sp.expand(sp.simplify((-A ** 3) ** (-w) * bracket(pd, flips)))
    return sp.expand(V.subs(A, t ** sp.Rational(-1, 4)))


def resolve(pd, e):
    m = 2 * len(pd); return {i for i, x in enumerate(pd) if e[i] != sign(x, m)}


KNOWN = {"unknot": sp.Integer(1), "right trefoil": t + t ** 3 - t ** 4, "left trefoil": 1 / t + t ** -3 - t ** -4,
         "figure-eight": t ** 2 - t + 1 - 1 / t + t ** -2}


def classify(V):
    for k, v in KNOWN.items():
        if sp.simplify(V - v) == 0: return k
    return None


def walsh(f, n):
    R = list(itertools.product((1, -1), repeat=n)); out = {}
    for k in range(n + 1):
        for S in itertools.combinations(range(n), k):
            out[S] = sum(f(e) * np.prod([e[i] for i in S]) for e in R) / 2 ** n
    return out
