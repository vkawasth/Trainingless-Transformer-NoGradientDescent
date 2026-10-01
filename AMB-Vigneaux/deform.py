"""Deformations of the toric model [T] at its boundary: T^1, T^2 and what they mean for supports (component [D]).

The no-top-interaction model on n binary measurements is the hypersurface X = V(f), f = prod_{chi=+1} p - prod_{chi=-1} p
(a binomial of degree 2^{n-1}; toric.py). Facts used and checked here:
  (1) Boundary supports. A zero set Z (cells that are impossible) occurs in the closure of the model iff Z meets both
      parity classes (an even-parity and an odd-parity cell). [checked: all 2^8 sets for n = 3; |Z| <= 4 for n = 4]
  (2) Singular locus. X is smooth at a boundary point with a even and b odd zeros iff min(a, b) = 1; it is singular iff
      a, b >= 2. Locally (other coordinates units) f ~ x_1...x_a - y_1...y_b.
  (3) T^2 = 0 everywhere: X is a hypersurface (a complete intersection), so every first-order deformation is
      unobstructed. The 'Ext^2 obstruction' of the deformation theory vanishes identically for these models.
  (4) T^1. At a generic point of the singular locus (a = b = 2) the transversal type is the 3-fold node (conifold)
      x1 x2 = y1 y2: Tjurina number 1, so T^1 is one-dimensional (the smoothing x1 x2 - y1 y2 = eps). Deeper strata
      (a + b > 4) are non-isolated; their T^1 is supported on the lower-dimensional singular strata.
  (5) The node has two small resolutions (the Atiyah flop), one for each perfect matching of the two vanishing even
      cells with the two vanishing odd cells. Statistically: near such a support, WHICH odd cell is forced together
      with a given even cell is not determined by the model; the AMB toggle picks the cheaper partner, and the choice
      flips under perturbations as small as the difference of the partners' masses.
  (6) The level sets of the top interaction, f_c = prod_+ p - e^c prod_- p, are related by the torus action (rescale one
      coordinate), so they are equisingular: changing theta ([I]) never smooths a node. The T^1 direction (an additive
      relaxation of the binomial) is not a log-linear (statistical) deformation.
"""
from __future__ import annotations

import itertools
from typing import Dict, List

import numpy as np

from .strata import design, is_facial
from . import toric as T


def parity_classes(n):
    x = T.chi(n)
    return [i for i in range(2 ** n) if x[i] > 0], [i for i in range(2 ** n) if x[i] < 0]


def boundary_supports(n, max_zeros=None) -> Dict[tuple, Dict[str, int]]:
    """count zero sets Z by (#even, #odd) zeros, split by whether the complement is facial (Z occurs on the boundary)"""
    A, _ = design((2,) * n, T.margins(n)); N = 2 ** n; ev, od = parity_classes(n); out = {}
    for k in range(1, (max_zeros or N) + 1):
        for Z in itertools.combinations(range(N), k):
            a = sum(z in ev for z in Z); key = (a, k - a)
            S = [j for j in range(N) if j not in Z]
            r = out.setdefault(key, dict(facial=0, nonfacial=0))
            r["facial" if is_facial(A, S) else "nonfacial"] += 1
    return out


def singular(a, b):
    return a >= 2 and b >= 2


def tjurina(a, b, c=0.0):
    """dimension of C[x, y]/(f, df) for f = x1..xa - e^c y1..yb (None if infinite, i.e. non-isolated), via Groebner"""
    import sympy as sp
    xs = sp.symbols(f"x1:{a + 1}"); ys = sp.symbols(f"y1:{b + 1}"); V = list(xs) + list(ys)
    f = sp.Mul(*xs) - sp.exp(sp.Rational(c)) * sp.Mul(*ys) if c else sp.Mul(*xs) - sp.Mul(*ys)
    I = [f] + [sp.diff(f, v) for v in V]
    G = sp.groebner(I, *V, order="grevlex")
    if any(g == 1 for g in G.exprs):
        return 0                                   # smooth: (f, df) is the unit ideal
    lead = [sp.Poly(g, *V).monoms(order="grevlex")[0] for g in G.exprs]
    # finite iff every variable has a pure power among the leading monomials
    for i in range(len(V)):
        if not any(all(m[j] == 0 for j in range(len(V)) if j != i) and m[i] > 0 for m in lead):
            return None
    # count standard monomials
    bound = max(max(m) for m in lead) + 1; cnt = 0
    for e in itertools.product(range(bound), repeat=len(V)):
        if not any(all(e[j] >= m[j] for j in range(len(V))) for m in lead):
            cnt += 1
    return cnt


def singular_locus_dim(a, b):
    """dimension of the singular locus of x1..xa = y1..yb (inside C^{a+b}): at least two x's and two y's vanish"""
    return (a - 2) + (b - 2) if singular(a, b) else -1


def radii(p) -> dict:
    """toggle radius of the smooth boundary (cheapest even+odd pair) and node radius (cheapest 2 even + 2 odd), with
    the matching margin at the node: how close the two pairings are (the flop ambiguity)"""
    n = int(np.log2(len(p))); ev, od = parity_classes(n)
    pe, po = sorted(ev, key=lambda i: p[i]), sorted(od, key=lambda i: p[i])
    pair = p[pe[0]] + p[po[0]]; node = p[pe[0]] + p[pe[1]] + p[po[0]] + p[po[1]]
    e1 = pe[0]; o1, o2 = po[0], po[1]
    return dict(toggle_mass=float(pair), toggle_radius=float(-np.log1p(-pair)), node_mass=float(node), node_radius=float(-np.log1p(-node)),
                node_cells=[T.cells(n)[i] for i in (pe[0], pe[1], po[0], po[1])],
                partner_margin=float(abs(p[o1] - p[o2]) / (p[o1] + p[o2])))


def flop_switch_rate(p, eps, reps=400, rng=None):
    """perturb p multiplicatively by exp(eps * noise); how often does the cheapest odd partner of the cheapest even
    cell change?"""
    rng = rng or np.random.default_rng(0); n = int(np.log2(len(p))); ev, od = parity_classes(n)
    base = min(od, key=lambda i: p[i]); sw = 0
    for _ in range(reps):
        q = p * np.exp(eps * rng.normal(size=len(p))); q /= q.sum()
        sw += min(od, key=lambda i: q[i]) != base
    return sw / reps
