"""Algebra of the probability operations, and selection that is not concave (companion note, section 12.24). New module; it modifies nothing.
  markov_entry(g1j, g2j, qj)      one term of Markov gluing, g1(i, j) g2(j, k) / q(j), extended by 0 at q(j) = 0 (removable singularity)
  ring_maxent(rho)                maximum entropy on the 4-cycle with uniform singles and every adjacent agreement probability (1 + rho)/2:
                                  the coupling t = tanh J solves rho (1 + t^4) = t + t^3, and P(A0 = A1) = (1 + 2 t^2 / (1 + t^4)) / 2
  lattice_count(V, xlaw, N)       the number of integer joint tables of size N on the readings V with the same marginal counts as N * xlaw
  min_entropy(d)                  the minimum-entropy law in the fibre (a concave objective minimised: attained at a vertex; it jumps)
"""
from __future__ import annotations
import itertools
from fractions import Fraction as Fr
import numpy as np
import sympy as sp
from . import margsec as G
from . import voigeom as V


def markov_entry(a, b, q):
    return Fr(0) if q == 0 else Fr(a) * Fr(b) / Fr(q)


def ring_maxent_exact(rho):
    """exact: returns (t as a sympy algebraic number, P(A0 = A1) simplified) for rational rho"""
    t = sp.symbols("t", positive=True); r = sp.Rational(rho)
    roots = [s for s in sp.solve(sp.Eq(r * (1 + t ** 4), t + t ** 3), t) if s.is_real and 0 < float(s) < 1]
    t0 = roots[0]; val = sp.nsimplify(sp.simplify((1 + 2 * t0 ** 2 / (1 + t0 ** 4)) / 2))
    return t0, sp.radsimp(sp.simplify(val))


def lattice_count(Vset, xlaw, N, tol=1e-9):
    """integer tables x >= 0 on the readings Vset with A x = N * (pi xlaw, 1). The fibre has dimension k = |V| - rank; k free cells are
    chosen so that the others are determined, and every integer value of the free cells in [0, N]^k is tried (vectorised)."""
    M = np.array(G.M, float); Vs = sorted(Vset); A = np.vstack([M[:, Vs], np.ones(len(Vs))]); b = N * np.r_[M @ xlaw, 1.0]
    r = np.linalg.matrix_rank(A); k = len(Vs) - r
    for free in itertools.combinations(range(len(Vs)), k):
        rest = [i for i in range(len(Vs)) if i not in free]
        if np.linalg.matrix_rank(A[:, rest]) == r: break
    Ar, Af = A[:, rest], A[:, list(free)]; pinv = np.linalg.pinv(Ar)
    grids = np.array(np.meshgrid(*[np.arange(N + 1)] * k, indexing="ij")).reshape(k, -1) if k else np.zeros((0, 1))
    xr = pinv @ (b[:, None] - Af @ grids)
    okc = np.all(np.abs(Ar @ xr + Af @ grids - b[:, None]) < 1e-6, axis=0) & np.all(np.abs(xr - np.round(xr)) < tol, axis=0) & np.all(xr > -tol, axis=0)
    return int(okc.sum()), k


def min_entropy(d):
    vs = V.fibre_vertices(d); hs = [-(v[v > 1e-12] * np.log(v[v > 1e-12])).sum() for v in vs]; order = np.argsort(hs)
    return vs[order[0]], hs[order[0]], (vs[order[1]], hs[order[1]]) if len(vs) > 1 else None
