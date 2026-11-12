"""Imprecision without contextuality. New module; it modifies nothing.

Tie resolutions come from a global direction omega (one complex number per source). A resolution pattern assigns to each tied pair the
sector its difference falls in. Its realisable patterns form a set R; local data are laws on the patterns of each context (here: each pair,
or each pair of pairs). Local data are noncontextual relative to the geometry when some law on R has them as marginals (a linear
programme); otherwise no global law exists.

  realisable_patterns(fpairs, k, draws)  the set R of sign patterns of the pairwise functionals, sampled
  collinear_sign(omega, a, b)            1 if source a is resolved before b on the line (real parts), else 0
  orders(k)                              the k! total orders as pattern tuples over pairs (a < b)
  global_feasible(R, pairs, target)      is there a law on R whose single-pair marginals equal target[pair]? (LP)
  bounds(R, pairs, target, event)        lower and upper probability of an event (a set of patterns) over all such laws (LPs)
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import linprog


def orders(k):
    pairs = list(itertools.combinations(range(k), 2)); out = []
    for perm in itertools.permutations(range(k)):
        pos = {s: i for i, s in enumerate(perm)}; out.append(tuple(int(pos[a] < pos[b]) for a, b in pairs))
    return pairs, sorted(set(out))


def realisable_patterns(sector_fns, k, draws, rng):
    R = set()
    for _ in range(draws):
        om = rng.standard_normal(k) + 1j * rng.standard_normal(k); R.add(tuple(f(om) for f in sector_fns))
    return sorted(R)


def _marg_rows(R, pairs, idx):
    return np.array([[1.0 if r[idx] == 1 else 0.0 for r in R]])


def global_feasible(R, pairs, target):
    """target: dict pair -> P(pattern coordinate = 1)"""
    m = len(R); A = [np.ones(m)]; b = [1.0]
    for p, v in target.items():
        A.append(np.array([1.0 if r[pairs.index(p)] == 1 else 0.0 for r in R])); b.append(v)
    res = linprog(np.zeros(m), A_eq=np.array(A), b_eq=np.array(b), bounds=[(0, None)] * m, method="highs")
    return res.status == 0


def bounds(R, pairs, target, event):
    m = len(R); A = [np.ones(m)]; b = [1.0]
    for p, v in target.items():
        A.append(np.array([1.0 if r[pairs.index(p)] == 1 else 0.0 for r in R])); b.append(v)
    c = np.array([1.0 if r in event else 0.0 for r in R]); out = []
    for sgn in (1, -1):
        res = linprog(sgn * c, A_eq=np.array(A), b_eq=np.array(b), bounds=[(0, None)] * m, method="highs")
        if res.status != 0: return None
        out.append(sgn * res.fun)
    return tuple(out)
