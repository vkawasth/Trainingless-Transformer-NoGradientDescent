"""The overlap algebra: for a family of windows, what joins, what does not, why, and how much each higher cell constrains the lower ones.

For every subfamily J of the windows the engine keeps
  * Phi(J): the fibre, the polytope of laws on the variables of J whose marginals are the windows of J (empty = J does not join);
  * lam*(J): the largest probability mass of J that joins.
For J inside K it keeps the projective overlap
  * pi_J Phi(K): the fillers of J that extend to K. Its dimension against dim Phi(J) says how much K constrains J.

  report(domains, windows)          every cell: filled or not, lam*, fibre dimension
  maximal_filled(cx)                the facets of the gluability complex: the largest subfamilies that join (what can join)
  minimal_unfilled(cx)              the minimal subfamilies that do not join (what did not join); their witnesses say why
  removal_sets(cx)                  the minimal sets of windows whose removal makes the rest join (complements of the facets)
  extension_dim(domains, windows, J, K)   dim pi_J Phi(K), and dim Phi(J)
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import linprog
from .coherence import Window, _constraints
from .integrate import gluable_fraction
from .complex import gluability_complex, filled, minimal_obstructions
from .cohomology import _res


def _vars(windows, J):
    return sorted({v for j in J for v in windows[j].vars})


def _affine(domains, windows, J, tol=1e-9):
    """support S (atoms of the relative interior), the atoms, the variable order, and a basis N of the directions of Phi(J)."""
    names = _vars(windows, J); dom = {v: domains[v] for v in names}
    A, t, atoms, _ = _constraints(dom, [windows[j] for j in J])
    n = A.shape[1]; Aeq = np.vstack([A, np.ones(n)]); beq = np.concatenate([t, [1.0]])
    S = []
    for k in range(n):
        e = np.zeros(n); e[k] = -1
        r = linprog(e, A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs")
        if r.status != 0: return None
        if -r.fun > tol: S.append(k)
    M = Aeq[:, S]
    u, s, vt = np.linalg.svd(M); rank = int((s > 1e-10).sum())
    N = vt[rank:].T                                                      # directions inside the affine hull, on the support
    return S, atoms, names, N


def report(domains, windows):
    cx = gluability_complex(domains, windows); out = {}
    for J, lam in cx.items():
        aff = _affine(domains, windows, J) if lam > 1 - 1e-7 else None
        out[J] = {"filled": lam > 1 - 1e-7, "lam": lam, "dim": None if aff is None else int(aff[3].shape[1])}
    return out, cx


def maximal_filled(cx, tol=1e-7):
    F = filled(cx, 0.0, tol)
    return sorted([J for J in F if not any(set(J) < set(K) for K in F)], key=lambda J: (-len(J), J))


def minimal_unfilled(cx, tol=1e-7):
    return minimal_obstructions(cx, tol)


def removal_sets(cx, n):
    return sorted([tuple(sorted(set(range(n)) - set(J))) for J in maximal_filled(cx)], key=lambda R: (len(R), R))


def extension_dim(domains, windows, J, K):
    """(dim pi_J Phi(K), dim Phi(J)); dim pi = -1 if Phi(K) is empty."""
    aJ = _affine(domains, windows, J)
    dJ = -1 if aJ is None else int(aJ[3].shape[1])
    aK = _affine(domains, windows, K)
    if aK is None: return -1, dJ
    S, atoms, namesK, N = aK
    namesJ = _vars(windows, J)
    P = _res(domains, tuple(namesK), tuple(namesJ))[:, S]                # marginal map from the support of Phi(K) to laws on J's variables
    return int(np.linalg.matrix_rank(P @ N, 1e-9)) if N.size else 0, dJ
