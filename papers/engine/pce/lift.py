"""Closing the lift: one criterion per level for a global coherent lift, and the coupling simplicial set.

Level one. A family of windows lifts (has a global law) iff lam* = 1 (pce.integrate.gluable_fraction), iff the least negative mass
nu* of a signed lift is 0 (pce.cohomology.negativity); otherwise the LP witness certifies that no lift exists.

Level two. Each context i carries a finite law Pi_i over tables on its variables. A level-two lift is a finite law over global laws whose
pushforward to every context is Pi_i.
  lift2(domains, contexts)   returns the four answers in order:
     1. pairwise congruent at level two (pushforwards to every overlap agree)
     2. compatible on all higher intersections (automatic from 1: restriction is functorial -- checked, not assumed)
     3. lifts: there is a coupling x of the Pi_i supported on tuples of tables that glue at level one
        (max sum x subject to the marginal of x on context i being <= Pi_i: lam2* = 1 iff the coupling exists)
     4. obstructed: lam2* < 1, with the LP dual as certificate
     When it lifts, the lift is built explicitly: each coupled tuple is replaced by its maximum-entropy gluing.

The coupling simplicial set. Fix finite spaces X_0, ..., X_m with laws p_i. An n-simplex on (i_0 < ... < i_n) is a joint law on
X_{i_0} x ... x X_{i_n} whose one-dimensional marginals are the p's; the face d_k marginalises the k-th coordinate out and the
degeneracy s_k duplicates it diagonally.
  face(J, k), degeneracy(J, k)       the simplicial operators on a simplex given as (indices, array)
  compose(w01, w12)                  canonical filler of an inner 2-horn (conditional product) and its third face
  fill_horn(domains, faces)          a filler of a horn (any family of faces) or the witness that none exists
"""
from __future__ import annotations
import itertools
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linprog
from .coherence import Window
from .integrate import gluable_fraction, maxent_merge, _marg
from .complex import restrict2, congruent2, _restr, _key


# --------------------------------------------------------------------------------------------------------------- level two
def _canon(Pi, vars_):
    acc = {}
    for T, w in Pi:
        T = np.asarray(T, float); k = _key(T); acc[k] = (T, acc.get(k, (T, 0.0))[1] + w)
    return [(T, w) for T, w in acc.values() if w > 0]


@dataclass
class Lift2:
    pairwise: bool
    higher: bool
    lifts: bool
    lam2: float
    failing_overlaps: list
    certificate: dict | None = None
    lift: list | None = None          # [(global table over sorted variables, weight)]
    variables: tuple = ()
    details: dict = field(default_factory=dict)


def lift2(domains, contexts, tol=1e-9):
    ctx = [(tuple(v), _canon(Pi, v)) for v, Pi in contexts]
    n = len(ctx)
    fails = []
    for i, j in itertools.combinations(range(n), 2):
        keep = [v for v in ctx[i][0] if v in ctx[j][0]]
        if keep and not congruent2(ctx[i][1], ctx[i][0], ctx[j][1], ctx[j][0], keep): fails.append((i, j))
    higher = True
    for k in range(3, n + 1):
        for J in itertools.combinations(range(n), k):
            keep = [v for v in ctx[J[0]][0] if all(v in ctx[j][0] for j in J)]
            if not keep: continue
            ref = restrict2(ctx[J[0]][1], ctx[J[0]][0], keep)
            for j in J[1:]:
                other = restrict2(ctx[j][1], ctx[j][0], keep)
                higher &= set(ref) == set(other) and all(abs(ref[a][1] - other[a][1]) < tol for a in ref)
    names = sorted({v for vs, _ in ctx for v in vs}); dom = {v: domains[v] for v in names}
    sizes = [len(Pi) for _, Pi in ctx]
    tuples, glue_ok = [], []
    for tau in itertools.product(*[range(s) for s in sizes]):
        W = [Window(ctx[i][0], ctx[i][1][tau[i]][0]) for i in range(n)]
        if gluable_fraction(dom, W).lam > 1 - 1e-7: tuples.append(tau)
    if not tuples:
        return Lift2(not fails, higher, False, 0.0, fails, {"reason": "no tuple of tables glues at level one"}, None, tuple(names),
                     {"gluable_tuples": 0, "all_tuples": int(np.prod(sizes))})
    rows, rhs, idx = [], [], []
    for i in range(n):
        for s in range(sizes[i]):
            rows.append([1.0 if tau[i] == s else 0.0 for tau in tuples]); rhs.append(ctx[i][1][s][1]); idx.append((i, s))
    A = np.array(rows); b = np.array(rhs)
    r = linprog(-np.ones(len(tuples)), A_ub=A, b_ub=b, bounds=(0, None), method="highs")
    x = r.x; lam2 = float(x.sum())
    lifts = lam2 > 1 - 1e-7
    cert = None
    if not lifts:
        d = linprog(b, A_ub=-A.T, b_ub=-np.ones(len(tuples)), bounds=(0, None), method="highs")
        cert = {idx[k]: float(v) for k, v in enumerate(d.x) if abs(v) > 1e-12}
    lift = None
    if lifts:
        lift = []
        for tau, xv in zip(tuples, x):
            if xv <= 1e-12: continue
            W = [Window(ctx[i][0], ctx[i][1][tau[i]][0]) for i in range(n)]
            q, atoms, _, _ = maxent_merge(dom, W)
            T = np.zeros([dom[v] for v in names])
            for a, val in zip(atoms, q): T[a] = val
            lift.append((T, float(xv)))
    return Lift2(not fails, higher, lifts, min(lam2, 1.0), fails, cert, lift, tuple(names),
                 {"gluable_tuples": len(tuples), "all_tuples": int(np.prod(sizes))})


def check_lift2(lift, variables, contexts, tol=1e-7):
    """does the level-two lift restrict to every context's law over tables? (tables matched up to tol in the sup norm)"""
    ok = True
    for vs, Pi in contexts:
        got = [(m, w) for m, w in restrict2(lift, list(variables), list(vs)).values()]
        want = [(np.asarray(T, float), w) for T, w in _canon(Pi, vs)]
        used = [False] * len(got)
        for T, w in want:
            mass = 0.0
            for k, (m, wg) in enumerate(got):
                if not used[k] and np.abs(m - T).max() < tol: mass += wg; used[k] = True
            ok &= abs(mass - w) < 1e-7
        ok &= all(used)
    return ok


# ------------------------------------------------------------------------------------------------ the coupling simplicial set
def face(J, A, k):
    """d_k: marginalise out the k-th coordinate of the simplex (J, A)."""
    return J[:k] + J[k + 1:], A.sum(axis=k)


def degeneracy(J, A, k):
    """s_k: repeat the k-th coordinate, placing the copy right after it, on the diagonal."""
    n = A.ndim; shape = list(A.shape); shape.insert(k + 1, A.shape[k])
    B = np.zeros(shape)
    for idx in itertools.product(*[range(s) for s in A.shape]):
        j = list(idx); j.insert(k + 1, idx[k]); B[tuple(j)] = A[idx]
    return J[:k + 1] + J[k:k + 1] + J[k + 1:], B


def compose(w01, w12):
    """canonical filler of the inner horn (w01, w12): the conditional product, and its third face (w02)."""
    m1 = w01.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        K = np.where(m1[:, None] > 0, w12 / np.where(m1[:, None] > 0, m1[:, None], 1), 0.0)
    F = np.einsum("ab,bc->abc", w01, K)
    return F, F.sum(1)


def fill_horn(domains, faces):
    """faces: list of Window (the faces of a horn, or any family). Returns (filler table over sorted variables or None, Gluable)."""
    names = sorted(domains); dom = {v: domains[v] for v in names}
    g = gluable_fraction(dom, faces)
    if g.lam < 1 - 1e-7: return None, g
    q, atoms, _, _ = maxent_merge(dom, faces)
    T = np.zeros([domains[v] for v in names])
    for a, val in zip(atoms, q): T[a] = val
    return T, g
