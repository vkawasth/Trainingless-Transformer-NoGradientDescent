"""The context poset, congruence on overlaps, and the gluability complex (a probability cell complex).

Contexts are subsets of a universe X of finite variables, ordered by inclusion. A state assigns to each context a law on its variables
(level one), or a law over such laws (level two). The functions below make the intersection structure explicit and track, cell by cell,
what matches and what does not.

  poset(contexts)                    intersection closure of the contexts, ordered by inclusion, with its Hasse diagram
  nerve(contexts)                    subfamilies with a non-empty common intersection
  gluability_complex(domains, W)     for every subfamily J of the windows: lam*(J), the largest probability mass of a global law that fits
                                     under every window of J ("pick max probability"); J is a filled cell iff lam*(J) = 1
  filled(cx, t)                      cells with lam* >= 1 - t: a simplicial complex for every t (a filtration)
  minimal_obstructions(cx)           unfilled cells all of whose faces are filled: an obstruction of order |J|
  reduced_betti(cells)               reduced Betti numbers over GF(2) of a simplicial complex given by its cells
  fibre(domains, W)                  the fibre of global sections: dimension and the support of its relative interior
  triangle_type(...)                 for three pairwise couplings (2-cells): strict (the composite through one vertex equals the third),
                                     fillable (a joint law exists: a non-identity 3-cell, a point of the fibre), or hollow (no joint law)
  restrict2, congruent2, glue2       level two: laws over laws, their restriction, congruence on an overlap, and the gluing of two
                                     contexts congruent at level two
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import linprog
from .coherence import Window, _constraints
from .integrate import gluable_fraction, _marg


# --------------------------------------------------------------------------------------------------------------- the poset
def poset(contexts: dict):
    """contexts: {name: set of variables}. Returns (elements {name: frozenset}, order pairs (a, b) with a <= b, Hasse edges)."""
    el = {k: frozenset(v) for k, v in contexts.items()}
    changed = True
    while changed:
        changed = False
        for (a, A), (b, B) in itertools.combinations(list(el.items()), 2):
            I = A & B
            if I and I not in el.values():
                el[f"({a})&({b})"] = I; changed = True
    order = [(a, b) for a in el for b in el if el[a] <= el[b]]
    hasse = [(a, b) for a, b in order if a != b and el[a] != el[b]
             and not any(c not in (a, b) and el[a] < el[c] < el[b] for c in el)]
    return el, order, hasse


def nerve(contexts: dict):
    names = list(contexts)
    return [J for k in range(1, len(names) + 1) for J in itertools.combinations(names, k)
            if frozenset.intersection(*[frozenset(contexts[n]) for n in J])]


# ------------------------------------------------------------------------------------------------- the gluability complex
def _sub(domains, windows, J):
    W = [windows[j] for j in J]
    used = {v for w in W for v in w.vars}
    return {v: domains[v] for v in domains if v in used}, W


def gluability_complex(domains, windows):
    n = len(windows)
    return {J: gluable_fraction(*_sub(domains, windows, J)).lam
            for k in range(1, n + 1) for J in itertools.combinations(range(n), k)}


def filled(cx, t=0.0, tol=1e-7):
    return {J for J, lam in cx.items() if lam >= 1 - t - tol}


def minimal_obstructions(cx, tol=1e-7):
    F = filled(cx, 0.0, tol)
    return [(J, 1 - cx[J]) for J in cx if J not in F and all(S in F for S in itertools.combinations(J, len(J) - 1) if S)]


def _rank2(M):
    M = M.copy() % 2; r = 0; rows, cols = M.shape
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i, c]), None)
        if piv is None: continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c]: M[i] ^= M[r]
        r += 1
        if r == rows: break
    return r


def reduced_betti(cells):
    """reduced Betti numbers over GF(2); cells: iterable of tuples (closed under faces)."""
    cells = {tuple(sorted(c)) for c in cells}
    top = max(len(c) for c in cells) - 1
    by = {d: sorted(c for c in cells if len(c) == d + 1) for d in range(-1, top + 1)}
    by[-1] = [()]
    ranks = {-1: 0}
    for d in range(0, top + 1):                       # boundary from dimension d to d - 1
        rows, cols = by[d - 1], by[d]
        idx = {c: i for i, c in enumerate(rows)}
        M = np.zeros((len(rows), len(cols)), dtype=np.uint8)
        for j, c in enumerate(cols):
            for f in itertools.combinations(c, len(c) - 1): M[idx[f], j] = 1
        ranks[d] = _rank2(M) if M.size else 0
    ranks[top + 1] = 0
    return {d: len(by[d]) - ranks[d] - ranks[d + 1] for d in range(-1, top + 1)}


# -------------------------------------------------------------------------------------------------------------- the fibre
def fibre(domains, windows, tol=1e-9):
    """dimension of the polytope of global laws gluing the windows (None if empty), and the support of its relative interior."""
    A, t, atoms, _ = _constraints(domains, windows)
    n = A.shape[1]; Aeq = np.vstack([A, np.ones(n)]); beq = np.concatenate([t, [1.0]])
    supp = []
    for k in range(n):
        e = np.zeros(n); e[k] = -1
        r = linprog(e, A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs")
        if r.status != 0: return None, []
        if -r.fun > tol: supp.append(k)
    dim = len(supp) - np.linalg.matrix_rank(Aeq[:, supp]) if supp else 0
    return int(dim), [atoms[k] for k in supp]


# ----------------------------------------------------------------------------------- 2-cells, their composite, and 3-cells
def compose(w01: np.ndarray, w12: np.ndarray):
    """vertical composite of couplings (X0,X1) and (X1,X2): the (X0,X2) marginal of the conditional product through X1."""
    m1 = w01.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        K = np.where(m1[:, None] > 0, w12 / np.where(m1[:, None] > 0, m1[:, None], 1), 0.0)
    return w01 @ K


def triangle_type(w01, w12, w02, tol=1e-9):
    """'strict' if for some choice of middle vertex the composite of two sides equals the third; 'fillable' if the three admit a joint
    law (a 3-cell exists but is not the identity); 'hollow' if none."""
    if (np.abs(compose(w01, w12) - w02).max() < tol or np.abs(compose(w01.T, w02) - w12).max() < tol
            or np.abs(compose(w02.T, w01) - w12.T).max() < tol or np.abs(compose(w12.T, w01.T) - w02.T).max() < tol
            or np.abs(compose(w02, w12.T) - w01).max() < tol or np.abs(compose(w12, w02.T) - w01.T).max() < tol):
        return "strict"
    d = {"X0": w01.shape[0], "X1": w01.shape[1], "X2": w12.shape[1]}
    W = [Window(("X0", "X1"), w01), Window(("X1", "X2"), w12), Window(("X0", "X2"), w02)]
    return "fillable" if gluable_fraction(d, W).lam > 1 - 1e-7 else "hollow"


# ---------------------------------------------------------------------------------------------------------------- level two
def _restr(T, vars_, keep):
    order, m = _marg(Window(tuple(vars_), np.asarray(T, float)), keep)
    return np.transpose(m, [order.index(v) for v in keep])


def _key(m): return tuple(np.round(np.asarray(m).ravel(), 12))


def restrict2(Pi, vars_, keep):
    """Pi: list of (table over vars_, weight). Returns the pushforward on the variables keep, merged by equal tables."""
    acc = {}
    for T, w in Pi:
        m = _restr(T, vars_, keep); k = _key(m); acc[k] = (m, acc.get(k, (m, 0.0))[1] + w)
    return acc


def congruent2(PiC, varsC, PiD, varsD, keep):
    a, b = restrict2(PiC, varsC, keep), restrict2(PiD, varsD, keep)
    return set(a) == set(b) and all(abs(a[k][1] - b[k][1]) < 1e-9 for k in a)


def congruent1(PiC, varsC, PiD, varsD, keep):
    mc = sum(w * m for m, w in restrict2(PiC, varsC, keep).values())
    md = sum(w * m for m, w in restrict2(PiD, varsD, keep).values())
    return np.allclose(mc, md, atol=1e-12)


def _glue_tables(T1, v1, T2, v2):
    keep = [v for v in v1 if v in v2]; allv = list(v1) + [v for v in v2 if v not in v1]
    dims = {**{v: s for v, s in zip(v1, T1.shape)}, **{v: s for v, s in zip(v2, T2.shape)}}
    o, M = _marg(Window(tuple(v1), T1), keep); M = np.transpose(M, [o.index(v) for v in keep]) if keep else M
    Q = np.zeros([dims[v] for v in allv])
    for x in itertools.product(*[range(dims[v]) for v in allv]):
        a = dict(zip(allv, x)); m = M[tuple(a[v] for v in keep)] if keep else 1.0
        if m > 0: Q[x] = T1[tuple(a[v] for v in v1)] * T2[tuple(a[v] for v in v2)] / m
    return Q, allv


def glue2(PiC, varsC, PiD, varsD):
    """two contexts congruent at level two on their overlap: couple the two laws over laws along the overlap, glue each coupled pair
    by the conditional product. Returns (level-two law over the union, variables)."""
    keep = [v for v in varsC if v in varsD]
    out = []; allv = None
    for kc, (mc, wc) in restrict2(PiC, varsC, keep).items():
        A = [(T, w) for T, w in PiC if _key(_restr(T, varsC, keep)) == kc]
        B = [(T, w) for T, w in PiD if _key(_restr(T, varsD, keep)) == kc]
        for (T1, w1), (T2, w2) in itertools.product(A, B):
            Q, allv = _glue_tables(np.asarray(T1, float), list(varsC), np.asarray(T2, float), list(varsD))
            out.append((Q, w1 * w2 / wc))
    return out, allv
