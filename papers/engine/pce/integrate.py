"""Integrating new expertise into a current context: structure, maximal coherent merge, what did not agree, the cost of bringing
it in, and irreconcilability.

Contexts are window families as in coherence.py: each window is a table over a few named finite variables. The current context
rho and the incoming expertise sigma are two such families over a common pool of variables.

  structure(domains, windows)        which variables are shared, pairwise disagreement on every overlap (local, total variation),
                                     and the cycle rank of the variable-window incidence graph (0 = a forest: local agreement
                                     then implies gluing)
  gluable_fraction(domains, windows) the largest mass lam* of a global law whose marginals fit under every window table
                                     (max 1'b, A b <= t, b >= 0), the residual windows (t - A b*)/(1 - lam*) that carry the
                                     disagreement, and a witness y >= 0 with A'y >= 1: every gluable family has <y, t> >= 1, the data
                                     have <y, t> = lam* (strong duality), so the violation is exactly 1 - lam*
  maxent_merge(domains, windows)     the maximum-entropy law among the largest gluable parts: the merge that adds no information
                                     beyond the windows, i.e. has the largest typical set (volume 2^{nH})
  split(windows, splits)             refinement: variable v in window w is given its own sense v@w
  min_splits(domains, windows, protected)    the fewest splits of unprotected shared variables that make the family glue
  irreconcilable(domains, windows, protected)  class-relative impossibility: no refinement by splits of unprotected variables
                                     glues iff the projections onto the protected variables do not glue; the witness then rules out
                                     every refinement in the class at once
  integrate(domains, current, incoming, protected)   all of the above for rho and sigma, with a verdict

All optimisation is linear programming (scipy HiGHS) except the maximum-entropy step (iterative proportional fitting when the
family glues; SLSQP on the polytope of largest gluable parts otherwise).
"""
from __future__ import annotations
import itertools
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linprog, minimize
from .coherence import Window, _constraints


# ------------------------------------------------------------------------------------------------------------------- structure
def _marg(w: Window, keep):
    axes = tuple(i for i, v in enumerate(w.vars) if v not in keep)
    order = [v for v in w.vars if v in keep]
    return order, w.table.sum(axis=axes)


def structure(domains, windows):
    shared = sorted({v for v in domains if sum(v in w.vars for w in windows) >= 2})
    mismatch = {}
    for i, j in itertools.combinations(range(len(windows)), 2):
        common = [v for v in windows[i].vars if v in windows[j].vars]
        if not common: continue
        oi, mi = _marg(windows[i], common); oj, mj = _marg(windows[j], common)
        mj = np.transpose(mj, [oj.index(v) for v in oi])
        mismatch[(i, j)] = (tuple(oi), 0.5 * float(np.abs(mi - mj).sum()))
    used = [v for v in domains if any(v in w.vars for w in windows)]
    E = sum(len(w.vars) for w in windows); V = len(used) + len(windows)
    parent = {x: x for x in [("v", v) for v in used] + [("w", k) for k in range(len(windows))]}
    def find(x):
        while parent[x] != x: x = parent[x]
        return x
    for k, w in enumerate(windows):
        for v in w.vars: parent[find(("v", v))] = find(("w", k))
    C = len({find(x) for x in parent})
    return {"shared": shared, "mismatch": mismatch, "cycle_rank": E - V + C}


# ---------------------------------------------------------------------------------------------------------- gluable fraction
@dataclass
class Gluable:
    lam: float                    # largest gluable mass lam*; 1 - lam* is the irreconcilable (contextual) fraction
    part: np.ndarray              # b*, a sub-probability on global assignments
    atoms: list
    residual: list                # residual windows (what did not agree), normalised; None when lam* = 1
    residual_mass: list           # per window: mass of the residual (always 1 - lam*: every window loses the same mass)
    witness: dict                 # {window: coefficients y >= 0}; <y, t> >= 1 for every gluable family
    index: list = field(default_factory=list)


def gluable_fraction(domains, windows, tol=1e-9) -> Gluable:
    A, t, atoms, index = _constraints(domains, windows)
    m, n = A.shape
    r = linprog(-np.ones(n), A_ub=A, b_ub=t, bounds=(0, None), method="highs")
    b = r.x; lam = float(b.sum())
    d = linprog(t, A_ub=-A.T, b_ub=-np.ones(n), bounds=(0, None), method="highs")      # dual: min t'y, A'y >= 1, y >= 0
    y = d.x
    wit = {}
    for (w_i, cell), coef in zip(index, y):
        wit.setdefault(w_i, np.zeros(windows[w_i].table.shape))[cell] = coef
    res, mass = None, []
    R = t - A @ b
    for k, w in enumerate(windows):
        rk = np.zeros(w.table.shape)
        for (w_i, cell), val in zip(index, R):
            if w_i == k: rk[cell] = max(val, 0.0)
        mass.append(float(rk.sum()))
        if lam < 1 - tol:
            res = res or []; res.append(rk / max(rk.sum(), 1e-300))
    return Gluable(min(lam, 1.0), b, atoms, res, mass, wit, index)


# ---------------------------------------------------------------------------------------------------------------- max entropy
def _H(q):
    q = np.asarray(q, float); q = q[q > 0]; return float(-(q * np.log2(q)).sum())


def _ipf(A, t, n, sweeps=20000, tol=1e-13, groups=None):
    q = np.full(n, 1.0 / n)
    for _ in range(sweeps):
        for rows in groups:
            mq = A[rows] @ q
            f = np.where(mq > 0, t[rows] / np.where(mq > 0, mq, 1), 0.0)
            q = q * (A[rows].T @ f)                # each atom lies in exactly one cell of each window
        if np.abs(A @ q - t).max() < tol: break
    return q


def maxent_merge(domains, windows, tol=1e-9):
    """returns (law over atoms, atoms, entropy in bits, lam*). For a gluable family this is the maximum-entropy gluing; otherwise the
    maximum-entropy law among the largest gluable parts (normalised)."""
    A, t, atoms, index = _constraints(domains, windows)
    n = A.shape[1]
    groups = [[r for r, (w_i, _) in enumerate(index) if w_i == k] for k in range(len(windows))]
    g = gluable_fraction(domains, windows)
    if g.lam > 1 - 1e-7:
        q = _ipf(A, t, n, groups=groups)
    else:
        lam = g.lam; x0 = g.part / lam
        f = lambda x: float(np.sum(np.where(x > 1e-15, x * np.log(np.maximum(x, 1e-15)), 0.0)))
        jac = lambda x: np.log(np.maximum(x, 1e-15)) + 1
        cons = [{"type": "ineq", "fun": lambda x: t / lam - A @ x, "jac": lambda x: -A},
                {"type": "eq", "fun": lambda x: x.sum() - 1, "jac": lambda x: np.ones((1, n))}]
        r = minimize(f, x0, jac=jac, constraints=cons, bounds=[(0, 1)] * n, method="SLSQP",
                     options={"maxiter": 500, "ftol": 1e-14})
        q = np.clip(r.x, 0, None); q = q / q.sum()
        if np.any(A @ q > t / lam + 1e-7): q = x0                      # fall back to the LP vertex if SLSQP drifted
    return q, atoms, _H(q), g.lam


# ------------------------------------------------------------------------------------------------------------------ refinement
def split(domains, windows, splits):
    """splits: iterable of (window index, variable). Variable v in window k becomes the new variable 'v@k'."""
    dom = dict(domains); new = []
    S = set(splits)
    for k, w in enumerate(windows):
        names = tuple(f"{v}@{k}" if (k, v) in S else v for v in w.vars)
        for v, nm in zip(w.vars, names): dom[nm] = domains[v]
        new.append(Window(names, w.table))
    used = {v for w in new for v in w.vars}
    return {v: d for v, d in dom.items() if v in used}, new


def glues(domains, windows, tol=1e-7):
    return gluable_fraction(domains, windows).lam > 1 - tol


def candidates(windows, protected):
    count = {}
    for w in windows:
        for v in w.vars: count[v] = count.get(v, 0) + 1
    return [(k, v) for k, w in enumerate(windows) for v in w.vars if v not in protected and count[v] >= 2]


def min_splits(domains, windows, protected=(), max_k=None):
    """the fewest splits of unprotected shared variables after which the family glues; None if irreconcilable."""
    sols = all_min_splits(domains, windows, protected, max_k)
    return None if sols is None else sols[0]


def all_min_splits(domains, windows, protected=(), max_k=None):
    """every minimal-size refinement that glues (the alternative repairs); None if irreconcilable."""
    cands = candidates(windows, protected)
    for k in range(0, (max_k if max_k is not None else len(cands)) + 1):
        sols = [list(S) for S in itertools.combinations(cands, k) if glues(*split(domains, windows, S))]
        if sols: return sols
    return None


def project(domains, windows, keep):
    out = []
    for w in windows:
        kv = [v for v in w.vars if v in keep]
        if kv:
            order, m = _marg(w, kv); out.append(Window(tuple(order), m))
    return {v: domains[v] for v in keep if any(v in w.vars for w in out)}, out


def irreconcilable(domains, windows, protected):
    """(is_irreconcilable, Gluable of the protected projection). Irreconcilable relative to the class of refinements that split
    unprotected variables iff the projections of the windows onto the protected variables do not glue."""
    d, w = project(domains, windows, set(protected))
    if not w: return False, None
    g = gluable_fraction(d, w)
    return g.lam < 1 - 1e-7, g


# ------------------------------------------------------------------------------------------------------------------- integrate
@dataclass
class Integration:
    verdict: str                  # redundant | complementary | repairable | irreconcilable
    kind: str                     # for repairable / irreconcilable: "local conflict" (type I) or "global obstruction" (type II)
    lam: float                    # gluable fraction of current + incoming
    structure: dict
    residual: list | None         # the part that did not agree: normalised residual windows (mass 1 - lam each); None if lam = 1
    repairs: list | None          # all minimal refinements
    splits: list | None           # minimal refinement (which shared terms need separate senses); [] if none needed
    bits_added: float | None      # H(merge of rho) - H(merge of rho and sigma), in the refined space; = KL (Pythagoras)
    state_bits: float             # log2 of the number of global assignments after refinement (state-size cost)
    merge: tuple | None           # (law, atoms) maximum-entropy merge after refinement
    witness: dict | None          # for irreconcilable: witness on the protected projection


def integrate(domains, current, incoming, protected=(), max_k=None):
    W = list(current) + list(incoming); nc = len(current)
    st = structure(domains, W)
    g = gluable_fraction(domains, W)
    irr, gp = irreconcilable(domains, W, protected)
    local = any(mm > 1e-9 for _, mm in st["mismatch"].values())
    kind = "local conflict" if local else "global obstruction"
    if irr:
        dp, wp = project(domains, W, set(protected))
        kind = "local conflict" if any(mm > 1e-9 for _, mm in structure(dp, wp)["mismatch"].values()) else "global obstruction"
        return Integration("irreconcilable", kind, g.lam, st, g.residual, None, None, None,
                           float(np.log2(np.prod(list(domains.values())))), None, gp.witness)
    R = [[]] if g.lam > 1 - 1e-7 else all_min_splits(domains, W, protected, max_k)
    S = R[0]
    d2, W2 = split(domains, W, S)
    q, atoms, Hm, _ = maxent_merge(d2, W2)
    q0, _, H0, _ = maxent_merge(d2, W2[:nc]) if nc else (None, None, float(np.log2(np.prod(list(d2.values())))), 1)
    bits = H0 - Hm
    verdict = "repairable" if S else ("redundant" if bits < 1e-6 else "complementary")
    return Integration(verdict, kind if S else "", g.lam, st, g.residual, R, S, bits,
                       float(np.log2(np.prod(list(d2.values())))), (q, atoms), None)
