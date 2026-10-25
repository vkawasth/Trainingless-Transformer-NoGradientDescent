"""Coherence of local evidence: can window tables be glued into one global law?

Variables are finite (named, with given domain sizes). Evidence is a list of windows, each a tuple of variable names with a table
(a probability array over the product of their domains). The engine answers three questions:

  * distance to coherence (consistency radius): the smallest total-variation adjustment of the window tables that admits a global law;
  * if the radius is zero: is the global law unique, and how wide is the set of global laws (fibre width per atom);
  * if the radius is positive: a finite certificate -- a functional on window cells that is >= 0 for every deterministic assignment
    (hence for every global law) but negative on the observed tables (a Bell / cycle-type inequality, found by Farkas duality).

All computations are linear programs (scipy HiGHS).
"""
from __future__ import annotations
import itertools
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linprog


@dataclass
class Window:
    vars: tuple
    table: np.ndarray            # shape = (domain size of each var in vars), entries sum to 1


@dataclass
class CoherenceReport:
    status: str                  # "coherent-unique", "coherent-nonunique", "incoherent"
    radius: float                # consistency radius: average total-variation adjustment per window needed for a global law
    width: float                 # max over atoms of (max - min) probability across global laws (0 if unique); nan if incoherent
    certificate: dict | None = None   # {window index: array of coefficients}, value on data, and its guaranteed lower bound 0
    violation: float = 0.0       # how far the certificate is below 0 on the observed tables
    details: dict = field(default_factory=dict)


def _atoms(domains):
    names = list(domains)
    return names, list(itertools.product(*[range(domains[v]) for v in names]))


def _constraints(domains, windows):
    names, atoms = _atoms(domains)
    rows, rhs, index = [], [], []
    for w_i, w in enumerate(windows):
        pos = [names.index(v) for v in w.vars]
        for cell in itertools.product(*[range(domains[v]) for v in w.vars]):
            rows.append([1.0 if tuple(a[p] for p in pos) == cell else 0.0 for a in atoms])
            rhs.append(float(w.table[cell])); index.append((w_i, cell))
    return np.array(rows), np.array(rhs), atoms, index


def coherence(domains: dict, windows: list[Window], tol: float = 1e-9) -> CoherenceReport:
    A, b, atoms, index = _constraints(domains, windows)
    m, n = A.shape
    # 1. consistency radius: min sum |A x - b| over global laws x (x >= 0, sum x = 1); variables x, s+ , s-
    c = np.concatenate([np.zeros(n), np.ones(2 * m)])
    Aeq = np.hstack([A, -np.eye(m), np.eye(m)])
    Aeq = np.vstack([Aeq, np.concatenate([np.ones(n), np.zeros(2 * m)])])
    beq = np.concatenate([b, [1.0]])
    r = linprog(c, A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs")
    radius = 0.5 * float(r.fun) / max(1, len(windows))
    if r.fun > tol:
        # 2a. Farkas certificate: y with A^T y >= 0 (every atom, hence every global law) and b^T y < 0, normalised |y| <= 1
        res = linprog(b, A_ub=-A.T, b_ub=np.zeros(n), bounds=(-1, 1), method="highs")
        y = res.x
        cert = {}
        for (w_i, cell), coef in zip(index, y):
            cert.setdefault(w_i, np.zeros(windows[w_i].table.shape))[cell] = coef
        return CoherenceReport("incoherent", radius, float("nan"), cert, float(-(b @ y)),
                               {"min_over_atoms": float((A.T @ y).min())})
    # 2b. fibre width: for each atom, min and max of its probability over all global laws with these windows
    Aeq2 = np.vstack([A, np.ones(n)]); beq2 = np.concatenate([b, [1.0]])
    width = 0.0; lo_hi = []
    for k in range(n):
        e = np.zeros(n); e[k] = 1
        lo = linprog(e, A_eq=Aeq2, b_eq=beq2, bounds=(0, None), method="highs").fun
        hi = -linprog(-e, A_eq=Aeq2, b_eq=beq2, bounds=(0, None), method="highs").fun
        lo_hi.append((lo, hi)); width = max(width, hi - lo)
    status = "coherent-unique" if width <= 1e-7 else "coherent-nonunique"
    return CoherenceReport(status, radius, width, None, 0.0, {"atom_ranges": dict(zip(atoms, lo_hi))})


def pair_window(i, j, table):
    return Window((i, j), np.asarray(table, float))
