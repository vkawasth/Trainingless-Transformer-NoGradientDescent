"""Sign twists around a cycle of contexts (companion note, section on every cycle). New module; it modifies nothing.
  cycle_box(n, s)                the n-cycle box with uniform singles and perfect correlations <X_i X_{i+1}> = s_i (s_i = +-1): for each edge,
                                 the 2 x 2 table of (X_i, X_{i+1}) in {+1, -1}^2
  noncontextual_fraction(n, box, edges)
                                 the largest total weight of a sub-box made of deterministic assignments (LP); 1 means a compatible joint law
                                 exists, 0 means the box is strongly contextual
  mermin_solutions(row, col)     the +-1 assignments of a 3 x 3 square whose row products are `row` and column products `col`
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import linprog

V = (1, -1)


def cycle_box(n, s):
    return {i: {(a, b): (0.5 if a * b == s[i] else 0.0) for a in V for b in V} for i in range(n)}


def noncontextual_fraction(n, box, edges=None):
    edges = list(range(n)) if edges is None else list(edges); X = list(itertools.product(V, repeat=n)); rows, rhs = [], []
    for i in edges:
        j = (i + 1) % n
        for (a, b), p in box[i].items():
            rows.append([1.0 if (x[i] == a and x[j] == b) else 0.0 for x in X]); rhs.append(p)
    r = linprog(-np.ones(len(X)), A_ub=np.array(rows), b_ub=np.array(rhs), bounds=[(0, None)] * len(X), method="highs")
    return -r.fun


def mermin_solutions(row, col):
    out = []
    for v in itertools.product(V, repeat=9):
        m = np.array(v).reshape(3, 3)
        if tuple(np.prod(m, axis=1)) == tuple(row) and tuple(np.prod(m, axis=0)) == tuple(col): out.append(m)
    return out
