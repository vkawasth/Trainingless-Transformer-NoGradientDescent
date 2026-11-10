"""Cutting a pseudo-diagram between two regions and gluing at a cut state (companion note, prop:cutglue). New module; it modifies nothing
and uses pce/pdknot.py.
Variables: crossings e1, e2 (twist region 1), e3, e4 (twist region 2) of the figure-eight pseudo-diagram, and a cut state y in {0, 1}.
Observed tables: (e1, e2, y) and (y, e3, e4). Cells are indexed by (e1, e2, e3, e4, y), crossings +1 first.
  CELLS                         the 32 cells
  tables(x)                     the two observed tables of a joint law x on CELLS
  interval(f, x)                [min, max] of E f over joint laws with the tables of x (LP)
  frechet(x)                    sum_y q(y) [max(0, a_y + b_y - 1), min(a_y, b_y)], with a_y, b_y the probabilities that regions 1, 2 agree given y
  markov(x)                     the Markov glue: P(e1, e2, y) P(e3, e4 | y), and its P(knotted) = sum_y q(y) a_y b_y
  ipf_maxent(x)                 the maximum-entropy law with the two tables (iterative proportional fitting from uniform)
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import linprog

CELLS = [e + (y,) for e in itertools.product((1, -1), repeat=4) for y in (0, 1)]
KNOT = np.array([1.0 if (c[0] == c[1] and c[2] == c[3]) else 0.0 for c in CELLS])        # 1[knotted] for the figure-eight diagram
CTX = [(0, 1, 4), (4, 2, 3)]


def _rows():
    rows = [np.ones(32)]
    for ctx in CTX:
        for v in itertools.product(*[((1, -1) if i < 4 else (0, 1)) for i in ctx]):
            rows.append(np.array([1.0 if all(c[i] == vv for i, vv in zip(ctx, v)) else 0.0 for c in CELLS]))
    return np.array(rows)


ROWS = _rows()


def tables(x): return ROWS @ np.asarray(x, float)


def interval(f, x):
    b = tables(x); f = np.asarray(f, float)
    lo = linprog(f, A_eq=ROWS, b_eq=b, bounds=[(0, None)] * 32, method="highs"); hi = linprog(-f, A_eq=ROWS, b_eq=b, bounds=[(0, None)] * 32, method="highs")
    return lo.fun, -hi.fun


def _cond(x):
    x = np.asarray(x, float); out = {}
    for y in (0, 1):
        q = sum(x[i] for i, c in enumerate(CELLS) if c[4] == y)
        a = sum(x[i] for i, c in enumerate(CELLS) if c[4] == y and c[0] == c[1]) / q
        b = sum(x[i] for i, c in enumerate(CELLS) if c[4] == y and c[2] == c[3]) / q
        out[y] = (q, a, b)
    return out


def frechet(x):
    cd = _cond(x); return (sum(q * max(0.0, a + b - 1) for q, a, b in cd.values()), sum(q * min(a, b) for q, a, b in cd.values()))


def markov(x):
    x = np.asarray(x, float); m = np.zeros(32)
    for i, c in enumerate(CELLS):
        y = c[4]; qy = sum(x[j] for j, d in enumerate(CELLS) if d[4] == y)
        p12y = sum(x[j] for j, d in enumerate(CELLS) if d[0] == c[0] and d[1] == c[1] and d[4] == y)
        p34y = sum(x[j] for j, d in enumerate(CELLS) if d[2] == c[2] and d[3] == c[3] and d[4] == y)
        m[i] = p12y * p34y / qy
    cd = _cond(x); return m, sum(q * a * b for q, a, b in cd.values())


def ipf_maxent(x, iters=5000, tol=1e-14):
    b = tables(x); z = np.full(32, 1 / 32)
    groups = []
    for ctx in CTX:
        for v in itertools.product(*[((1, -1) if i < 4 else (0, 1)) for i in ctx]):
            groups.append((np.array([all(c[i] == vv for i, vv in zip(ctx, v)) for c in CELLS]), None))
    targets = ROWS[1:] @ np.asarray(x, float)
    for _ in range(iters):
        err = 0.0
        for (m, _), tgt in zip(groups, targets):
            cur = z[m].sum(); err = max(err, abs(cur - tgt))
            if cur > 0: z[m] *= tgt / cur
        if err < tol: break
    return z
