"""Arity-graded holonomy for multi-source measurements.

Data: values v(u, s) of a real measurement (e.g. bias) of unit u (event, sentence) by source s; a unit may be
measured by only some sources. Connection on the source graph: moving A -> B along unit u adds
d_u(A, B) = v(u, A) - v(u, B). The holonomy of a loop of sources C = (s_1, ..., s_k) is
    Phi = sum_edges  mean_{u in U_e} d_u(s_i, s_{i+1}),
where U_e is the set of units used for edge e.

ARITY GRADE of a unit relative to the loop = the number of the loop's sources that measured it. Grade-g
holonomy uses, for every edge, only units of grade exactly g that cover that edge.
  Theorem (filled cells are flat). At full grade g = k every edge uses the SAME units (those measured by all
  of C), and for each unit the edge differences telescope (delta delta = 0), so Phi_k = 0 exactly.
  At g < k the edges use different units and Phi_g is free: it measures scale inconsistency that no single
  multi-source unit can see.
Test: z = Phi_g / sqrt(sum_e var_e / n_e), and a permutation test that reshuffles each edge's differences
across grades (null: holonomy does not depend on the grade).
"""
from __future__ import annotations
import itertools
import numpy as np


def grade_units(values: dict, loop):
    """values: {unit: {source: value}}. Returns {unit: grade} relative to the loop."""
    L = set(loop)
    return {u: len(L & set(vs)) for u, vs in values.items()}


def edge_diffs(values, loop, units):
    out = []
    k = len(loop)
    for i in range(k):
        a, b = loop[i], loop[(i + 1) % k]
        out.append(np.array([values[u][a] - values[u][b] for u in units if a in values[u] and b in values[u]], float))
    return out


def holonomy(values, loop, grade, min_units=2):
    """Phi at a given arity grade; returns dict(phi, se, z, n_per_edge) or None if some edge lacks units."""
    g = grade_units(values, loop)
    units = [u for u in values if g[u] == grade]
    diffs = edge_diffs(values, loop, units)
    if any(len(d) < min_units for d in diffs):
        return None
    phi = float(sum(d.mean() for d in diffs))
    se = float(np.sqrt(sum(d.var(ddof=1) / len(d) for d in diffs)))
    return dict(phi=phi, se=se, z=phi / se if se > 0 else (0.0 if abs(phi) < 1e-12 else np.inf),
                n_per_edge=[int(len(d)) for d in diffs], max_abs_unit_loop=_unit_loop(values, loop, units))


def _unit_loop(values, loop, units):
    """largest |sum of edge differences| over individual units that cover the whole loop (0 by delta delta = 0)"""
    full = [u for u in units if set(loop) <= set(values[u])]
    if not full:
        return None
    k = len(loop)
    return float(max(abs(sum(values[u][loop[i]] - values[u][loop[(i + 1) % k]] for i in range(k))) for u in full))


def thin(values, keep_prob, rng):
    """randomly drop source measurements (simulated incomplete coverage); keeps at least one per unit"""
    out = {}
    for u, vs in values.items():
        kept = {s: v for s, v in vs.items() if rng.random() < keep_prob}
        if kept:
            out[u] = kept
    return out
