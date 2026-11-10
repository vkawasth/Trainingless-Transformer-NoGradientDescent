"""Projections of the decision maker's closed trajectory (sec. 'Beliefs under contamination', 'One knot, many projections'). New module.
The DM's state at each step is summarised by six scalars, each computable from the data he holds:
  voi    the best expected reduction, over inspections not yet made, of the width of the k-robust interval of the decision functional
         (expectation over laws sampled from the current least-incoherent fibre)
  incoh  the residual: the smallest L1 discrepancy of any law from the admitted tables (the sheaves of the neighbourhoods not aligning)
  width  the width of the k-robust interval (the polytope view)
  value  the midpoint of the least-incoherent fibre's interval (the outcome view)
  drift  the total L1 change of the admitted tables since each was first reported (the trace of the tables)
  imped  the number of withheld reports and refused inspections so far (impediments)
A history closed by the canonical replay of its final cells is a closed polygon in R^6. Every closed curve in R^n, n >= 4, is unknotted,
so a knot type exists only for a projection to three of the six coordinates. For each of the 20 choices:
  singular(poly)        the self-intersections of the polygon in R^3 (vertex repeats and segment contacts): pseudo-crossings, whose
                        over/under the three chosen coordinates cannot decide
  polygon_pd(poly)      a PD code from a generic plane projection (fixed rotation), over/under from the height
  knot_type(poly)       Jones polynomial and its name (pce/pdknot.classify) when the polygon is not singular
Knot type in R^3 does not change under positive rescaling, or any increasing reparametrisation, of each coordinate; reversing one
coordinate gives the mirror image.
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import linprog
from . import tamper as T
from . import pdknot as PDK

COMPONENTS = ["voi", "incoh", "width", "value", "drift", "imped"]


def _rot(seed=7):
    rng = np.random.default_rng(seed); Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return Q if np.linalg.det(Q) > 0 else -Q


def _seg_dist(p1, p2, q1, q2):
    """minimum distance between segments in R^3 (sampled closed form via clamped least squares)"""
    d1, d2, r = p2 - p1, q2 - q1, p1 - q1
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a < 1e-15 and e < 1e-15: return np.linalg.norm(p1 - q1)
    best = np.inf
    for s in np.linspace(0, 1, 41):
        x = p1 + s * d1
        tt = 0.0 if e < 1e-15 else np.clip((x - q1) @ d2 / e, 0, 1)
        best = min(best, np.linalg.norm(x - (q1 + tt * d2)))
    return best


def clean(poly, eps=1e-9):
    out = [poly[0]]
    for p in poly[1:]:
        if np.linalg.norm(p - out[-1]) > eps: out.append(p)
    if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) <= eps: out.pop()
    return np.array(out)


def singular(poly, tol=1e-6):
    """pairs of non-adjacent edges of the closed polygon that touch in R^3"""
    P = clean(poly); m = len(P); hits = []
    for i in range(m):
        for j in range(i + 1, m):
            if j == i + 1 or (i == 0 and j == m - 1): continue
            if _seg_dist(P[i], P[(i + 1) % m], P[j], P[(j + 1) % m]) < tol: hits.append((i, j))
    return hits


def polygon_pd(poly, seed=7):
    P = clean(poly) @ _rot(seed).T; m = len(P); X = []
    for i in range(m):
        for j in range(i + 2, m):
            if i == 0 and j == m - 1: continue
            a, b, c, d = P[i][:2], P[(i + 1) % m][:2], P[j][:2], P[(j + 1) % m][:2]
            M = np.array([b - a, c - d]).T
            if abs(np.linalg.det(M)) < 1e-14: continue
            s, t = np.linalg.solve(M, c - a)
            if 0 < s < 1 and 0 < t < 1:
                zi = P[i][2] + s * (P[(i + 1) % m][2] - P[i][2]); zj = P[j][2] + t * (P[(j + 1) % m][2] - P[j][2])
                X.append((i + s, j + t, zi > zj, b - a, d - c))
    if not X: return []
    passages = []
    for k, (pi, pj, i_over, di, dj) in enumerate(X):
        passages.append((pi, k, i_over, di)); passages.append((pj, k, not i_over, dj))
    passages.sort(key=lambda z: z[0]); n2 = len(passages)
    pos = {}
    for idx, (_, k, over, d) in enumerate(passages):
        pos.setdefault(k, {})["over" if over else "under"] = (idx, d)
    pd = []
    for k in range(len(X)):
        (iu, du), (io, do) = pos[k]["under"], pos[k]["over"]
        a_in = iu if iu > 0 else n2; a_out = iu + 1
        b_in = io if io > 0 else n2; b_out = io + 1
        cross = du[0] * do[1] - du[1] * do[0]
        pd.append((a_in, b_in, a_out, b_out) if cross > 0 else (a_in, b_out, a_out, b_in))
    return pd


def knot_type(poly, seed=7):
    pd = polygon_pd(poly, seed)
    if len(pd) <= 2: return "unknot", 1                                  # every diagram with at most two crossings is the unknot

    V = PDK.jones(pd); return PDK.classify(V), V
