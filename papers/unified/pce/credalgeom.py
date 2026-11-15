"""Credal geometry, mixing and flow (section sec:credalgeom). New module; it modifies nothing.
A credal set is a polytope of laws, given either by vertices or by linear constraints on base variables p >= 0 (spec: A_ub, b_ub, A_eq,
b_eq). Coarse cells are linear images y = Y p. All ratio problems are linear-fractional and are solved exactly by Charnes-Cooper.
  ratio_max(spec, num, den)        max over the polytope of (num . p) / (den . p); None if den . p can vanish with num . p > 0 (unbounded)
  lin_min / lin_max(spec, v)       min / max of v . p
  diam_lp(spec, Y)                 Hilbert diameter of the image Y p: drops structural zeros (cells zero on the whole set), reports cells that
                                   touch zero (diameter infinite) and otherwise max_{i,j} log r_ij + log r_ji, r_ij = max y_i / y_j
  eps_profile(spec, Y, eps)        diameters of the set restricted to y_i >= eps (finite), for each eps
  diam_vertices(V)                 brute force over vertex pairs (for checks)
  band(spec, num, den)             [min, max] of (num . p) / (den . p); 'undefined' if the condition can have probability 0
  logodds_band(spec, aC, nC)       [min, max] of log((aC . p) / (nC . p))
  vertex_weights(lam, V, cond)     condition-dependent mixture weights lam_k(C) = lam_k V_k(C) / sum_j lam_j V_j(C)
  flow_matrix(L, a, t)             M_t = exp(t (L + diag a)): the replicator-mutator (normalised Feynman-Kac) flow p -> p M_t / <p M_t, 1>
"""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import linprog
from scipy.linalg import expm
def _cc(spec, num, den, sense):
    """Charnes-Cooper: optimise num.z subject to den.z = 1, A_ub z <= b_ub s, A_eq z = b_eq s, z >= 0, s >= 0."""
    n = len(num); A_ub, b_ub, A_eq, b_eq = spec["A_ub"], spec["b_ub"], spec["A_eq"], spec["b_eq"]
    Aub = np.hstack([A_ub, -b_ub[:, None]]) if A_ub is not None and len(A_ub) else None
    Aeq = np.vstack([np.hstack([A_eq, -b_eq[:, None]]), np.concatenate([den, [0.0]])[None, :]])
    beq = np.concatenate([np.zeros(len(A_eq)), [1.0]])
    r = linprog(sense * np.concatenate([num, [0.0]]), A_ub=Aub, b_ub=None if Aub is None else np.zeros(len(Aub)), A_eq=Aeq, b_eq=beq,
                bounds=[(0, None)] * (n + 1), method="highs")
    if r.status == 3: return None                                  # unbounded
    if r.status != 0: return float("nan")
    return sense * r.fun
def ratio_max(spec, num, den):
    v = _cc(spec, num, den, -1.0); return None if v is None else v
def ratio_min(spec, num, den):
    v = _cc(spec, num, den, 1.0); return v
def lin_min(spec, v):
    r = linprog(v, A_ub=spec["A_ub"], b_ub=spec["b_ub"], A_eq=spec["A_eq"], b_eq=spec["b_eq"], bounds=[(0, None)] * len(v), method="highs")
    return float(r.fun)
def lin_max(spec, v): return -lin_min(spec, -np.asarray(v, float))
def diam_lp(spec, Y, tol=1e-12):
    Y = np.asarray(Y, float); k = Y.shape[0]
    mx = [lin_max(spec, Y[i]) for i in range(k)]; mn = [lin_min(spec, Y[i]) for i in range(k)]
    structural = [i for i in range(k) if mx[i] <= tol]; keep = [i for i in range(k) if i not in structural]
    touching = [i for i in keep if mn[i] <= tol]
    if touching: return dict(diam=float("inf"), structural=structural, touching=touching, pair=None)
    best, pair = 0.0, None; R = {}
    for i in keep:
        for j in keep:
            if i != j: R[(i, j)] = ratio_max(spec, Y[i], Y[j])
    for i in keep:
        for j in keep:
            if i < j:
                d = math.log(R[(i, j)]) + math.log(R[(j, i)])
                if d > best: best, pair = d, (i, j)
    return dict(diam=best, structural=structural, touching=[], pair=pair)
def restrict(spec, Y, eps):
    Y = np.asarray(Y, float); A_ub = np.vstack([spec["A_ub"], -Y]) if spec["A_ub"] is not None else -Y
    b_ub = np.concatenate([spec["b_ub"], -eps * np.ones(len(Y))]) if spec["b_ub"] is not None else -eps * np.ones(len(Y))
    return dict(A_ub=A_ub, b_ub=b_ub, A_eq=spec["A_eq"], b_eq=spec["b_eq"])
def eps_profile(spec, Y, eps_list):
    out = []
    for e in eps_list:
        s = restrict(spec, Y, e)
        if np.isnan(lin_min(s, np.zeros(spec["A_eq"].shape[1]))): out.append((e, None)); continue
        out.append((e, diam_lp(s, Y)["diam"]))
    return out
def hilbert(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float); r = np.log(p) - np.log(q); return float(r.max() - r.min())
def diam_vertices(V):
    V = [np.asarray(v, float) for v in V]; keep = [i for i in range(len(V[0])) if any(v[i] > 0 for v in V)]
    if any(min(v[i] for v in V) <= 0 for i in keep): return float("inf")
    return max(hilbert(v[keep], w[keep]) for v in V for w in V)
def band(spec, num, den, tol=1e-12):
    if lin_min(spec, den) <= tol: return dict(lo=None, hi=None, undefined=True)
    return dict(lo=ratio_min(spec, num, den), hi=ratio_max(spec, num, den), undefined=False)
def logodds_band(spec, aC, nC):
    lo = ratio_min(spec, aC, nC); hi = ratio_max(spec, aC, nC)
    if lo is None or hi is None or lo <= 0: return None
    return math.log(lo), math.log(hi)
def vertex_weights(lam, V, cond):
    w = np.array([l * float(np.dot(v, cond)) for l, v in zip(lam, V)]); return w / w.sum()
def flow_matrix(L, a, t): return expm(t * (np.asarray(L, float) + np.diag(np.asarray(a, float))))
