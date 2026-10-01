"""Bounds on an outcome over all gluings: the fibre of the restriction map.

R : Delta(E(X)) -> Fam(U) sends a global law to its context marginals. For a family e the fibre R^{-1}(e) is the
polytope of global laws consistent with every local table. For an outcome functional f on global sections (the
indicator of an event that may involve measurements never observed jointly, or any real score), the range of
<f, P> over the fibre is an interval [lo, hi], computed by two linear programmes.

  * it is directional: f is ordered, and relabelling the outcome maps [lo, hi] to [1 - hi, 1 - lo];
  * it chooses nothing: it ranges over every gluing, so it cannot be cherry-picked;
  * it is empty exactly when the counit of image -| preimage fails, i.e. when e is contextual (CF > 0) or signalling.

When the fibre is empty there are two honest fallbacks, both reported with the size of the failure:
  projected   bound over the fibre of the nearest consistent family R(q*), q* = weighted I-projection
              (deficits.nearest_consistent), reported with the gluing defect;
  nc_part     bound over the normalised non-contextual parts b / (1 - CF), b >= 0, R b <= e, |b| = 1 - CF
              (the optimal face of the CF programme), reported with CF.
The maximum-entropy member of the fibre (IPF) is returned as the point estimate; it always lies in [lo, hi].
"""
from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional

import numpy as np
from scipy.optimize import linprog

from .scenario import EmpiricalModel, Scenario
from .deficits import nearest_consistent
from .outcome import contextual_fraction


def functional(sc: Scenario, pred: Callable[[Dict[str, object]], float]) -> np.ndarray:
    """vector over global sections: f[g] = pred({measurement: value})"""
    X = sc.measurements
    return np.array([float(pred(dict(zip(X, g)))) for g in sc.global_sections()])


def _R(sc: Scenario) -> np.ndarray:
    X = sc.measurements
    return np.vstack([sc.restriction_matrix(X, C) for C in sc.contexts])


def _range(f, A_eq, b_eq, A_ub=None, b_ub=None, tol=1e-9):
    n = len(f); out = []
    for sgn in (1, -1):
        if A_ub is None:            # equality up to tol, as two inequalities (robust to rounding)
            A = np.vstack([A_eq, -A_eq]); b = np.concatenate([b_eq + tol, -b_eq + tol])
            r = linprog(sgn * f, A_ub=A, b_ub=b, A_eq=np.ones((1, n)), b_eq=[1.0], bounds=(0, None), method="highs")
        else:
            r = linprog(sgn * f, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
        if r.status != 0:
            return None
        out.append(sgn * r.fun)
    return out[0], out[1]


def _maxent(sc, tables, iters=3000, tol=1e-12):
    X = sc.measurements; R = {C: sc.restriction_matrix(X, C) for C in sc.contexts}
    n = next(iter(R.values())).shape[1]; P = np.full(n, 1.0 / n)
    for _ in range(iters):
        for C in sc.contexts:
            m = R[C] @ P
            P = P * (R[C].T @ np.divide(tables[C], m, out=np.zeros_like(m), where=m > 0))
        if max(np.abs(R[C] @ P - tables[C]).max() for C in sc.contexts) < tol:
            break
    return P / P.sum()


def outcome_bounds(model: EmpiricalModel, f: np.ndarray, weights: Optional[Mapping] = None, tol: float = 1e-9) -> dict:
    sc = model.scenario; A = _R(sc)
    b = np.concatenate([model.tables[C] for C in sc.contexts])
    r = _range(f, A, b, tol=tol)
    if r is not None:
        return dict(lo=r[0], hi=r[1], mode="exact", defect=0.0, point=float(f @ _maxent(sc, model.tables)))
    q = nearest_consistent(model, weights=weights)
    X = sc.measurements
    tabs = {C: sc.restriction_matrix(X, C) @ q for C in sc.contexts}
    defect = float(sum(0.5 * np.abs(tabs[C] - model.tables[C]).sum() for C in sc.contexts) / len(sc.contexts))
    bq = np.concatenate([tabs[C] for C in sc.contexts])
    r = _range(f, A, bq, tol=1e-7)
    return dict(lo=r[0], hi=r[1], mode="projected", defect=defect, point=float(f @ q))


def nc_part_bounds(model: EmpiricalModel, f: np.ndarray) -> dict:
    """range of <f, b>/(1-CF) over the optimal non-contextual parts b (R b <= e, |b| = 1 - CF, b >= 0)"""
    sc = model.scenario; A = _R(sc); b = np.concatenate([model.tables[C] for C in sc.contexts])
    cf = contextual_fraction(model).value; m = 1 - cf
    if m < 1e-12:
        return dict(lo=np.nan, hi=np.nan, cf=cf)
    out = []
    for sgn in (1, -1):
        r = linprog(sgn * f, A_ub=A, b_ub=b + 1e-9, A_eq=np.ones((1, len(f))), b_eq=[m - 1e-9], bounds=(0, None), method="highs")
        out.append(sgn * r.fun / m)
    return dict(lo=out[0], hi=out[1], cf=cf)


# ------------------------------------------------------------------------------------------- grey (what-if) contexts
"""A grey context is an assumption added to the cover: a variable (a filter such as 'age >= 55') whose joint with the
outcome is never observed, entered as linear constraints on the global law. Two kinds:
  population  a known table on some variables (e.g. the age distribution within each ideology group, from a census):
              equality constraints, exactly like an observed context;
  hypothesis  a range for a (conditional) probability, lo <= P(E | G) <= hi, i.e. P(E,G) - lo P(G) >= 0 and
              P(E,G) - hi P(G) <= 0: linear in P.
Then (i) consistency: is there a global law fitting data + grey? If not, the minimal total slack on the hypothesis
constraints (an L1 distance) says how far the what-if is from possible; (ii) bounds on a target, possibly a
conditional P(E | G) (linear-fractional; Charnes--Cooper); (iii) what-if curves by sweeping a hypothesis."""


def event(sc: Scenario, pred) -> np.ndarray:
    return functional(sc, lambda g: float(bool(pred(g))))


def hypothesis(sc: Scenario, E, G, lo: float, hi: float):
    """rows (a, l, u) meaning l <= a.P <= u, encoding lo <= P(E | G) <= hi"""
    e, g = event(sc, lambda x: E(x) and G(x)), event(sc, G)
    return [(e - lo * g, 0.0, np.inf), (e - hi * g, -np.inf, 0.0)]


def _data_rows(model: EmpiricalModel, project: bool, weights=None):
    sc = model.scenario; X = sc.measurements
    tables = model.tables
    if project:
        A = np.vstack([sc.restriction_matrix(X, C) for C in sc.contexts]); b = np.concatenate([tables[C] for C in sc.contexts])
        r = linprog(np.zeros(A.shape[1]), A_eq=A, b_eq=b, bounds=(0, None), method="highs")
        if r.status != 0:
            q = nearest_consistent(model, weights=weights)
            tables = {C: sc.restriction_matrix(X, C) @ q for C in sc.contexts}
    A = np.vstack([sc.restriction_matrix(X, C) for C in sc.contexts]); b = np.concatenate([tables[C] for C in sc.contexts])
    return A, b, tables


def _stack(rows, n):
    """rows (a, l, u) -> A_ub, b_ub for A_ub x <= b_ub (homogeneous version uses scale t)"""
    A, l, u = [], [], []
    for a, lo, hi in rows:
        A.append(a); l.append(lo); u.append(hi)
    return (np.array(A) if A else np.zeros((0, n))), np.array(l), np.array(u)


def grey_bounds(model: EmpiricalModel, target, given=None, rows=(), weights=None, tol=1e-7) -> dict:
    """Bounds on <target, P> (or on <target, P>/<given, P> when `given` is a vector) over global laws P that reproduce
    the data contexts of `model` (projected to the nearest consistent family if needed) and satisfy the grey rows.
    Returns lo, hi, feasible, and if infeasible the minimal L1 slack on the grey rows."""
    A, b, _ = _data_rows(model, True, weights); n = A.shape[1]
    G, gl, gu = _stack(rows, n)
    g = np.ones(n) if given is None else np.asarray(given, float)
    # Charnes--Cooper: y = t P, t = 1 / <g, P>;  constraints  A y = t b,  sum y = t,  gl t <= G y <= gu t,  <g, y> = 1
    Aeq = np.vstack([np.hstack([A, -b[:, None]]), np.hstack([np.ones((1, n)), [[-1.0]]]), np.hstack([g[None], [[0.0]]])])
    beq = np.concatenate([np.zeros(len(b)), [0.0], [1.0]])
    Aub, bub = [], []
    for k in range(len(G)):
        if np.isfinite(gu[k]):
            Aub.append(np.concatenate([G[k], [-gu[k]]])); bub.append(tol)
        if np.isfinite(gl[k]):
            Aub.append(np.concatenate([-G[k], [gl[k]]])); bub.append(tol)
    Aub = np.array(Aub) if Aub else None; bub = np.array(bub) if bub else None
    out = []
    for sgn in (1, -1):
        r = linprog(sgn * np.concatenate([target, [0.0]]), A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq,
                    bounds=(0, None), method="highs")
        if r.status != 0:
            return dict(lo=np.nan, hi=np.nan, feasible=False, slack=grey_slack(A, b, G, gl, gu))
        out.append(sgn * r.fun)
    return dict(lo=out[0], hi=out[1], feasible=True, slack=0.0)


def grey_slack(A, b, G, gl, gu):
    """min sum of slacks s >= 0 with  l - s <= G P <= u + s,  A P = b,  P >= 0, sum P = 1  (distance of the what-if
    from the data, in probability units)"""
    n, m = A.shape[1], len(G)
    c = np.concatenate([np.zeros(n), np.ones(m)])
    Aub, bub = [], []
    for k in range(m):
        e = np.zeros(m); e[k] = 1
        if np.isfinite(gu[k]):
            Aub.append(np.concatenate([G[k], -e])); bub.append(gu[k])
        if np.isfinite(gl[k]):
            Aub.append(np.concatenate([-G[k], -e])); bub.append(-gl[k])
    Aeq = np.vstack([np.hstack([A, np.zeros((len(b), m))]), np.concatenate([np.ones(n), np.zeros(m)])[None]])
    r = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub), A_eq=Aeq, b_eq=np.concatenate([b, [1.0]]), bounds=(0, None), method="highs")
    return float(r.fun) if r.status == 0 else np.nan
