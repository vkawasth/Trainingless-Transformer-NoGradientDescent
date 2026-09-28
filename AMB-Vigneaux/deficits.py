"""Renyi deficits: the alpha-dial applied to the EXTENSION problem, not to per-context functionals.

    Delta_alpha(e) = min over global laws q of  sum_C w_C D_alpha(e_C || q_C),     q_C = marginal of q on C,
    D_alpha(p||q)  = log( sum_s p(s)^alpha q(s)^(1-alpha) ) / (alpha - 1),   D_1 = KL,   D_0 = -log q(supp p).

For alpha in [0, 1] the objective is convex in q (sum p^a q^(1-a) is concave, -log of it is convex).
Facts (see the write-up):
  * Delta_alpha is non-decreasing in alpha (Renyi divergences are).
  * alpha in (0, 1]:  Delta_alpha = 0  iff a global law reproduces every e_C  iff  e is non-contextual.
  * alpha = 0:        Delta_0 = 0      iff some global law lives on the global sections of the support
                                        presheaf  iff  e is NOT strongly contextual.
So the dial runs from the possibilistic (support-level, F_Z-side) strong obstruction at alpha = 0 to
probabilistic contextuality for every alpha > 0. The objective is multi-context, so Layer-2 blindness
(which concerns per-context functionals) does not apply.
"""
from __future__ import annotations

import numpy as np

from .scenario import EmpiricalModel


def _setup(model: EmpiricalModel):
    sc = model.scenario
    X = sc.measurements
    R = {C: sc.restriction_matrix(X, C) for C in sc.contexts}
    return sc, R


def renyi_div(p, q, alpha):
    msk = p > 0
    if alpha == 0:
        m = q[msk].sum()
        return np.inf if m <= 0 else -np.log(m)
    if alpha == 1:
        return float((p[msk] * np.log(p[msk] / np.maximum(q[msk], 1e-300))).sum())
    s = (p[msk] ** alpha * np.maximum(q[msk], 1e-300) ** (1 - alpha)).sum()
    return float(np.log(s) / (alpha - 1))


def renyi_deficit(model: EmpiricalModel, alpha: float, iters: int = 4000, lr: float = 0.5, seed: int = 0):
    """Mirror descent (exponentiated gradient) on the simplex of global laws."""
    sc, R = _setup(model)
    C = list(sc.contexts); w = 1.0 / len(C)
    nG = R[C[0]].shape[1]
    q = np.full(nG, 1.0 / nG)

    def obj(q):
        return sum(w * renyi_div(model.tables[c], R[c] @ q, alpha) for c in C)

    best = obj(q)
    for t in range(iters):
        g = np.zeros(nG)
        for c in C:
            p, m = model.tables[c], R[c] @ q
            msk = p > 0
            gc = np.zeros_like(m)
            if alpha == 0:
                tot = m[msk].sum()
                gc[msk] = -1.0 / max(tot, 1e-300)
            elif alpha == 1:
                gc[msk] = -p[msk] / np.maximum(m[msk], 1e-300)
            else:
                s = (p[msk] ** alpha * np.maximum(m[msk], 1e-300) ** (1 - alpha)).sum()
                gc[msk] = (1 - alpha) * p[msk] ** alpha * np.maximum(m[msk], 1e-300) ** (-alpha) / (s * (alpha - 1))
            g += w * (R[c].T @ gc)
        q = q * np.exp(-lr * (g - g.min()) / max(1.0, np.abs(g).max()))
        q /= q.sum()
        if t % 50 == 0 or t == iters - 1:
            best = min(best, obj(q))
    return float(max(best, 0.0))


# ---------------------------------------------------------------- repair hints
def nearest_consistent(model: EmpiricalModel, weights=None, iters: int = 5000, tol: float = 1e-13):
    """Weighted I-projection: q* = argmin_q sum_C w_C KL(e_C || q_C) over global laws (EM).
    Weights encode trust or sample size: a heavily weighted context is expensive to change."""
    sc, R = _setup(model)
    C = list(sc.contexts)
    w = np.ones(len(C)) if weights is None else np.asarray([weights[c] for c in C], float)
    w = w / w.sum()
    nG = R[C[0]].shape[1]
    q = np.full(nG, 1.0 / nG)
    for _ in range(iters):
        acc = np.zeros(nG)
        for wc, c in zip(w, C):
            m = R[c] @ q
            ratio = np.divide(model.tables[c], m, out=np.zeros_like(m), where=m > 0)
            acc += wc * q * (R[c].T @ ratio)
        new = acc / acc.sum()
        if np.abs(new - q).max() < tol:
            q = new; break
        q = new
    return q


def repair_hints(model: EmpiricalModel, weights=None):
    """Probability -> outcome and back.  Returns, per context,
      cost      w_C KL(e_C || q*_C): how much of the inconsistency this context carries
      delta     q*_C - e_C: the smallest change (in the KL sense) that makes the family consistent;
                negative cells should lose mass, positive cells gain it
      total     the deficit sum_C w_C KL(e_C || q*_C)"""
    sc, R = _setup(model)
    C = list(sc.contexts)
    q = nearest_consistent(model, weights)
    w = np.ones(len(C)) if weights is None else np.asarray([weights[c] for c in C], float)
    w = w / w.sum()
    out = {}
    for wc, c in zip(w, C):
        e, m = model.tables[c], R[c] @ q
        msk = e > 0
        kl = float((e[msk] * np.log(e[msk] / np.maximum(m[msk], 1e-300))).sum())
        out[c] = dict(cost=wc * kl, delta=m - e, sections=list(sc.sections(c)))
    return out, float(sum(v["cost"] for v in out.values()))
