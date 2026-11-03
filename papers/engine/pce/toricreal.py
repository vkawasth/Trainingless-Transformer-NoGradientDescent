"""Geometric realization: the toric picture (layers paper, Section "Geometric realization: the toric picture"). New module.

Cells with non-negative coefficients are sums of monomials c_i x^(S_i). The monomial law q(x) (q_i proportional to c_i x^(S_i)) is a point
of the toric (log-linear) model of the support S; the algebraic moment map mu(x) = x grad log P(x) = sum_i q_i S_i identifies the positive
model with the relative interior of the Newton polytope Q = conv(S) (Birch's theorem), and the cell law is the aggregation of q.

  monomial_law(S, c, x)            q(x) (floats)
  inverse_moment(S, c, y, starts)  the monomial laws with moment y, from several starts of the strictly convex problem
                                   min_u log sum_i c_i e^(<S_i, u>) - <y, u>   (they agree: Birch)
  face_law(S, c, F, cv)            the limit along an arc with leading coefficient cv and face F: normalise(c_i cv^(S_i)) on F (exact)
  faces_by_weight(S, weights)      {face F (indices minimising <w, .>): one weight exposing it}
  lower(S, F)                      is F a lower face, i.e. exposed by some strictly positive weight? (linear programme)
"""
from __future__ import annotations
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import minimize, linprog


def monomial_law(S, c, x):
    A = np.array(S, float); lw = np.log(np.array(c, float)) + A @ np.log(np.array(x, float)); w = np.exp(lw - lw.max())
    return w / w.sum()


def inverse_moment(S, c, y, starts=4, rng=None):
    A = np.array(S, float); lc = np.log(np.array(c, float)); y = np.array(y, float); rng = rng or np.random.default_rng(0)
    def f(u):
        z = lc + A @ u; m = z.max(); return m + np.log(np.exp(z - m).sum()) - y @ u
    def g(u):
        z = lc + A @ u; w = np.exp(z - z.max()); return A.T @ (w / w.sum()) - y
    sols = []
    for _ in range(starts):
        r = minimize(f, rng.normal(0, 1, A.shape[1]), jac=g, method="BFGS", options=dict(gtol=1e-14, maxiter=10000))
        z = lc + A @ r.x; w = np.exp(z - z.max()); sols.append(w / w.sum())
    return sols


def face_law(S, c, F, cv):
    vals = {i: Fr(c[i]) * np.prod([Fr(v) ** e for v, e in zip(cv, S[i])], dtype=object) for i in F}; tot = sum(vals.values())
    return {i: v / tot for i, v in vals.items()}


def faces_by_weight(S, weights):
    out = {}
    for w in weights:
        v = [sum(wi * ai for wi, ai in zip(w, a)) for a in S]; m = min(v)
        out.setdefault(tuple(i for i in range(len(S)) if v[i] == m), tuple(w))
    return out


def lower(S, F, witness=False):
    """LP: w >= 1 componentwise, <w, S_i> = <w, S_j> on F, and <w, S_k> >= <w, S_F> + 1 off F (with witness=True, also return w)"""
    F = list(F); d = len(S[0]); f0 = np.array(S[F[0]], float)
    A_eq = [np.array(S[i], float) - f0 for i in F[1:]]; A_ub = [f0 - np.array(S[k], float) for k in range(len(S)) if k not in F]
    r = linprog(np.zeros(d), A_ub=np.array(A_ub) if A_ub else None, b_ub=-np.ones(len(A_ub)) if A_ub else None,
                A_eq=np.array(A_eq) if A_eq else None, b_eq=np.zeros(len(A_eq)) if A_eq else None, bounds=[(1, None)] * d, method="highs")
    return (r.status == 0, r.x if r.status == 0 else None) if witness else r.status == 0
