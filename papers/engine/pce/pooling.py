"""Validity of pooling: Beck-Chevalley defect, the KL ball, and the decision-error bound.

  * defect(P): for a joint law P[a, m, b] of (A, M, B), the Beck-Chevalley defect of the windows sigma(A, M), sigma(M, B) through M:
    c = max over m of the largest non-trivial singular value of the normalised (A, B) table given M = m (Theorem 14.4), and
    I(A; B | M) in bits. Pooling through M (the Markov glue) is exact iff c = 0.
  * kl_ball(P): for a joint law P[x, i, j] of (X, q_i, q_j), the radius I(X; q_j | q_i) of the ball of pooling rules that are at least
    as good as agent i in log loss, and the bound sqrt(2 I) (I in nats) on the excess decision error of any rule in the ball
    (companion note, Theorem 7 and Proposition 9).
"""
from __future__ import annotations
import math
import numpy as np


def defect(P):
    P = np.asarray(P, float); c = 0.0; I = 0.0
    for m in range(P.shape[1]):
        T = P[:, m, :]; w = T.sum()
        if w <= 0: continue
        T = T / w; r, k = T.sum(1), T.sum(0)
        R, K = r > 0, k > 0
        N = T[np.ix_(R, K)] / np.sqrt(np.outer(r[R], k[K]))
        sv = np.linalg.svd(N, compute_uv=False)
        if len(sv) > 1: c = max(c, float(sv[1]))
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(T > 0, T * np.log2(T / np.outer(r, k)), 0.0)
        I += w * float(np.nansum(terms))
    return c, I


def markov_glue(P):
    """the Beck-Chevalley (conditionally independent) law with the same two windows."""
    P = np.asarray(P, float); PAM, PMB, PM = P.sum(2), P.sum(0), P.sum((0, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        Q = np.einsum("am,mb->amb", PAM, PMB) / np.where(PM > 0, PM, 1)[None, :, None]
    return Q


def kl_ball(P):
    """P[x, i, j]: X binary or finite; returns (radius in nats, decision bound sqrt(2 radius))."""
    P = np.asarray(P, float); I = 0.0
    Pij = P.sum(0); Pi = P.sum((0, 2))
    for i in range(P.shape[1]):
        for j in range(P.shape[2]):
            if Pij[i, j] <= 0: continue
            qs = P[:, i, j] / Pij[i, j]; qi = P[:, i, :].sum(1) / Pi[i]
            I += Pij[i, j] * float(sum(a * math.log(a / b) for a, b in zip(qs, qi) if a > 0))
    I = max(I, 0.0)                                         # clamp rounding error: I >= 0 exactly
    return I, math.sqrt(2 * I)
