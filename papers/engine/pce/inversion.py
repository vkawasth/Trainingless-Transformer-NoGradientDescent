"""Bayesian inversion, the conditioning operator it induces, and the rectangular hull (lab Section "What does not matter",
subsection "Inversion"). New module.

  bayes_inverse(p, K)          K: X -> Y a stochastic matrix (rows x), p a law on X. Returns (q, Kdag) with q = p K and
                               Kdag[y, x] = p(x) K[x, y] / q(y) where q(y) > 0 (rows with q(y) = 0 are NaN: the partial domain)
  joint(p, K)                  the coupling p(x) K[x, y]
  cond_op(f, P, axes)          C_X f(P) = sum_x P(X = x) f(P | X = x), computed through the Bayesian inverse of the marginal map
  posterior_types(w, a, x)     the Bayesian inverse of a k-type realisation at the record x: P(z | x)
  rect_hull(Qs, axes)          extreme points of Rect_X(F) for F = conv(Qs): every marginal of X in F pasted with, independently for each
                               x, every conditional F | X = x in F
"""
from __future__ import annotations
import itertools
import numpy as np


def joint(p, K): return p[:, None] * K


def bayes_inverse(p, K):
    q = p @ K; Kd = np.full((K.shape[1], K.shape[0]), np.nan)
    for y in range(K.shape[1]):
        if q[y] > 0: Kd[y] = p * K[:, y] / q[y]
    return q, Kd


def _cells(P, axes):
    """the marginal map X: cells -> values of X as a stochastic (0/1) matrix, and the flattened law"""
    m = P.ndim; xs = list(itertools.product((0, 1), repeat=len(axes))); C = list(itertools.product((0, 1), repeat=m))
    K = np.array([[1.0 if tuple(c[a] for a in axes) == x else 0.0 for x in xs] for c in C]); return C, xs, K


def cond_op(f, P, axes):
    C, xs, K = _cells(P, axes); p = P.ravel(); q, Kd = bayes_inverse(p, K); tot = 0.0
    for j, x in enumerate(xs):
        if q[j] > 0: tot += q[j] * f(Kd[j].reshape(P.shape))
    return tot


def posterior_types(w, a, x):
    lik = np.prod(np.where(np.asarray(x) == 1, a, 1 - a), axis=1); post = w * lik; return post / post.sum()


def rect_hull(Qs, axes):
    shape = Qs[0].shape; C, xs, K = _cells(Qs[0], axes)
    margs = []; posts = {j: [] for j in range(len(xs))}
    for P in Qs:
        q, Kd = bayes_inverse(P.ravel(), K)
        if not any(np.allclose(q, m) for m in margs): margs.append(q)
        for j in range(len(xs)):
            if q[j] > 0 and not any(np.allclose(Kd[j], b) for b in posts[j]): posts[j].append(Kd[j])
    out = []
    for m in margs:
        for choice in itertools.product(*[posts[j] for j in range(len(xs))]):
            out.append(sum(m[j] * choice[j] for j in range(len(xs))).reshape(shape))
    return out
