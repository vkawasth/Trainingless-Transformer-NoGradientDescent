"""The finite Giry monad as used in this work: Kleisli category FinStoch, Bayesian inversion, and the
mutual-information persistence functor (with the HSIC counterexample).

* Kleisli morphisms X -> D(Y) are stochastic matrices; composition is Chapman-Kolmogorov (matrix product).
* Decoding is Bayesian inversion: for a prior p on Z and a kernel k: Z -> D(X), the inverse
  k^dagger: X -> D(Z) has k^dagger(z | x) = p(z) k(x | z) / sum_z' p(z') k(x | z'). In the tree, the INSIDE vector of a
  block is the likelihood k(x_block | z) (the block's own leaves only), the OUTSIDE vector carries the rest of the
  sequence, and the block posterior inside * outside (normalised) is the Bayesian inverse given the whole sequence.
* MI persistence: for a joint law of variables X_1..X_k, the weighted graph with weights I(X_i; X_j) gives a
  filtration K(t) = {edges with I >= t}. A coarse-graining f (a deterministic surjection on one variable) cannot increase
  any mutual information (data processing), so K_f(t) is a subcomplex of K(t) for every t: coarse-graining induces
  a morphism of persistence modules (contravariant). HSIC does not satisfy data processing, so the same construction
  with HSIC weights is not functorial.
"""
from __future__ import annotations
import itertools
import numpy as np


def kleisli(A, B):
    """compose stochastic matrices A: X -> D(Y) (rows sum to 1) and B: Y -> D(Z)"""
    return A @ B


def bayes_inverse(prior, K):
    """prior on Z (|Z|), kernel K[z, x] = k(x | z); returns K_dag[x, z] = p(z | x)"""
    J = prior[:, None] * K
    return (J / J.sum(0, keepdims=True)).T


def mutual_information(P2):
    P2 = np.asarray(P2, float); P2 = P2 / P2.sum()
    a, b = P2.sum(1, keepdims=True), P2.sum(0, keepdims=True)
    m = P2 > 0
    return float((P2[m] * np.log(P2[m] / (a @ b)[m])).sum())


def pairwise_mi(J):
    k = J.ndim; M = np.zeros((k, k))
    for i, j in itertools.combinations(range(k), 2):
        other = tuple(a for a in range(k) if a not in (i, j))
        M[i, j] = M[j, i] = mutual_information(J.sum(axis=other))
    return M


def coarse_grain(J, axis, mapping):
    """push the joint table forward along a surjection mapping: old state -> new state on one axis"""
    n_new = max(mapping) + 1
    shape = list(J.shape); shape[axis] = n_new
    out = np.zeros(shape)
    for old, new in enumerate(mapping):
        idx_o = [slice(None)] * J.ndim; idx_o[axis] = old
        idx_n = [slice(None)] * J.ndim; idx_n[axis] = new
        out[tuple(idx_n)] += J[tuple(idx_o)]
    return out


def edges_at(M, t):
    k = M.shape[0]
    return {(i, j) for i, j in itertools.combinations(range(k), 2) if M[i, j] >= t}


def hsic(x, y, s=1.0):
    """biased empirical HSIC with Gaussian kernels of bandwidth s"""
    x, y = np.asarray(x, float), np.asarray(y, float); n = len(x)
    K = np.exp(-(x[:, None] - x[None]) ** 2 / (2 * s * s)); L = np.exp(-(y[:, None] - y[None]) ** 2 / (2 * s * s))
    H = np.eye(n) - 1.0 / n
    return float(np.trace(K @ H @ L @ H) / n ** 2)
