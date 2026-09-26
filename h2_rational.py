#!/usr/bin/env python3
"""H^* of the delta_t complex (trivial action) on a small information structure.

Cochains C^N = A-valued functions of N-tuples of observables; with A discrete
and cochains local + measurable in P, they are constant in P, so C^N = A^{|S|^N}.

delta_t F(S_0;...;S_N) = F(S_1;...;S_N)
                        + sum_{i=0}^{N-1} (-1)^{i+1} F(...;(S_i,S_{i+1});...)
                        + (-1)^{N+1} F(S_0;...;S_{N-1})      [BB eq. (2), twisted]

The join of observables is the monoid product.

Result: H^0 = Q, H^{>0} = 0.  The observable monoid has an absorbing element
(the finest partition), so its classifying space is contractible.  This is the
computation behind the negative settlement of T3'.
"""
import itertools
import numpy as np


def build(S, join, N):
    """Matrix of delta_t : C^N -> C^{N+1}."""
    dom = list(itertools.product(S, repeat=N))
    cod = list(itertools.product(S, repeat=N + 1))
    di = {t: i for i, t in enumerate(dom)}
    M = np.zeros((len(cod), len(dom)))
    for r, T in enumerate(cod):
        M[r, di[T[1:]]] += 1                                  # drop S_0
        for i in range(N):                                    # merge adjacent
            merged = T[:i] + (join(T[i], T[i + 1]),) + T[i + 2:]
            M[r, di[merged]] += (-1) ** (i + 1)
        M[r, di[T[:-1]]] += (-1) ** (N + 1)                   # drop S_N
    return M


def cohomology(S, join, upto=3):
    ds = {N: build(S, join, N) for N in range(0, upto + 1)}
    print(f"    |S| = {len(S)}   dim C^N = {[len(S)**N for N in range(upto+2)]}")
    for N in range(0, upto):
        assert np.abs(ds[N + 1] @ ds[N]).max() < 1e-9, "delta^2 != 0"
    res = {}
    for N in range(0, upto + 1):
        dimC = len(S) ** N
        rk_N = np.linalg.matrix_rank(ds[N]) if N in ds else 0
        ker = dimC - rk_N
        im_prev = np.linalg.matrix_rank(ds[N - 1]) if N - 1 in ds and N >= 1 else 0
        res[N] = ker - im_prev
    return res


def key(a):
    return frozenset() if a == '1' else frozenset(a)


def join_(a, b):
    u = key(a) | key(b)
    return '1' if not u else ''.join(sorted(u))


if __name__ == "__main__":
    S2 = ['1', 'X', 'Y', 'XY']
    print("  two binary variables: S = {1, X, Y, XY}, join = union of partitions")
    h = cohomology(S2, join_, upto=3)
    for N in sorted(h):
        print(f"      H^{N}_t = Q^{h[N]}")

    S3 = ['1', 'X', 'Y', 'Z', 'XY', 'XZ', 'YZ', 'XYZ']
    print("\n  three binary variables: S = 8 observables")
    h3 = cohomology(S3, join_, upto=2)
    for N in sorted(h3):
        print(f"      H^{N}_t = Q^{h3[N]}")
