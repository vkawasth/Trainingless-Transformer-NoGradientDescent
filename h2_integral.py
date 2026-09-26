#!/usr/bin/env python3
"""Integral version of h2_rational.py: same delta_t complex, computed over Z.

Rational vanishing cannot see torsion, and a torsion class in H^2 would be
exactly the kind of obstruction T3' needed.  Smith normal form exposes it.

Result: H^0 = Z, H^1 = H^2 = 0, and NO torsion.  T3' is settled negatively
over Z as well as over Q.
"""
import itertools
import numpy as np
from sympy import Matrix, ZZ
from sympy.matrices.normalforms import smith_normal_form as snf

from h2_rational import build, join_


def integral_cohomology(S, join, upto=2):
    d = {N: build(S, join, N) for N in range(0, upto + 2)}
    for N in range(0, upto + 1):
        assert np.abs(d[N + 1] @ d[N]).max() == 0, "delta^2 != 0"
    out = {}
    for N in range(0, upto + 1):
        dimC = len(S) ** N
        rkN = int(round(np.linalg.matrix_rank(d[N].astype(float)))) if dimC else 0
        rkP = int(round(np.linalg.matrix_rank(d[N - 1].astype(float)))) if N >= 1 else 0
        free = dimC - rkN - rkP
        tors = []
        if N >= 1:
            S_ = snf(Matrix(d[N - 1].astype(int).tolist()), domain=ZZ)
            divs = [abs(S_[i, i]) for i in range(min(S_.rows, S_.cols)) if S_[i, i] != 0]
            tors = [x for x in divs if x not in (0, 1)]
        out[N] = (free, tors)
    return out


if __name__ == "__main__":
    for label, S, upto in [
        ("two binary variables: S = {1, X, Y, XY}", ['1', 'X', 'Y', 'XY'], 2),
        ("three binary variables: S = 8 observables",
         ['1', 'X', 'Y', 'Z', 'XY', 'XZ', 'YZ', 'XYZ'], 2),
    ]:
        print(f"  {label}")
        res = integral_cohomology(S, join_, upto=upto)
        for N in sorted(res):
            free, tors = res[N]
            t = f"  torsion: {tors}" if tors else "  (no torsion)"
            print(f"      H^{N}(S;Z) = Z^{free}{t}")
        print()
