"""Exact integer linear algebra: solve A x = b over Z.

The AMB cohomological obstruction is decided by the solvability of an integer
linear system (a compatible family of Z-linear combinations of sections).
Solving over Q would give the wrong answer in general, so we diagonalise A
with unimodular row/column operations (the first half of Smith normal form)
using Python's exact integers.
"""
from __future__ import annotations

from typing import List, Optional, Sequence


def solve_integer(A: Sequence[Sequence[int]], b: Sequence[int]) -> Optional[List[int]]:
    """Return an integer vector x with A x = b, or None if none exists."""
    m = len(A)
    n = len(A[0]) if m else 0
    M = [[int(v) for v in row] for row in A]
    c = [int(v) for v in b]
    # V tracks column operations: x = V y
    V = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    def swap_rows(i, j):
        M[i], M[j] = M[j], M[i]
        c[i], c[j] = c[j], c[i]

    def swap_cols(i, j):
        for row in M:
            row[i], row[j] = row[j], row[i]
        for row in V:
            row[i], row[j] = row[j], row[i]

    def add_row(dst, src, k):  # row_dst += k * row_src
        if k:
            rs, rd = M[src], M[dst]
            for j in range(n):
                if rs[j]:
                    rd[j] += k * rs[j]
            c[dst] += k * c[src]

    def add_col(dst, src, k):  # col_dst += k * col_src
        if k:
            for row in M:
                if row[src]:
                    row[dst] += k * row[src]
            for row in V:
                if row[src]:
                    row[dst] += k * row[src]

    r = 0
    for t in range(min(m, n)):
        # pick the smallest nonzero pivot in the trailing block
        best = None
        for i in range(t, m):
            for j in range(t, n):
                v = M[i][j]
                if v and (best is None or abs(v) < best[0]):
                    best = (abs(v), i, j)
                    if best[0] == 1:
                        break
            if best and best[0] == 1:
                break
        if best is None:
            break
        _, i, j = best
        swap_rows(t, i)
        swap_cols(t, j)
        while True:
            done = True
            # clear column t below the pivot
            for i in range(t + 1, m):
                if M[i][t]:
                    q = M[i][t] // M[t][t]
                    add_row(i, t, -q)
                    if M[i][t]:
                        done = False
                        if abs(M[i][t]) < abs(M[t][t]):
                            swap_rows(t, i)
            # clear row t right of the pivot
            for j in range(t + 1, n):
                if M[t][j]:
                    q = M[t][j] // M[t][t]
                    add_col(j, t, -q)
                    if M[t][j]:
                        done = False
                        if abs(M[t][j]) < abs(M[t][t]):
                            swap_cols(t, j)
            if done:
                break
        r = t + 1

    # now M is diagonal on the first r entries: D y = c
    y = [0] * n
    for i in range(r):
        d = M[i][i]
        if c[i] % d:
            return None
        y[i] = c[i] // d
    for i in range(r, m):
        if c[i]:
            return None
    return [sum(V[i][k] * y[k] for k in range(n)) for i in range(n)]


def solve_mod_p(A: Sequence[Sequence[int]], b: Sequence[int], p: int) -> Optional[List[int]]:
    """Return x with A x = b over the field Z/p (p prime), or None."""
    m = len(A)
    n = len(A[0]) if m else 0
    M = [[int(v) % p for v in row] + [int(bi) % p] for row, bi in zip(A, b)]
    piv_cols, r = [], 0
    for c in range(n):
        piv = next((i for i in range(r, m) if M[i][c]), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        inv = pow(M[r][c], p - 2, p)
        M[r] = [(v * inv) % p for v in M[r]]
        for i in range(m):
            if i != r and M[i][c]:
                f = M[i][c]
                M[i] = [(vi - f * vr) % p for vi, vr in zip(M[i], M[r])]
        piv_cols.append(c)
        r += 1
        if r == m:
            break
    if any(M[i][n] for i in range(r, m)):
        return None
    x = [0] * n
    for i, c in enumerate(piv_cols):
        x[c] = M[i][n]
    return x


def solve_rational(A: Sequence[Sequence[int]], b: Sequence[int]):
    """Exact solvability of A x = b over Q (Fractions); returns x or None."""
    from fractions import Fraction
    m = len(A)
    n = len(A[0]) if m else 0
    M = [[Fraction(int(v)) for v in row] + [Fraction(int(bi))] for row, bi in zip(A, b)]
    piv_cols, r = [], 0
    for c in range(n):
        piv = next((i for i in range(r, m) if M[i][c] != 0), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        pv = M[r][c]
        M[r] = [v / pv for v in M[r]]
        for i in range(m):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [vi - f * vr for vi, vr in zip(M[i], M[r])]
        piv_cols.append(c)
        r += 1
        if r == m:
            break
    if any(M[i][n] != 0 for i in range(r, m)):
        return None
    x = [Fraction(0)] * n
    for i, c in enumerate(piv_cols):
        x[c] = M[i][n]
    return x
