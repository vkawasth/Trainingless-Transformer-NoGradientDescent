"""A small dagger category for finite probability, and how little it takes to change a composed decision (companion note,
sections 12.13-12.14). New module; it modifies nothing and uses pce/selcomp.py for selection.

Couplings are lists of lists of Fractions with full-support marginals.
  cpl(p, q, t)                 the 2 x 2 coupling of margins (p, 1-p), (q, 1-q) with 00 entry t
  kron, T, merge_cols, merge_rows, diag   tensor product, transpose, coarse-graining of the last / first index, diagonal coupling
  markov(g1, g2)               g1 D^-1 g2, exact (zero-mass middle states are skipped)
  sel(g1, g2)                  selection (facet-leximin), exact
  identified(g1, g2)           for binary end spaces: [lo, hi], the set of 00 entries of all composites compatible with g1, g2
  flip_edit(f, t0, lo, hi, c)  the smallest margin-preserving edit s (t0 -> t0 + s) with f(t0 + s) = c, for f piecewise linear or linear
UMBRELLA                       the worked example: forecast -> umbrella -> wet, margins (2/5, 1/3, 3/7), 00 entries 13/60, 7/30
"""
from __future__ import annotations
from fractions import Fraction as Fr
from . import selcomp as S


def cpl(p, q, t): return [[Fr(t), Fr(p) - Fr(t)], [Fr(q) - Fr(t), 1 - Fr(p) - Fr(q) + Fr(t)]]
def T(A): return [list(r) for r in zip(*A)]
def diag(w): return [[Fr(w[i]) if i == j else Fr(0) for j in range(len(w))] for i in range(len(w))]
def kron(A, B): return [[A[i][j] * B[k][l] for j in range(len(A[0])) for l in range(len(B[0]))] for i in range(len(A)) for k in range(len(B))]
def merge_cols(A, groups): return [[sum(r[j] for j in g) for g in groups] for r in A]
def merge_rows(A, groups): return [[sum(A[i][j] for i in g) for j in range(len(A[0]))] for g in groups]
def eq(A, B): return len(A) == len(B) and all(A[i][j] == B[i][j] for i in range(len(B)) for j in range(len(B[0])))
def rows(A): return [sum(r) for r in A]
def cols(A): return [sum(A[i][j] for i in range(len(A))) for j in range(len(A[0]))]
def show(A): return [[str(x) for x in r] for r in A]


def markov(g1, g2):
    q = cols(g1)
    return [[sum(g1[i][j] * g2[j][k] / q[j] for j in range(len(q)) if q[j] != 0) for k in range(len(g2[0]))] for i in range(len(g1))]


def sel(g1, g2): return [[Fr(x) for x in r] for r in S.select([g1, g2])]


def identified(g1, g2):
    """X_0, X_2 binary: the composites compatible with g1, g2 are the end marginals of the fibre, a product of slice segments [L_b, U_b];
    their 00 entries fill [sum L_b, sum U_b] (Proposition 'what the data determine')"""
    lo = hi = Fr(0)
    for b in range(len(g2)):
        q = g1[0][b] + g1[1][b]; r, s = g1[0][b], g2[b][0]; lo += max(Fr(0), r + s - q); hi += min(r, s)
    return lo, hi


def flip_edit(f, t0, lo, hi, c, iters=90):
    """smallest |s| with t0 + s in [lo, hi] and f(t0 + s) = c, searching both directions; f piecewise linear (finitely many pieces).
    Bisection to a bracket inside one piece, then the exact root of that piece, checked exactly. Returns s or None."""
    best = None
    for end in (lo, hi):
        a, b = Fr(t0), Fr(end); fa, fb = f(a) - c, f(b) - c
        if fa == 0: return Fr(0)
        if (fa > 0) == (fb > 0) and fb != 0: continue
        # first crossing from t0: walk on a fine grid, then bisect inside the first bracket
        n = 400; prev = a
        for k in range(1, n + 1):
            x = a + (b - a) * k / n
            if (f(x) - c == 0) or ((f(x) - c > 0) != (fa > 0)): b = x; break
            prev = x
        a = prev
        for _ in range(iters):
            m = (a + b) / 2
            if (f(m) - c > 0) == (fa > 0) and f(m) != c: a = m
            else: b = m
        # exact root on the piece through a and b: a affine piece through two points well inside the bracket
        x0, x1 = a, b; y0, y1 = f(x0) - c, f(x1) - c
        root = x0 - y0 * (x1 - x0) / (y1 - y0) if y1 != y0 else x0
        root = Fr(root).limit_denominator(10 ** 9)
        if f(root) != c: continue
        s = root - Fr(t0)
        if best is None or abs(s) < abs(best): best = s
    return best


# ---------------------------------------------------------------- the worked example
UMBRELLA = dict(p=Fr(2, 5), q=Fr(1, 3), r=Fr(3, 7), t1=Fr(13, 60), t2=Fr(7, 30), tau=Fr(1, 2))


def umbrella():
    E = UMBRELLA; return cpl(E["p"], E["q"], E["t1"]), cpl(E["q"], E["r"], E["t2"])
