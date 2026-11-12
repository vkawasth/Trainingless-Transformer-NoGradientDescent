"""A pseudo-trefoil with incomplete crossing data (companion note, example in section on knots and trajectories). New module; it modifies
nothing.
The diagram is the closure of the 2-strand braid s^e1 s^e2 s^e3. A resolution is e in {+1, -1}^3; its closure is the right-handed trefoil when
e = (+,+,+), the left-handed trefoil when e = (-,-,-), and the unknot otherwise.
  RES                         the 8 resolutions, lexicographic with +1 first
  knot(e), writhe(e)          knot type ('U', 'R', 'L') and writhe sum(e)
  bracket(e), jones(e)        Kauffman bracket via the Temperley-Lieb algebra TL_2 (s = A + A^-1 E, s^-1 = A^-1 + A E, E^2 = delta E,
                              closure tr(1) = delta^2, tr(E) = delta, normalised by delta) and Jones V = (-A^3)^(-w) <K> at t = A^-4 (sympy)
  walsh(f)                    Walsh-Fourier coefficients f^(S) = mean_e f(e) prod_{i in S} e_i, S a subset of {0, 1, 2}
  constraints(x, contexts)    (A, b): the joint law has the tables of x on each context (a tuple of crossings) and total mass 1
  interval(f, x, contexts)    [min, max] of E_P f over the fibre (LP)
  identified(f, contexts)     the Walsh criterion: every S with f^(S) != 0 lies inside some context (S = {} always)
  spectrum_vertices(x, ctx)   the image of the fibre in the (P(R), P(L)) plane: images of the fibre's vertices (basic solutions)
"""
from __future__ import annotations
import itertools
import numpy as np
import sympy as sp
from scipy.optimize import linprog

RES = list(itertools.product((1, -1), repeat=3))
A_ = sp.Symbol("A"); t_ = sp.Symbol("t")


def writhe(e): return sum(e)
def knot(e): return "R" if sum(e) == 3 else "L" if sum(e) == -3 else "U"


def bracket(e):
    d = -A_ ** 2 - A_ ** -2; c1, cE = sp.Integer(1), sp.Integer(0)                        # element c1 * 1 + cE * E of TL_2
    for s in e:
        a, b = (A_, 1 / A_) if s == 1 else (1 / A_, A_)                                     # s = a 1 + b E
        c1, cE = sp.expand(c1 * a), sp.expand(c1 * b + cE * a + cE * b * d)
    return sp.simplify((c1 * d ** 2 + cE * d) / d)


def jones(e):
    V = sp.expand(sp.simplify((-A_ ** 3) ** (-writhe(e)) * bracket(e)))
    return sp.expand(V.subs(A_, t_ ** sp.Rational(-1, 4)))


def walsh(f):
    out = {}
    for k in range(4):
        for S in itertools.combinations(range(3), k):
            out[S] = sum(f(e) * np.prod([e[i] for i in S]) for e in RES) / 8.0
    return out


def constraints(x, contexts):
    x = np.asarray(x, float); rows = [np.ones(8)]
    for ctx in contexts:
        for vals in itertools.product((1, -1), repeat=len(ctx)):
            rows.append(np.array([1.0 if all(e[i] == v for i, v in zip(ctx, vals)) else 0.0 for e in RES]))
    A = np.array(rows); return A, A @ x


def interval(f, x, contexts):
    A, b = constraints(x, contexts); c = np.array([f(e) for e in RES], float)
    lo = linprog(c, A_eq=A, b_eq=b, bounds=[(0, None)] * 8, method="highs"); hi = linprog(-c, A_eq=A, b_eq=b, bounds=[(0, None)] * 8, method="highs")
    return (lo.fun, -hi.fun) if lo.status == 0 and hi.status == 0 else None


def identified(f, contexts, tol=1e-12):
    return all(abs(v) < tol or len(S) == 0 or any(set(S) <= set(c) for c in contexts) for S, v in walsh(f).items())


def spectrum_vertices(x, contexts, tol=1e-10):
    A, b = constraints(x, contexts); r = np.linalg.matrix_rank(A); keep = []
    for i in range(A.shape[0]):                                                            # independent rows
        if np.linalg.matrix_rank(A[keep + [i]]) > len(keep): keep.append(i)
    A, b = A[keep], b[keep]; pts = []
    for S in itertools.combinations(range(8), r):
        B = A[:, S]
        if abs(np.linalg.det(B)) < 1e-12: continue
        xs = np.linalg.solve(B, b)
        if xs.min() < -tol: continue
        P = np.zeros(8); P[list(S)] = np.clip(xs, 0, None); pts.append((P[RES.index((1, 1, 1))], P[RES.index((-1, -1, -1))]))
    return sorted(set((round(a, 10), round(c, 10)) for a, c in pts))
