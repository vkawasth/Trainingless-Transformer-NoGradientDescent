"""D1: soft evidence as a shift of the Novikov exponents, and the walls it can cross (companion note kahler/conditioned_states_kahler.tex,
section "Soft evidence moves lambda"). New module; it imports pce/kahlerbridge.py and does not modify it.

Convention. W_T(y) = sum_k T^(-lam_k) y^(v_k) with T = e^(-N) (N the reference scale of the model's exponents). Evidence that multiplies term k
by e^(-n rho_k) per n observations multiplies it by T^((n / N) rho_k), so lam_eff = lam - s rho with s = n / N.

  shift(lam, r, s)                 lam - s r
  balance_lp(v, lam)               h*(lam) = -max {<q, lam> : q a law on the facets with mean exponent 0} (LP duality), an optimal q,
                                   and the support of the optimal face (the points of the candidate cell sigma_0)
  realized_sets(v, lam)            every leading set K(u) that occurs, with the dimension of its tropical cell {u : K(u) = K}
  argmax_dim(v, lam)               the affine dimension of argmax min_k l_k
  circuits(v)                      a basis of the affine dependences a (sum a_k v_k = 0, sum a_k = 0); integer and primitive when one-dimensional
  first_wall(v, lam, r, smax)      the first s in (0, smax] at which the optimal face of the balance LP changes along lam - s r (bisection)
  min_evidence(v, lam)             the least s over rate vectors with range max r - min r <= 1 that reaches the wall, for a single circuit
"""
from __future__ import annotations
import itertools
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import linprog
from scipy.linalg import null_space


def shift(lam, r, s): return np.array(lam, float) - s * np.array(r, float)


def balance_lp(v, lam, tol=1e-9):
    V = np.array(v, float).T; m = len(lam); lam = np.array(lam, float); A = np.r_[V, np.ones((1, m))]; b = np.r_[np.zeros(V.shape[0]), 1.0]
    r = linprog(-lam, A_eq=A, b_eq=b, bounds=[(0, None)] * m, method="highs"); best = -r.fun
    supp = []
    for k in range(m):                                  # maximise q_k over the optimal face
        c = np.zeros(m); c[k] = -1
        rk = linprog(c, A_eq=np.r_[A, lam[None, :]], b_eq=np.r_[b, best], bounds=[(0, None)] * m, method="highs")
        if rk.status == 0 and -rk.fun > tol: supp.append(k)
    return -best, r.x, tuple(supp)


def realized_sets(v, lam, margin=1e-7):
    """K occurs iff some u has l_k(u) = t for k in K and l_j(u) >= t + margin otherwise; the cell dimension is n - rank of the differences"""
    V = np.array(v, float); n = V.shape[1]; m = len(v); lam = np.array(lam, float); out = []
    for size in range(2, m + 1):
        for K in itertools.combinations(range(m), size):
            Ae = [np.r_[V[k], -1.0] for k in K]; be = [lam[k] for k in K]
            Au = [np.r_[-V[j], 1.0] for j in range(m) if j not in K]; bu = [-lam[j] - margin for j in range(m) if j not in K]
            r = linprog(np.zeros(n + 1), A_eq=np.array(Ae), b_eq=np.array(be), A_ub=np.array(Au) if Au else None, b_ub=np.array(bu) if bu else None,
                        bounds=[(None, None)] * (n + 1), method="highs")
            if r.status == 0:
                D = V[list(K)][1:] - V[K[0]]; out.append((K, n - np.linalg.matrix_rank(D)))
    return out


def argmax_dim(v, lam, trials=24, seed=0):
    V = np.array(v, float); n = V.shape[1]; m = len(v); lam = np.array(lam, float)
    h, _, _ = balance_lp(v, lam); rng = np.random.default_rng(seed); P = []
    for _ in range(trials):
        d = rng.normal(size=n); r = linprog(-d, A_ub=-V, b_ub=-(lam + h) + 1e-12, bounds=[(None, None)] * n, method="highs"); P.append(r.x)
    P = np.array(P); return int(np.linalg.matrix_rank(P[1:] - P[0], tol=1e-6))


def circuits(v):
    V = np.array(v, float).T; N = null_space(np.r_[V, np.ones((1, V.shape[1]))])
    if N.shape[1] == 1:
        a = N[:, 0] / np.min(np.abs(N[np.abs(N[:, 0]) > 1e-9, 0])); fr = [Fr(x).limit_denominator(60) for x in a]
        from math import lcm, gcd
        L = lcm(*[f.denominator for f in fr]); ints = [int(f * L) for f in fr]; g = 0
        for x in ints: g = gcd(g, abs(x))
        ints = [x // g for x in ints]
        if ints[0] < 0: ints = [-x for x in ints]
        return [ints]
    return [list(N[:, j]) for j in range(N.shape[1])]


def first_wall(v, lam, r, smax=5.0, n=2001, tol=1e-10):
    s0 = balance_lp(v, lam)[2]; ss = np.linspace(0, smax, n)
    for a, b in zip(ss[1:-1], ss[2:]):
        if balance_lp(v, shift(lam, r, b))[2] != s0:
            lo, hi = a, b
            while hi - lo > tol:
                mid = (lo + hi) / 2
                if balance_lp(v, shift(lam, r, mid))[2] != s0: hi = mid
                else: lo = mid
            return (lo + hi) / 2
    return None


def min_evidence(v, lam):
    """single circuit a: the wall is c(lam) = -sum a_k lam_k = 0 and c(lam - s r) = c + s <a, r>; with range(r) <= 1 the best <a, r> is
    sum a_k^+ (= half the l1 norm, since sum a_k = 0), so the least s is |c| / (||a||_1 / 2), attained at r = 1[a > 0] (or 1[a < 0])"""
    a = np.array(circuits(v)[0], float); c = -a @ np.array(lam, float); half = np.abs(a).sum() / 2
    r = (a > 0).astype(float) if c < 0 else (a < 0).astype(float); return abs(c) / half, r, c, a
