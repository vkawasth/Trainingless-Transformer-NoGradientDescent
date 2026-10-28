"""Data injection on the hiding segment (lab paper, Part II). Genuine records follow the two-type law P; an adversary adds whole records
from a law Q of its choice, an epsilon share of the data, with no relation between injected and genuine points (a coproduct of samples).
Every window then sees the marginal of  P_eps = (1 - eps) P + eps Q.

  mix(P, Q, eps)                       the observed law
  chi(m, S)                            the Walsh character prod_{i in S}(2 x_i - 1) on {0,1}^m
  stealth_range(P, S)                  [s_lo, s_hi]: Q = P + s chi_S is a law (for |S| = m = 3 these are all injections that leave every
                                       single and pair table unchanged)
  pipeline(P, acts, u0, h=0.0)         the regulator's population pipeline on m = 3: (mu, v, R, rhat, pre-probe act, post-probe act)
  constraints(P, acts, u0, a)          the set-rule constraints for act a on the whole segment R (pre-probe): W_a - u0, W_a - W_b
  influence(F, P, eps=1e-6)            IF(x) = d/d eps F((1 - eps) P + eps delta_x) at 0, for every outcome x (vector F allowed)
  first_order_budget(P, F)             min over j, x with IF_j(x) < 0 of F_j(P) / (-IF_j(x)): the vertex (point-mass) attack
  exact_budget(P, Q, flipped)          smallest eps at which flipped(mix(P, Q, eps)) holds (scan + bisection)
  tetrads(P)                           the two tetrads of the pair covariances (m = 4): M01 M23 - M02 M13, M01 M23 - M03 M12
  tetrad_jacobian(P)                   derivative of the tetrads along delta_x - P, for every x (2 x 2^m)
  stealth_tilt_lp(P, tilt, N, delta)   max over laws Q of tilt . Q subject to |N (Q - P)|_inf <= delta  (a linear programme)
"""
from __future__ import annotations
import itertools
import numpy as np
from numpy.polynomial import polynomial as Pn
from scipy.optimize import linprog
from .lab import moments, fit_v
from . import segment_decision as SD


def outcomes(m): return list(itertools.product((0, 1), repeat=m))


def as_vec(P): return np.array([P[x] for x in outcomes(P.ndim)])


def as_law(p, m):
    P = np.zeros((2,) * m)
    for k, x in enumerate(outcomes(m)): P[x] = p[k]
    return P


def mix(P, Q, eps): return (1 - eps) * P + eps * Q


def chi(m, S): return np.array([np.prod([2 * x[i] - 1 for i in S]) for x in outcomes(m)], float)


def stealth_range(P, S):
    p = as_vec(P); c = chi(P.ndim, S); return float(max(-p[c > 0])), float(min(p[c < 0]))


def pipeline(P, acts, u0, h=0.0):
    """m = 3, exact tables of P: mu, v (closed form), the segment R, rhat, the pre-probe set-rule act on R and the post-probe act on
    [rhat - h, rhat + h] cap R. Returns None when the pair tables admit no two-type explanation."""
    e = SD.from_tables(SD.exact_tables(P))
    if e is None: return None
    mu, v, rhat, tau, R = e; ps = [SD.U_poly(mu, v, u, range(3)) for u in acts]
    pre = SD.set_rule(ps, R, u0); post = SD.set_rule(ps, SD.clip((rhat - h, rhat + h), R), u0) if R[0] <= rhat <= R[1] else "empty"
    return dict(mu=mu, v=v, R=R, rhat=rhat, tau=tau, polys=ps, pre=pre, post=post)


def constraints(P, acts, u0, a):
    out = pipeline(P, acts, u0)
    if out is None: return None
    W = [SD.worst(p, out["R"]) for p in out["polys"]]
    return np.array([W[a] - u0] + [W[a] - W[b] for b in range(len(W)) if b != a])


def influence(F, P, eps=1e-6):
    m = P.ndim; rows = []
    for x in outcomes(m):
        D = np.zeros_like(P); D[x] = 1.0
        rows.append((np.asarray(F(mix(P, D, eps))) - np.asarray(F(mix(P, D, -eps)))) / (2 * eps))
    return np.array(rows)                                     # shape (2^m, len(F))


def first_order_budget(P, F):
    g = np.atleast_1d(np.asarray(F(P), float)); IF = influence(F, P).reshape(2 ** P.ndim, -1); best = (np.inf, None, None)
    for k in range(IF.shape[0]):
        for j in range(IF.shape[1]):
            if IF[k, j] < 0 and g[j] / (-IF[k, j]) < best[0]: best = (g[j] / (-IF[k, j]), k, j)
    return best


def exact_budget(P, Q, flipped, emax=0.5, grid=500, tol=1e-10):
    es = np.linspace(0, emax, grid + 1)[1:]; prev = 0.0
    for e in es:
        if flipped(mix(P, Q, e)):
            lo, hi = prev, e
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if flipped(mix(P, Q, mid)): hi = mid
                else: lo = mid
                if hi - lo < tol: break
            return hi
        prev = e
    return np.inf


def pair_cov(P):
    mu, M, _ = moments(P); return M


def tetrads(P):
    M = pair_cov(P); return np.array([M[0, 1] * M[2, 3] - M[0, 2] * M[1, 3], M[0, 1] * M[2, 3] - M[0, 3] * M[1, 2]])


def tetrad_jacobian(P):
    return influence(tetrads, P).T                           # 2 x 2^m: row j is d T_j along delta_x - P


def stealth_tilt_lp(P, tilt, N, delta):
    """maximise tilt . (Q - P) over laws Q with |N (Q - P)|_inf <= delta, where tilt and the rows of N are the influence vectors
    (derivatives along delta_x - P). Since sum_x Q(x) = 1, the first-order change along Q - P is sum_x Q(x) IF(x)."""
    k = len(tilt); A = np.vstack([N, -N]); b = np.full(2 * N.shape[0], delta)
    res = linprog(-np.asarray(tilt), A_ub=A, b_ub=b, A_eq=np.ones((1, k)), b_eq=[1.0], bounds=[(0, None)] * k, method="highs")
    return (float(-res.fun), res.x) if res.status == 0 else (None, None)
