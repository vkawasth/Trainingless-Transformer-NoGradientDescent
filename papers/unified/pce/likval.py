"""Likelihood rates as Novikov valuation shifts (section sec:likval). New module; it modifies nothing; uses pce/datawalls.py.
Term k of W_T = sum_k T^(-lam_k) y^(v_k) (T = e^-N) carries a hypothesis P_k (a law on a finite sample space). Data x_1..x_n multiply
term k by its likelihood prod_i P_k(x_i), which is exactly the shift lam -> lam - (n / N) * H_hat, H_hat_k = -(1/n) sum_i log P_k(x_i).
  emp_rates(P, counts)          empirical cross-entropies H_hat_k from a vector of counts over the sample space
  rates(Q, P)                   cross-entropies H(Q, P_k) and divergences D(Q || P_k)
  lam_eff(lam, P, counts, N)    lam - (n / N) H_hat
  circuit_llr(a, P)             the circuit log-likelihood ratio Lambda_a(x) = sum_k a_k log P_k(x): <a, H(Q, P)> = -E_Q Lambda_a
  crossing(v, lam, a, rho)      the data size s* = -c(lam) / <a, rho> at which the outcome wall c = -<a, lam> = 0 is reached (None if never)
  f1_fibres(lam, N, dps)        the torus fibres of F_1 with non-vanishing Floer cohomology (Cho-Oh: critical points of W_T), as the
                                distinct valuations (u1, u2) of the four critical points at T = e^-N, clustered at tolerance 0.02
"""
from __future__ import annotations
import numpy as np
import mpmath as mp
def emp_rates(P, counts):
    counts = np.asarray(counts, float); n = counts.sum(); return np.array([-(counts @ np.log(Pk)) / n for Pk in P])
def rates(Q, P):
    H = np.array([-(Q @ np.log(Pk)) for Pk in P]); HQ = -(Q[Q > 0] @ np.log(Q[Q > 0])); return H, H - HQ
def lam_eff(lam, P, counts, N): return np.asarray(lam, float) - (np.sum(counts) / N) * emp_rates(P, counts)
def circuit_llr(a, P): return sum(ak * np.log(Pk) for ak, Pk in zip(a, P))
def crossing(lam, a, rho):
    c = -float(np.dot(a, lam)); d = float(np.dot(a, rho))
    if d == 0: return None
    s = -c / d; return s if s > 0 else None
def f1_fibres(lam, N=60, dps=80, tol=0.02):
    """critical points of W = T^-l1 y1 + T^-l2 y2 + T^-l3 / (y1 y2) + T^-l4 / y2: with y2 = w^2, y1 = T^((l1 - l3)/2) / w, the quartic
    T^-l2 w^4 - T^-((l1 + l3)/2) w - T^-l4 = 0. Returns the clustered valuations u = -log|y| / N."""
    mp.mp.dps = dps; T = mp.e ** (-N); l1, l2, l3, l4 = [mp.mpf(x) for x in lam]
    roots = mp.polyroots([T ** (-l2), 0, 0, -T ** (-(l1 + l3) / 2), -T ** (-l4)], maxsteps=400, extraprec=4 * dps)
    vals = []
    for w in roots:
        y2 = w ** 2; y1 = T ** ((l1 - l3) / 2) / w; vals.append((float(-mp.log(abs(y1)) / N), float(-mp.log(abs(y2)) / N)))
    clus = []
    for u in vals:
        if not any(abs(u[0] - c[0]) < tol and abs(u[1] - c[1]) < tol for c in clus): clus.append(u)
    return clus, vals
