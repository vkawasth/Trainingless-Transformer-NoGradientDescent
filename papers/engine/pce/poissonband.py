"""The band on the Poisson ray (layers paper, Step 4d, "A band in the Poisson regime"). New module.

Flags (records in a rare event A) arrive as independent Poisson counts: X_C ~ Poisson(mu_C) from the suspect source C and X_O ~ Poisson(mu_O)
from the genuine sources. Under the null, C flags at the genuine rate, so mu_C = mu_0 = pi mu_O / (1 - pi), pi being C's share of records.
The exact conditional test: given T = X_C + X_O, X_C ~ Binomial(T, pi) under the null; reject when X_C >= k_alpha(T).

  k_alpha(T, pi, alpha)            the smallest k with P(Binomial(T, pi) >= k) <= alpha (T + 1 if none: the test cannot reject)
  blind_total(pi, alpha)           the largest T at which the test can never reject: k_alpha(T) = T + 1
  exposure(mu_C, mu_O, pi, alpha)  P(reject), exactly (truncated sums, error < 1e-12)
  mu_C_max(mu_O, pi, alpha, beta)  the largest mu_C with exposure <= beta (exposure is increasing in mu_C)
  band(mu_O, pi, alpha, beta)      (Delta_max, B_pois): the excess expected flags mu_C_max - mu_0, and the same in pooled standard errors
                                   sqrt(N p) = sqrt(mu_O / (1 - pi))
  gauss_band(pi, alpha, beta)      J's Gaussian band (Phi^-1(1 - alpha) - Phi^-1(1 - beta)) sqrt(pi / (1 - pi))
"""
from __future__ import annotations
import math
from functools import lru_cache
import numpy as np
from scipy.stats import binom, poisson, norm


@lru_cache(maxsize=None)
def k_alpha(T, pi, alpha):
    k = max(0, int(binom.isf(alpha, T, pi)) - 1)              # start just below the quantile, then step to the exact boundary
    while k > 0 and binom.sf(k - 2, T, pi) <= alpha: k -= 1
    while k <= T and binom.sf(k - 1, T, pi) > alpha: k += 1
    return k


def blind_total(pi, alpha):
    T = 0
    while k_alpha(T + 1, pi, alpha) == T + 2: T += 1
    return T


def exposure(mu_C, mu_O, pi, alpha):
    """Poisson splitting: T ~ Poisson(mu_C + mu_O) and, given T, X_C ~ Binomial(T, theta) with theta = mu_C / (mu_C + mu_O)"""
    mu = mu_C + mu_O
    if mu == 0: return 0.0
    th = mu_C / mu; Tm = int(poisson.isf(1e-15, mu)) + 5; Ts = np.arange(Tm + 1)
    ks = np.array([k_alpha(int(T), pi, alpha) for T in Ts])
    return float(np.sum(poisson.pmf(Ts, mu) * binom.sf(ks - 1, Ts, th)))


def mu_C_max(mu_O, pi, alpha, beta, tol=1e-7):
    mu0 = pi * mu_O / (1 - pi); lo = mu0; hi = max(2 * mu0, mu0 + 1.0)
    while exposure(hi, mu_O, pi, alpha) <= beta: hi *= 2
    while hi - lo > tol * max(1.0, hi):
        mid = (lo + hi) / 2
        if exposure(mid, mu_O, pi, alpha) <= beta: lo = mid
        else: hi = mid
    return lo


def band(mu_O, pi, alpha, beta):
    mu0 = pi * mu_O / (1 - pi); d = mu_C_max(mu_O, pi, alpha, beta) - mu0; return d, d / math.sqrt(mu_O / (1 - pi))


def gauss_band(pi, alpha, beta): return (norm.isf(alpha) - norm.isf(beta)) * math.sqrt(pi / (1 - pi))
