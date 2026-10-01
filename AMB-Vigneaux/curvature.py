"""Curvature: the fifth component [C]. An external variable (poll sponsorship) bends the reported law; how does it bend?

Each poll i reports a share y_i with natural parameter eta_i = logit p_i (a product of Bernoulli exponential families,
Fisher weight I_i = n_i p_i (1 - p_i), plus over-dispersion). A sponsorship of intensity w moves the whole vector
eta(w). Two geometries:
  e-flat (shift)   eta_i(w) = eta_i + w delta              a straight line in natural coordinates: the log-linear /
                                                            toric description [T] of any covariate; curvature 0.
  m-flat (pull)    p_i(w) = (1 - w) p_i + w q              a straight line in probability (mixture) coordinates,
                                                            pulled toward a target q; curved in natural coordinates.
Jets: velocity eta'(0) is the first-order house effect; acceleration eta''(0) is the second order. Efron's statistical
curvature of the family at w = 0 (with Fisher metric I):
    gamma^2 = [ (eta''^T I eta'')(eta'^T I eta') - (eta'^T I eta'')^2 ] / (eta'^T I eta')^3 .
It is 0 for the shift and positive for the pull. Efron (1975): gamma^2 >= 1/8 is large, i.e. first-order (linear,
asymptotically normal) inference about w is unreliable.
Signature in data: under the shift, the sponsored residual on the logit scale is the same in every race; under the
pull toward q it varies with the race's baseline, d eta = w (q - p) / (p (1 - p)). The slope kappa of sponsored
residuals on the baseline separates the two.
Singular point: if sponsorship is hidden, a 'nonpartisan' poll is honest or bent: a mixture with weight pi. At pi = 0
the tangent space collapses; the likelihood-ratio statistic has a non-standard (chi-bar-squared type) null, so it is
calibrated by parametric bootstrap.
All fits are closed-form weighted least squares or EM with closed-form steps (no gradient steps).
"""
from __future__ import annotations

import numpy as np


def wls(X, y, w):
    WX = X * w[:, None]
    G = X.T @ WX
    beta = np.linalg.lstsq(G, WX.T @ y, rcond=None)[0]
    cov = np.linalg.pinv(G)
    return beta, cov


def efron_gamma2(d1, d2, I):
    a = d1 @ (I * d1); b = d2 @ (I * d2); c = d1 @ (I * d2)
    return float((b * a - c * c) / a ** 3) if a > 0 else np.nan


def pull_jets(p, q):
    """eta(w) = logit((1-w) p + w q): first and second derivatives at w = 0"""
    v = q - p; s = p * (1 - p)
    d1 = v / s
    d2 = -v ** 2 * (1 - 2 * p) / s ** 2
    return d1, d2


def shift_jets(p, delta):
    return np.full_like(p, delta), np.zeros_like(p)


def mixture_lr(r, s2, shifts, iters=300):
    """EM for r_i ~ (1 - sum pi_k) N(0, s2_i) + sum_k pi_k N(shift_k, s2_i), shifts known. Returns pi and the
    log-likelihood ratio against pi = 0."""
    K = len(shifts); pi = np.full(K, 0.05)
    def comp(mu):
        return np.exp(-0.5 * (r - mu) ** 2 / s2) / np.sqrt(2 * np.pi * s2)
    f0 = comp(0.0); fk = np.array([comp(m) for m in shifts])
    for _ in range(iters):
        mix = (1 - pi.sum()) * f0 + (pi[:, None] * fk).sum(0)
        resp = pi[:, None] * fk / mix
        new = resp.mean(1)
        if np.abs(new - pi).max() < 1e-10:
            pi = new; break
        pi = new
    mix = (1 - pi.sum()) * f0 + (pi[:, None] * fk).sum(0)
    return pi, float(2 * (np.log(mix).sum() - np.log(f0).sum()))
