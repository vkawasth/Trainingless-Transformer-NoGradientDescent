"""Statistical bands for decisions that depend on the decoration near the walls (layers paper, Section "Fractures and blind spots").
New module.

N records are drawn from a labelled law; the cells on L x A have probabilities f_c, the rest of the mass is off A. The decision compares two
acts through a gap g on the cells: the act is a iff D = sum_c g_c f_c / P(A) > 0, estimated by D_hat = sum_c g_c n_c / n_A.

  truth(f, g)                        (P(A), D, sigma): the event mass, the conditional gap and its conditional standard deviation
  predicted_gauss(N, f, g)           Phi(-|D| sqrt(N P(A)) / sigma): the limiting probability of the wrong act in the Gaussian regime
  simulate(N, f, g, reps, rng)       Monte Carlo: P(wrong act), P(no event), with multinomial counts (ties D_hat = 0 count as wrong)
  poisson_exact(lam, ell, g, Dsign)  the Poisson-ray limit: cells n_c ~ Poisson(lam ell_c) independent; P(wrong act), P(no event)
"""
from __future__ import annotations
import itertools, math
import numpy as np
from scipy.stats import norm, poisson


def truth(f, g):
    f = np.asarray(f, float); g = np.asarray(g, float); PA = f.sum(); ell = f / PA; D = float(ell @ g)
    return float(PA), D, float(math.sqrt(max(ell @ (g - D) ** 2, 0.0)))


def predicted_gauss(N, f, g):
    PA, D, sd = truth(f, g); return float(norm.cdf(-abs(D) * math.sqrt(N * PA) / sd))


def simulate(N, f, g, reps, rng):
    f = np.asarray(f, float); g = np.asarray(g, float); PA, D, _ = truth(f, g)
    probs = np.r_[f, 1 - f.sum()]; counts = rng.multinomial(int(N), probs, size=reps)[:, :-1]
    nA = counts.sum(axis=1); num = counts @ g; empty = nA == 0
    wrong = np.where(empty, True, (num > 0) != (D > 0)) | (num == 0)
    return float(wrong.mean()), float(empty.mean())


def poisson_exact(lam, ell, g, Dsign, cap=60):
    ell = np.asarray(ell, float); g = np.asarray(g, float); mus = lam * ell
    ranges = [range(0, int(poisson.isf(1e-12, m)) + 3 if m > 0 else 1) for m in mus]; wrong = 0.0; empty = 0.0
    pm = [poisson.pmf(np.arange(len(r)), m) for r, m in zip(ranges, mus)]
    for ns in itertools.product(*ranges):
        pr = float(np.prod([pm[i][n] for i, n in enumerate(ns)]))
        if pr < 1e-16: continue
        tot = sum(ns); val = float(np.dot(g, ns))
        if tot == 0: empty += pr; wrong += pr
        elif val == 0 or (val > 0) != (Dsign > 0): wrong += pr
    return wrong, empty
