"""Step 4d of "Probability, Context and Differentiation: The Layers": the decorated order fan, the exact cross term of pooling, Theorem J
of the lab on a statistical fan, and the cones where Gaussian bands are valid. New module.

  face_part(f, F)                  the part of a polynomial on a set F of exponents (the decoration on the cone of the face F)
  edge_limit(cells, F, c)          the limit map of a face: normalise(in_F f(c)) over the cells
  J_stat(D, pi, N, sigma)          the source statistic's mean a = D sqrt(pi (1 - pi) N) / sigma (lab, Theorem J(2), Gaussian model)
  J_impact(D, pi, N, sigma)        the pooled shift in pooled standard errors, pi D sqrt(N) / sigma = a sqrt(pi / (1 - pi))
  sim_J(N, pi, D, sigma, cs, reps, rng)   Monte Carlo in the Gaussian model: the exposure rate P(Z > cs) and the mean pooled shift in SEs
  sim_wald(N, pA, q, reps, rng)    events A ~ Binomial(N, pA), each from source 1 with probability q: coverage of the nominal 95% Wald
                                   interval for pi^A_1 = q, and the probability of no event (the empirical wall)
"""
from __future__ import annotations
from fractions import Fraction as Fr
import math
import numpy as np


def face_part(f, F): return {a: v for a, v in f.items() if a in F and v != 0}


def edge_limit(cells, F, c):
    vals = {}
    for k, f in cells.items():
        tot = Fr(0)
        for a, v in face_part(f, F).items():
            t = v
            for ci, ai in zip(c, a): t *= Fr(ci) ** int(ai)
            tot += t
        if tot: vals[k] = tot
    s = sum(vals.values()); return {k: v / s for k, v in vals.items()}


def J_stat(D, pi, N, sigma=1.0): return D * math.sqrt(pi * (1 - pi) * N) / sigma


def J_impact(D, pi, N, sigma=1.0): return pi * D * math.sqrt(N) / sigma


def sim_J(N, pi, D, sigma, cs, reps, rng):
    nC = max(1, int(round(pi * N))); nO = N - nC
    mC = rng.normal(D, sigma / math.sqrt(nC), reps); mO = rng.normal(0.0, sigma / math.sqrt(nO), reps)
    Z = (mC - mO) / (sigma * math.sqrt(1 / nC + 1 / nO)); pooled = (nC * mC + nO * mO) / N
    return float((Z > cs).mean()), float((pooled / (sigma / math.sqrt(N))).mean())


def sim_wald(N, pA, q, reps, rng, z=1.959964):
    nA = rng.binomial(N, pA, reps); k = rng.binomial(nA, q); ok = nA > 0
    ph = np.where(ok, k / np.maximum(nA, 1), np.nan); se = np.sqrt(np.where(ok, ph * (1 - ph) / np.maximum(nA, 1), np.nan))
    cover = ok & (np.abs(ph - q) <= z * se)
    return float(cover.mean()), float((~ok).mean())
