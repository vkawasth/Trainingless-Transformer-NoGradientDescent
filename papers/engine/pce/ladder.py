"""The ladder of base semirings: over which number system do the equations of the evidence have points?

  C  (complex)  >  R  (signed)  >  R>=0  (probability)

Level one (a law on outcomes with given window tables; linear equations A mu = t, sum mu = 1):
  linear_gap      min ||A mu - t||_1 over signed mu with total mass 1: 0 iff an R-point exists (the tables agree on overlaps,
                  i.e. no signalling / no direct influence); positive = the bottom rung (no points at all, over R or C)
  negativity      min total negative mass ||mu^-|| over signed mu with A mu = t (None if linear_gap > 0): 0 iff an R>=0-point exists
  lam             the noncontextual fraction lam* (largest sub-probability with marginals <= t)
Level two (two latent types, three binary variables; closed form via mean and central moments):
  Delta = m12 m13 m23,  Det = m123^2 + 4 Delta (Cayley's hyperdeterminant)
  rung            "R>=0"  genuine mixture with w in (0,1) and a in [0,1]^6;  "R>=0 off box": w in (0,1) but some a outside [0,1];
                  "R": real, w outside [0,1] (Det > 0 > Delta);  "C": complex-conjugate pair (Det < 0);  "none": not in the image
  negativity2     the negative weight of the signed explanation, (|m123|/sqrt(Det) - 1)/2 for Det > 0 > Delta
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from .coherence import _constraints
from .integrate import gluable_fraction


def level1(domains, windows, tol=1e-9):
    A, t, atoms, index = _constraints(domains, windows); m, n = A.shape
    # linear gap: mu = p - q signed, slack s+ s-: min sum s+ + s- s.t. A(p - q) - s+ + s- = t, sum(p - q) = 1
    c = np.r_[np.zeros(2 * n), np.ones(2 * m)]
    Aeq = np.vstack([np.hstack([A, -A, -np.eye(m), np.eye(m)]), np.r_[np.ones(n), -np.ones(n), np.zeros(2 * m)]])
    r = linprog(c, A_eq=Aeq, b_eq=np.r_[t, 1.0], bounds=(0, None), method="highs"); gap = float(r.fun)
    neg = None
    if gap < tol:
        r2 = linprog(np.r_[np.zeros(n), np.ones(n)], A_eq=np.vstack([np.hstack([A, -A]), np.r_[np.ones(n), -np.ones(n)]]),
                     b_eq=np.r_[t, 1.0], bounds=(0, None), method="highs")
        neg = float(r2.fun)
    lam = gluable_fraction(domains, windows).lam
    rung = "none" if gap > tol else ("R" if neg > tol else "R>=0")
    return {"linear_gap": gap, "negativity": neg, "lam": lam, "rung": rung}


def moments3(P):
    P = np.asarray(P, float).reshape(2, 2, 2); mu = np.array([P.sum(axis=(1, 2))[1], P.sum(axis=(0, 2))[1], P.sum(axis=(0, 1))[1]])
    x = np.array(np.meshgrid([0, 1], [0, 1], [0, 1], indexing="ij")); c = x - mu[:, None, None, None]
    m = lambda *S: float((P * np.prod([c[i] for i in S], axis=0)).sum())
    return mu, {"12": m(0, 1), "13": m(0, 2), "23": m(1, 2), "123": m(0, 1, 2)}


def level2_m3(P, tol=1e-12):
    mu, M = moments3(P); D = M["12"] * M["13"] * M["23"]; Det = M["123"] ** 2 + 4 * D
    out = {"Delta": D, "Det": Det, "mu": mu.tolist(), "negativity2": 0.0, "imag": 0.0, "w": None, "a": None}
    pair = [M["23"], M["13"], M["12"]]                                     # m_jk for i = 1, 2, 3
    zero = [abs(x) < tol for x in pair]
    if abs(Det) < 1e-12 * (M["123"] ** 2 + 4 * abs(D)) or (any(zero) and not all(zero)):
        out["rung"] = "none"; return out                                 # closure points (or non-flat strata; not classified here)
    if Det < 0:                                                            # complex-conjugate pair: measure the imaginary parts
        p = D / Det; q = np.sqrt(complex(1 - 4 * p)); w = np.array([(1 - q) / 2, (1 + q) / 2])
        if abs(q) > tol: d = np.array([M["123"] / (q * x) for x in pair])
        else:            d = np.sqrt(np.array([M["12"] * M["13"] / (p * M["23"]), M["12"] * M["23"] / (p * M["13"]), M["13"] * M["23"] / (p * M["12"])], complex))
        out["rung"] = "C"; out["imag"] = float(max(np.abs(w.imag).max(), np.abs(d.imag).max())); out["w"] = [str(x) for x in w]; return out
    p = D / Det; q = np.sqrt(max(0.0, 1 - 4 * p)); w = np.array([(1 - q) / 2, (1 + q) / 2])
    if q > tol: d = np.array([M["123"] / (q * x) for x in pair])
    else:       d = np.array([np.sqrt(M[a] * M[b] / (p * M[c])) for a, b, c in [("12", "13", "23"), ("12", "23", "13"), ("13", "23", "12")]]) * \
                    np.sign([1, M["12"], M["13"]])
    a1 = mu + w[1] * d; a2 = mu - w[0] * d; out["w"] = w.tolist(); out["a"] = [a1.tolist(), a2.tolist()]
    if D < 0:
        out["rung"] = "R"; out["negativity2"] = float(-w.min()); return out
    out["rung"] = "R>=0" if (np.all(a1 >= -1e-12) and np.all(a1 <= 1 + 1e-12) and np.all(a2 >= -1e-12) and np.all(a2 <= 1 + 1e-12)) else "R>=0 off box"
    return out
