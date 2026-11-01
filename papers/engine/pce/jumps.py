"""Strata and jumps (layers paper, Step 2, Section "Strata and jumps"): conditioning a labelled law, its provenance support, the fibres
of forgetting, and the behaviour of conditioning along paths (regular regime, source extinction, the wall, kinks). New module. Exact
arithmetic (Fraction) unless stated.

A labelled law is a dict {(l, x): Fraction}; A is a predicate on outcomes x.

  mass_A(Lam, A)                Lambda(L x A)
  cond_lab(Lam, A)              the labelled conditional C_{L x A}(Lambda) (None on the wall); its L-marginal is pi^A and its rows,
                                normalised, are the C_A(p_l)
  support_A(Lam, A)             L_A(Lambda) = {l : Lambda(l, A) > 0} = {l : pi_l p_l(A) > 0}
  forget(Lam)                   the pooled law (D eps)
  fibre_lift(Lp, P, A, l0)      a labelled law Lambda with forget(Lambda) = P and cond_lab(Lambda) = Lp, for any Lp with forget(Lp) = C_A(P):
                                Lambda = P(A) Lp + delta_{l0} (x) (P restricted off A)
  poly_eval(coeffs, t)          sum_j coeffs[j] t^j for a list of labelled (or plain) dicts
  jet_limit(coeffs, A)          the order m of t -> Lambda_t(L x A) at t = 0+ and the limit of cond_lab along the path: C_{L x A}(c_m)
"""
from __future__ import annotations
from fractions import Fraction as Fr


def mass_A(Lam, A): return sum((v for (l, x), v in Lam.items() if A(x)), Fr(0))


def cond_lab(Lam, A):
    a = mass_A(Lam, A)
    if a == 0: return None
    return {(l, x): v / a for (l, x), v in Lam.items() if A(x) and v != 0}


def support_A(Lam, A): return {l for (l, x), v in Lam.items() if A(x) and v > 0}


def forget(Lam):
    out = {}
    for (l, x), v in Lam.items():
        if v: out[x] = out.get(x, Fr(0)) + v
    return out


def weights(Lam):
    out = {}
    for (l, x), v in Lam.items():
        if v: out[l] = out.get(l, Fr(0)) + v
    return out


def fibre_lift(Lp, P, A, l0):
    a = sum((v for x, v in P.items() if A(x)), Fr(0)); out = {k: a * v for k, v in Lp.items()}
    for x, v in P.items():
        if not A(x) and v: out[(l0, x)] = out.get((l0, x), Fr(0)) + v
    return out


def poly_eval(coeffs, t):
    out = {}
    for j, c in enumerate(coeffs):
        for k, v in c.items(): out[k] = out.get(k, Fr(0)) + v * t ** j
    return out


def jet_limit(coeffs, A):
    for m, c in enumerate(coeffs):
        cA = {k: v for k, v in c.items() if A(k[1] if isinstance(k, tuple) else k) and v != 0}
        if cA:
            s = sum(cA.values()); return m, {k: v / s for k, v in cA.items()}
    return None, None
