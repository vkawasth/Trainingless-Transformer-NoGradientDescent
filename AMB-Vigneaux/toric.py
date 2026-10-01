"""The toric geometry of a sphere cover: the free top-order direction seen in two dual coordinates, kept as separate
components so that each one's effect can be read off on its own.

Setting: n binary measurements observed n-1 at a time (contexts = all (n-1)-subsets; nerve = S^{n-2}). The data fix
every (n-1)-way margin. What is left free is ONE coordinate, the top-order interaction. It has two faces:

  [F] fibre (mixture coordinate, linear)   the consistent global laws form the segment P0 + t*chi in the simplex,
                                            chi = (-1)^{x1+..+xn}; m = <chi, P> ranges over [m_lo, m_hi].
                                            Lower-order data determine the segment; nothing on it is preferred.
  [T] toric variety (exponential coordinate)  theta = 2^-n sum_x chi(x) log P(x), the top log-linear interaction.
                                            The model 'no top interaction' is theta = 0, i.e. the single binomial
                                            prod_{chi=+1} p = prod_{chi=-1} p (degree 2^{n-1}). Along the fibre theta is
                                            strictly increasing from -inf to +inf, so the segment meets the variety in
                                            exactly one point (Birch): the maximum-entropy gluing, which IPF finds.
  [I] interaction (needs the full joint)    theta_hat from counts on units where all n are observed, with the
                                            delta-method standard error 2^-n sqrt(sum 1/n_x). Where the observed law
                                            sits on the segment (t_obs) and how far it is from the Birch point.
  [S] strata (support)                      faces of the closure of the toric model: facial sets of the design of the
                                            (n-1)-margins; the exact AMB toggle radius of each cell (strata.exact_toggle)
                                            against the saturated radius -log(1 - p_c).
Contextual families (no positive point on the segment) have an empty [F]; then [T] and [S] are evaluated at the
nearest consistent family and flagged.
"""
from __future__ import annotations

import itertools
from typing import Optional

import numpy as np

from .strata import design, exact_toggle, ipf


def cells(n):
    return list(itertools.product((0, 1), repeat=n))


def chi(n):
    return np.array([(-1.0) ** sum(x) for x in cells(n)])


def binomial(n):
    """exponents of the defining binomial: cells with chi = +1 on one side, chi = -1 on the other"""
    c = cells(n); x = chi(n)
    return dict(plus=[c[i] for i in range(len(c)) if x[i] > 0], minus=[c[i] for i in range(len(c)) if x[i] < 0], degree=2 ** (n - 1))


def margins(n):
    return [m for m in itertools.combinations(range(n), n - 1)]


def theta(p):
    n = int(np.log2(len(p)))
    return float(chi(n) @ np.log(np.maximum(p, 1e-300)) / 2 ** n)


def fibre(P0):
    """[F] the segment P0 + t chi inside the simplex: t-range, mixture-coordinate range"""
    n = int(np.log2(len(P0))); x = chi(n)
    t_lo = max(-P0[i] for i in range(len(P0)) if x[i] > 0)          # p_i + t >= 0 on chi=+1 cells
    t_hi = min(P0[i] for i in range(len(P0)) if x[i] < 0)           # p_i - t >= 0 on chi=-1 cells
    m0 = float(x @ P0)
    return dict(t_lo=float(t_lo), t_hi=float(t_hi), m_lo=m0 + 2 ** n * t_lo, m_hi=m0 + 2 ** n * t_hi, nonempty=bool(t_lo <= t_hi))


def birch(P0, tol=1e-14):
    """[T] the unique point of the segment on the toric variety theta = 0 (bisection; theta is increasing in t)"""
    n = int(np.log2(len(P0))); x = chi(n); F = fibre(P0)
    lo, hi = F["t_lo"] + 1e-15, F["t_hi"] - 1e-15
    for _ in range(200):
        mid = (lo + hi) / 2
        if theta(P0 + mid * x) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return P0 + (lo + hi) / 2 * x, (lo + hi) / 2


def theta_hat(counts, smooth=0.5):
    """[I] top interaction from full-joint counts, with the delta-method standard error"""
    N = np.asarray(counts, float) + smooth; n = int(np.log2(len(N)))
    th = float(chi(n) @ np.log(N) / 2 ** n); se = float(np.sqrt((1.0 / N).sum()) / 2 ** n)
    return dict(theta=th, se=se, lo=th - 1.96 * se, hi=th + 1.96 * se, z=th / se)


def margins_of(p, n):
    T = np.asarray(p).reshape((2,) * n)
    return {m: T.sum(axis=tuple(a for a in range(n) if a not in m)) for m in margins(n)}


def point_from_margins(p, n):
    """IPF from the uniform table to the (n-1)-margins of p: the maximum-entropy member of the fibre"""
    T = np.asarray(p).reshape((2,) * n)
    return ipf(T, margins(n), iters=5000, tol=1e-13).ravel() / T.sum()


def strata_report(P, cells_idx=None, time_limit=30):
    """[S] exact AMB toggle radius of each cell in the no-top-interaction model vs the saturated radius"""
    n = int(np.log2(len(P))); A, _ = design((2,) * n, margins(n)); out = []
    for c in (range(len(P)) if cells_idx is None else cells_idx):
        r = exact_toggle(A, P, c, time_limit=time_limit)
        out.append(dict(cell=cells(n)[c], p=float(P[c]), saturated=float(-np.log1p(-P[c])), radius=r["radius"] if r else np.nan,
                        forced=[cells(n)[j] for j in r["forced"]] if r else None, mass=r["mass"] if r else np.nan))
    return out


def report(counts: Optional[np.ndarray] = None, p: Optional[np.ndarray] = None, with_strata=True) -> dict:
    """all four components on one table (full-joint counts, or a law p). Each component is computed independently."""
    if counts is not None:
        p = (np.asarray(counts, float) + 0.5) / (np.asarray(counts, float) + 0.5).sum()
    n = int(np.log2(len(p)))
    P_ipf = point_from_margins(p, n)                       # the max-ent member, from the margins alone
    F = fibre(P_ipf)
    P_b, t_b = birch(P_ipf)
    t_obs = float(chi(n) @ (p - P_ipf) / 2 ** n)            # where the observed law sits on the segment (P_ipf + t chi)
    out = dict(n=n, binomial_degree=2 ** (n - 1),
               F=dict(F, m_obs=float(chi(n) @ p)),
               T=dict(theta_at_ipf=theta(P_ipf), birch_minus_ipf=float(np.abs(P_b - P_ipf).max()), t_birch=float(t_b),
                      binomial_residual_at_ipf=float(np.prod(P_ipf[chi(n) > 0]) - np.prod(P_ipf[chi(n) < 0]))),
               I=dict(theta_hat(counts) if counts is not None else dict(theta=theta(p)), t_obs=t_obs,
                      tv_obs_to_birch=float(0.5 * np.abs(p - P_b).sum())))
    if with_strata:
        out["S"] = strata_report(P_b)
    return out
