"""The adversary's segment and the regulator's segment on one geometry (lab paper, Theorem H), and the obstruction to bent stealth
(Theorem G(4)). Kept separate from pce/injection.py, which is frozen with Study 12.

  pair_fibre_basis(m)            an orthonormal basis of the directions that leave every single and pair table unchanged
  segment_points(P, n)           laws of the regulator's hiding segment (two-type explanations of P's pair tables)
  segment_tangent(P)             d law / d r along the hiding segment at P (unit-free direction in law space)
  model_residuals(P)             the equations of the two-type model on m = 4 beyond the pair tables: the tetrads, the triple
                                 consistency r_S - r_S' (S, S' triples) and the quadruple m_0123 - v_0123 (r^2 + 1)
  m3_line(P)                     m = 3: the regulator's segment and the positivity segment in triple-moment coordinates
  obstruction(P, A)              second-order obstruction of a first-order tetrad-stealthy direction A (component of the tetrads'
                                 second derivative outside the image of their Jacobian); zero means a bent injection extends it
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.linalg import null_space
from .lab import law, moments, fit_v, segment, explanation, r_of_t, t_of_r
from . import injection as IJ
from .segment_decision import v_closed


def marginal_matrix(m, order=2):
    rows = [[1.0] * 2 ** m]
    for k in range(1, order + 1):
        for S in itertools.combinations(range(m), k):
            rows.append([float(all(x[i] == 1 for i in S)) for x in IJ.outcomes(m)])
    return np.array(rows)


def pair_fibre_basis(m): return null_space(marginal_matrix(m, 2))


def _mu_v(P):
    mu, M, _ = moments(P); return mu, (v_closed(M) if P.ndim == 3 else fit_v(mu, M))


def segment_points(P, n=7):
    mu, v = _mu_v(P); lo, hi = segment(mu, v); ts = np.exp(np.linspace(np.log(lo) + 0.02, np.log(hi) - 0.02, n))
    return [law(*explanation(mu, v, t)) for t in ts], r_of_t(ts)


def segment_tangent(P, h=1e-6):
    mu, M, mS = moments(P); v = fit_v(mu, M); m = P.ndim; S = tuple(range(3)); r0 = mS(S) / np.prod(v[list(S)])
    f = lambda r: IJ.as_vec(law(*explanation(mu, v, t_of_r(r))))
    return (f(r0 + h) - f(r0 - h)) / (2 * h)


def model_residuals(P):
    mu, M, mS = moments(P); v = fit_v(mu, M); T = IJ.tetrads(P)
    trip = list(itertools.combinations(range(4), 3)); rs = [mS(S) / np.prod(v[list(S)]) for S in trip]
    r = rs[0]; quad = mS((0, 1, 2, 3)) - np.prod(v) * (r * r + 1)
    return np.concatenate([T, np.array(rs[1:]) - rs[0], [quad]])


def m3_line(P):
    """m = 3: returns (model interval, positivity interval) for the triple moment m_T on the line of laws with P's pair tables"""
    mu, M, mS = moments(P); v = v_closed(M); vT = np.prod(v); lo, hi = segment(mu, v); R = sorted((r_of_t(lo) * vT, r_of_t(hi) * vT))
    sl, sh = IJ.stealth_range(P, (0, 1, 2)); m0 = mS((0, 1, 2))
    return (float(R[0]), float(R[1])), (float(m0 + sl), float(m0 + sh)), float(m0)


def _tet_vec(p): return IJ.tetrads(IJ.as_law(p, 4))


def jet(P, h=1e-4):
    """Jacobian J (2 x 15) and Hessian H (2 x 15 x 15) of the tetrads in an orthonormal basis B of the simplex's tangent space"""
    p = IJ.as_vec(P); B = null_space(np.ones((1, 16))); f = lambda z: _tet_vec(p + B @ z); n = 15
    J = np.array([(f(h * e) - f(-h * e)) / (2 * h) for e in np.eye(n)]).T; H = np.zeros((2, n, n))
    for i in range(n):
        for j in range(i, n):
            ei = h * np.eye(n)[i]; ej = h * np.eye(n)[j]
            H[:, i, j] = H[:, j, i] = (f(ei + ej) - f(ei - ej) - f(-ei + ej) + f(-ei - ej)) / (4 * h * h)
    return p, B, J, H


def obstruction(J, H, A, tol=1e-7):
    U, sv, _ = np.linalg.svd(J); rk = int(np.sum(sv > tol * max(sv[0], 1e-300))); q = 0.5 * np.einsum("kij,i,j->k", H, A, A)
    return float(np.linalg.norm(U[:, rk:].T @ q)), rk
