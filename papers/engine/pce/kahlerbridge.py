"""Conditioned states as toric Kahler data: the Birch bridge (companion note kahler/conditioned_states_kahler.tex). New module; it does not
import or modify pce/torickahler.py.

Tier A (classical). For a support S with positive coefficients c, phi(u) = log sum_i c_i e^(<S_i, u>) is the Kahler potential in log
coordinates of the torus-invariant Kahler form pulled back by the monomial map: its gradient is the moment map (the mean exponent of the
monomial law), its Hessian the Fisher information, and its Legendre dual psi(p) = sum q log(q / c) the symplectic potential.

Tier B. For a compact Delzant polytope Q = {p : <v_k, p> >= lam_k}, the potential W_T(y) = sum_k T^(-lam_k) y^(v_k) has exactly one critical
point y_T on the positive torus (Birch at target 0). Its valuation u* = lim log y_T / log T exists, and the leading terms at u* balance:
0 lies in the relative interior of conv{v_k : l_k(u*) minimal}. The converse fails: balance can hold on a segment of candidates.

  phi, moment, fisher, legendre          Tier A
  positive_crit(v, lam, T)               the positive critical point of W_T, in log coordinates b = log y (strictly convex minimisation)
  valuation(v, lam, Ts)                  log y_T / log T at a sequence of T
  ties(v, lam, u)                        K(u), the facets of least l_k(u) = <v_k, u> - lam_k
  balanced(v, K)                         is 0 in the relative interior of conv{v_k : k in K}? (linear programme with weights >= 1)
  signed_balanced(v, K)                  is 0 = sum a_k v_k with every a_k non-zero (any signs)? (linear dependence with full support)
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import minimize, linprog


def _lse(z):
    m = z.max(); return m + np.log(np.exp(z - m).sum())


def phi(S, c, u): return _lse(np.log(np.array(c, float)) + np.array(S, float) @ np.array(u, float))


def _law(S, c, u):
    z = np.log(np.array(c, float)) + np.array(S, float) @ np.array(u, float); w = np.exp(z - z.max()); return w / w.sum()


def moment(S, c, u): return _law(S, c, u) @ np.array(S, float)


def fisher(S, c, u):
    q = _law(S, c, u); A = np.array(S, float); m = q @ A; return (A - m).T @ ((A - m) * q[:, None])


def legendre(S, c, p):
    """psi(p) = sup_u (<p, u> - phi(u)), with the maximiser u(p) and its law q (Birch)"""
    A = np.array(S, float); lc = np.log(np.array(c, float)); p = np.array(p, float)
    f = lambda u: _lse(lc + A @ u) - p @ u
    g = lambda u: _law(S, c, u) @ A - p
    r = minimize(f, np.zeros(A.shape[1]), jac=g, method="BFGS", options=dict(gtol=1e-13, maxiter=10000))
    q = _law(S, c, r.x); return -r.fun, r.x, q


def positive_crit(v, lam, T):
    """minimise log W_T(e^b) = log sum_k T^(-lam_k) e^(<v_k, b>): strictly convex and coercive when the v_k positively span"""
    V = np.array(v, float); lw = -np.array(lam, float) * np.log(T)
    f = lambda b: _lse(lw + V @ b)
    def g(b):
        z = lw + V @ b; w = np.exp(z - z.max()); return (w / w.sum()) @ V
    r = minimize(f, np.zeros(V.shape[1]), jac=g, method="BFGS", options=dict(gtol=1e-14, maxiter=20000))
    z = lw + V @ r.x; w = np.exp(z - z.max()); return r.x, w / w.sum()


def valuation(v, lam, Ts):
    return [positive_crit(v, lam, T)[0] / np.log(T) for T in Ts]


def ties(v, lam, u, tol=1e-9):
    l = np.array(v, float) @ np.array(u, float) - np.array(lam, float); return tuple(int(k) for k in np.where(l <= l.min() + tol)[0])


def balanced(v, K):
    V = np.array([v[k] for k in K], float)
    r = linprog(np.zeros(len(K)), A_eq=V.T, b_eq=np.zeros(V.shape[1]), bounds=[(1, None)] * len(K), method="highs")
    return r.status == 0


def signed_balanced(v, K):
    """0 = sum a_k v_k with every a_k != 0: try all sign patterns with |a_k| >= 1"""
    V = np.array([v[k] for k in K], float)
    for sg in itertools.product((1, -1), repeat=len(K)):
        bnds = [(1, None) if s > 0 else (None, -1) for s in sg]
        r = linprog(np.zeros(len(K)), A_eq=V.T, b_eq=np.zeros(V.shape[1]), bounds=bnds, method="highs")
        if r.status == 0: return True
    return False


CP2 = ([(1, 0), (0, 1), (-1, -1)], [0, 0, -1])
def F1(alpha):
    """the Hirzebruch surface F_1 (CP^2 blown up at a point), Q = {p1 >= 0, p2 >= 0, p1 + p2 <= 1, p2 <= 1 - alpha}, 0 < alpha < 1"""
    return ([(1, 0), (0, 1), (-1, -1), (0, -1)], [0, 0, -1, -(1 - alpha)])


# ------------------------------------------------------------------------------------------------ the positive thimble and its tropical limit
def maxmin(v, lam):
    """h* = max_u min_k l_k(u) over R^n, and one maximiser (linear programme: maximise t with <v_k, u> - lam_k >= t)"""
    V = np.array(v, float); n = V.shape[1]; m = len(v)
    r = linprog(np.r_[np.zeros(n), -1.0], A_ub=np.c_[-V, np.ones(m)], b_ub=-np.array(lam, float), bounds=[(None, None)] * (n + 1), method="highs")
    return -r.fun, r.x[:n]


def tropical_density(v, lam, u, T, h):
    """the density, in u = log y / log T, of the thimble measure e^(-W_T(y) / T^h) dy / y on the positive orthant (up to a constant)"""
    l = np.array(v, float) @ np.array(u, float) - np.array(lam, float); return float(np.exp(-np.sum(np.exp((l - h) * np.log(T)))))


def laplace_moments(v, lam, T, hbar, n=241, width=8.0):
    """mean and covariance of b = log y under e^(-W_T(e^b) / hbar) db (two dimensions, quadrature on a box around the critical point)"""
    b0, _ = positive_crit(v, lam, T); V = np.array(v, float); cw = -np.array(lam, float) * np.log(T)
    H = (np.exp(cw + V @ b0)[:, None, None] * np.einsum("ki,kj->kij", V, V)).sum(0)      # Hess of W(e^b) at b0
    s = np.sqrt(np.diag(np.linalg.inv(H / hbar))); g1 = np.linspace(-width, width, n) * s[0] + b0[0]; g2 = np.linspace(-width, width, n) * s[1] + b0[1]
    B1, B2 = np.meshgrid(g1, g2, indexing="ij"); E = np.exp(cw[None, None, :] + B1[..., None] * V[:, 0] + B2[..., None] * V[:, 1]).sum(-1)
    w = np.exp(-(E - E.min()) / hbar); w /= w.sum(); m1 = (w * B1).sum(); m2 = (w * B2).sum()
    C = np.array([[(w * (B1 - m1) ** 2).sum(), (w * (B1 - m1) * (B2 - m2)).sum()], [0, (w * (B2 - m2) ** 2).sum()]]); C[1, 0] = C[0, 1]
    return b0, np.array([m1, m2]), C, np.linalg.inv(H / hbar)


def tropical_grid(v, lam, logT, h, box=((-0.3, 1.3), (-0.3, 1.3)), n=1201):
    """the thimble law e^(-W_T / T^h) dy / y pushed forward by u = log y / log T, normalised on a grid (logT = log T < 0; vectorised)"""
    (a0, a1), (b0, b1) = box; U1, U2 = np.meshgrid(np.linspace(a0, a1, n), np.linspace(b0, b1, n), indexing="ij")
    V = np.array(v, float); L = U1[..., None] * V[:, 0] + U2[..., None] * V[:, 1] - np.array(lam, float)
    with np.errstate(over="ignore"): E = np.exp((L - h) * logT).sum(-1)
    w = np.exp(-(E - E.min())); return U1, U2, w / w.sum()


def argmax_set(v, lam, u, tol=0):
    """is u in argmax min_k l_k, i.e. 0 in conv{v_k : k in K(u)} (the superdifferential of the concave function min_k l_k)?"""
    V = np.array([v[k] for k in ties(v, lam, u)], float)
    r = linprog(np.zeros(len(V)), A_eq=np.r_[V.T, np.ones((1, len(V)))], b_eq=np.r_[np.zeros(V.shape[1]), 1.0], bounds=[(0, None)] * len(V), method="highs")
    return r.status == 0


def polygon_moments(v, lam, h, interior):
    """mean and covariance of the uniform law on Q_h = {u : <v_k, u> >= lam_k + h} (two dimensions; interior: any interior point)"""
    from scipy.spatial import HalfspaceIntersection, ConvexHull
    A = np.array(v, float); P = HalfspaceIntersection(np.c_[-A, np.array(lam, float) + h], np.array(interior, float)).intersections
    P = P[ConvexHull(P).vertices]; c0 = P.mean(0); M = 0.0; m = np.zeros(2); S = np.zeros((2, 2))
    for i in range(len(P)):
        a, b = P[i] - c0, P[(i + 1) % len(P)] - c0; ar = abs(a[0] * b[1] - a[1] * b[0]) / 2; X = np.array([c0, P[i], P[(i + 1) % len(P)]])
        M += ar; m += ar * X.mean(0); S += ar / 12 * (X.T @ X + np.outer(X.sum(0), X.sum(0)))
    m /= M; return m, S / M - np.outer(m, m)
