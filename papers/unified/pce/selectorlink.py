"""Two selectors on one fibre: the toric (soft-min) family of the selection theorem and the maximum-entropy / I-projection selector.
New module; it modifies nothing (uses pce/datawalls.py for the lexicographic max-min point and the soft-min point).

2 x 3 couplings with margins f = (f1, f2), g = (g1, g2, g3). Coordinates u = (P11, P12). Cells in the order 11, 12, 13, 21, 22, 23 are the
affine functions P_k(u) = <v_k, u> - lam_k with
  v = (1,0), (0,1), (-1,-1), (-1,0), (0,-1), (1,1);  lam = 0, 0, -f1, -g1, -g2, f1 - g3.
  cells(f, g)                 (V, lam)
  law(u, f, g)                the six cell probabilities at u
  toric(f, g, L, a=None, idx=None)   the minimiser of sum_k a_k exp(-L P_k(u)) over the cells idx (all cells by default)
  lex(f, g, idx=None, shift=None)    lexicographic max-min point of the cells P_k - shift_k
  maxent(f, g, tau=None)      the I-projection of exp(tau) (uniform when tau is None) onto the fibre (iterative scaling)
  leastsq(f, g)               the minimiser of sum_k P_k^2 on the affine hull
  span_D()                    constants + row and column indicators on the six cells (the data span H_D)
"""
from __future__ import annotations
import numpy as np
from . import datawalls as DW

V6 = np.array([[1, 0], [0, 1], [-1, -1], [-1, 0], [0, -1], [1, 1]], float)


def cells(f, g):
    return V6, np.array([0.0, 0.0, -f[0], -g[0], -g[1], f[0] - g[2]])


def law(u, f, g):
    V, lam = cells(f, g); return V @ np.asarray(u, float) - lam


def toric(f, g, L, a=None, idx=None, u0=None):
    V, lam = cells(f, g); idx = list(range(6)) if idx is None else list(idx); a = np.ones(len(idx)) if a is None else np.asarray(a, float)
    return softmin(V[idx], lam[idx] + np.log(a) / L, L, u0)


def softmin(V, lam, L, u0=None, iters=500):
    """minimiser of (1/L) log sum_k exp(-L (<v_k, u> - lam_k)): Newton with least-squares steps and backtracking; started at the
    lexicographic max-min point when L is large (where the soft-min concentrates), else at the least-squares point"""
    if u0 is None: u0 = DW.lex_maxmin(V, lam)[0] if L > 20 else np.linalg.lstsq(V, lam, rcond=None)[0]
    u = np.array(u0, float)
    def F(u):
        z = -L * (V @ u - lam); mx = z.max(); return (mx + np.log(np.exp(z - mx).sum())) / L
    for _ in range(iters):
        z = -L * (V @ u - lam); w = np.exp(z - z.max()); w /= w.sum(); gr = -(w @ V)
        if np.abs(gr).max() < 1e-15: break
        mu = w @ V; H = L * ((V - mu).T @ ((V - mu) * w[:, None])); d = -np.linalg.lstsq(H, gr, rcond=None)[0]; s = 1.0; f0 = F(u)
        if np.abs(gr).max() < 1e-7:                     # near the optimum F differs below rounding: take the full step if it lowers the gradient
            z2 = -L * (V @ (u + d) - lam); w2 = np.exp(z2 - z2.max()); w2 /= w2.sum()
            if np.abs(w2 @ V).max() < np.abs(gr).max(): u = u + d; continue
            break
        while F(u + s * d) > f0 + 1e-4 * s * (gr @ d) and s > 1e-14: s /= 2
        if s <= 1e-14: break
        u = u + s * d
    return u


def lex(f, g, idx=None, shift=None):
    V, lam = cells(f, g); idx = list(range(6)) if idx is None else list(idx); s = np.zeros(len(idx)) if shift is None else np.asarray(shift, float)
    return DW.lex_maxmin(V[idx], lam[idx] + s)[0]


def maxent(f, g, tau=None, sweeps=5000):
    P = np.ones((2, 3)) if tau is None else np.exp(np.asarray(tau, float).reshape(2, 3)); P = P / P.sum()
    for _ in range(sweeps):
        P = P * (np.asarray(f)[:, None] / P.sum(1, keepdims=True)); P = P * (np.asarray(g)[None, :] / P.sum(0, keepdims=True))
    return np.array([P[0, 0], P[0, 1]])


def leastsq(f, g):
    V, lam = cells(f, g); return np.linalg.solve(V.T @ V, V.T @ lam)


def span_D():
    B = [np.ones(6)]
    for i in range(2): B.append(np.array([1.0 if k // 3 == i else 0.0 for k in range(6)]))
    for j in range(3): B.append(np.array([1.0 if k % 3 == j else 0.0 for k in range(6)]))
    return np.array(B).T
