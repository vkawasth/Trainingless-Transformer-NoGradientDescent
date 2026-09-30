"""Tree space for 3-leaf coalition trees (Billera-Holmes-Vogtmann space T_3 = the 'spider' with three legs).

A filled cell with three sources gives an ultrametric tree (single linkage of the values, or of the variogram):
its topology is the coalition {i,j} | k and its internal edge length is h2 - h1 (second merge height minus first).
Ties give the star tree (the origin). T_3 is three half-lines glued at the origin; it is CAT(0) with metric
    d((leg a, s), (leg b, t)) = |s - t| if a = b, else s + t.
Frechet mean (Hotz et al. 2013, sticky means on open books / spiders): with leg moments m_a = E[s 1{leg a}],
the mean lies on leg a at m_a - sum_{b != a} m_b if that is positive for some a (at most one can be), otherwise it
STICKS to the origin (no population coalition). Energy distance between two samples (Szekely-Rizzo) with d gives a
permutation test for a change of the coalition distribution (topology and edge length together).
"""
from __future__ import annotations
import numpy as np


def spider_point(x, labels=None):
    """x: 3 values -> (leg index 0/1/2 for coalitions (0,1),(0,2),(1,2), internal length); star -> (-1, 0)."""
    x = np.asarray(x, float)
    g = {(0, 1): abs(x[0] - x[1]), (0, 2): abs(x[0] - x[2]), (1, 2): abs(x[1] - x[2])}
    order = sorted(g.items(), key=lambda kv: kv[1])
    (pair, h1), (_, h2) = order[0], order[1]
    if h2 - h1 <= 1e-12:
        return (-1, 0.0)
    return ([(0, 1), (0, 2), (1, 2)].index(pair), float(h2 - h1))


def dist(P, Q):
    (a, s), (b, t) = P, Q
    if a == -1 or b == -1 or a != b:
        return s + t
    return abs(s - t)


def frechet_mean(pts):
    pts = list(pts); n = len(pts)
    m = np.zeros(3)
    for a, s in pts:
        if a >= 0:
            m[a] += s / n
    for a in range(3):
        v = m[a] - (m.sum() - m[a])
        if v > 0:
            return (a, float(v)), m
    return (-1, 0.0), m


def _dmat(pts):
    legs = np.array([p[0] for p in pts]); s = np.array([p[1] for p in pts])
    same = (legs[:, None] == legs[None, :]) & (legs[:, None] >= 0)
    return np.where(same, np.abs(s[:, None] - s[None, :]), s[:, None] + s[None, :])


def energy_test(X, Y, B=999, seed=0):
    rng = np.random.default_rng(seed)
    Z = list(X) + list(Y); D = _dmat(Z); n = len(X); N = len(Z)
    def stat(idx):
        a, b = idx[:n], idx[n:]
        return 2 * D[np.ix_(a, b)].mean() - D[np.ix_(a, a)].mean() - D[np.ix_(b, b)].mean()
    idx = np.arange(N); e0 = stat(idx)
    null = np.array([stat(rng.permutation(N)) for _ in range(B)])
    return dict(energy=float(e0), p=float((1 + (null >= e0).sum()) / (B + 1)), n_x=n, n_y=N - n)
