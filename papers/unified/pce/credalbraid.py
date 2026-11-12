"""Credal sets, conditioning and braid-valued evidence. New module; it modifies nothing.

Finite history spaces. A credal set is the convex hull of finitely many laws (its vertices, rows of a matrix V). Conditioning acts
vertex by vertex (it is linear-fractional, so the conditioned hull is the hull of the conditioned vertices), and so does pushing forward.

  cond(V, C)              rows of V conditioned on the index set C (rows with P(C) = 0 are dropped)
  push(V, Pi, m)          push-forward of rows along Pi: histories -> {0..m-1}
  saturate(C, Pi)         the smallest Pi-saturated set containing C: Pi^{-1}(Pi(C))
  dist_to_hull(x, V)      total-variation distance (half L1) from the law x to the hull of the rows of V (a linear programme)
  hausdorff(VA, VB)       Hausdorff distance in total variation between two hulls (attained at vertices, by convexity)
  gap_bound(V, C, Pi)     max over vertices of P(Chat \\ C) / P(Chat), Chat = saturate(C, Pi)

Pseudo-links (the tie history of 'pseudoprob'):
  arcs_broker()           the two sectors whose joint occurrence is 'triple invariant 1' on the interleaved broker visits, as arcs
  overlap(a1, a2, psi)    |a1 cap (a2 + psi)| / 2 pi for arcs on the circle
  frechet(s1, s2)         the Frechet interval [max(0, s1 + s2 - 1), min(s1, s2)]
  envelope_circular(a1, a2)  min and max of overlap over psi (the envelope over all circular laws), and the minimising/maximising psi

Brokerage credal model (histories = interleavings of Proposition 'brokerage', m = 2):
  laws(H, lam, beta)      honest law p_lam (h) ~ lam^{pos1(h)} and coordinated law q_{lam,beta}(h) ~ lam^{pos1(h)} exp(beta mu(h)),
                          pos1 = sum of the positions of visitor 1's events
  projections(H)          statistics of a history: 'counts' (constant), 'untyped' (P_all), 'typed' (P_oo, P_cc, P_all), 'braid' (mu), 'full'
  posterior_interval(...) lower and upper P(coordinated | Pi = y) over the credal set (prior pinned at pi, independent parameters for the
                          two components; extremes of a linear-fractional function sit at extreme points)
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import linprog
from . import tiebraid as TB, pseudoprob as PP, brokerage as BR

TWO_PI = 2 * np.pi


def cond(V, C):
    V = np.atleast_2d(np.asarray(V, float)); C = sorted(C); out = []
    for v in V:
        z = v[C].sum()
        if z > 0: w = np.zeros_like(v); w[C] = v[C] / z; out.append(w)
    return np.array(out)


def push(V, Pi, m):
    V = np.atleast_2d(V); out = np.zeros((len(V), m))
    for h, y in enumerate(Pi): out[:, y] += V[:, h]
    return out


def saturate(C, Pi):
    ys = {Pi[h] for h in C}; return sorted(h for h in range(len(Pi)) if Pi[h] in ys)


def dist_to_hull(x, V):
    """min 0.5 * sum |x - V^T a|, a in simplex: variables a (k), t (n) with -t <= x - V^T a <= t"""
    V = np.atleast_2d(V); k, n = V.shape
    c = np.r_[np.zeros(k), 0.5 * np.ones(n)]
    A_ub = np.block([[-V.T, -np.eye(n)], [V.T, -np.eye(n)]]); b_ub = np.r_[-x, x]
    A_eq = np.r_[np.ones(k), np.zeros(n)][None, :]
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=[(0, None)] * (k + n), method="highs")
    return float(r.fun)


def hausdorff(VA, VB):
    return max(max(dist_to_hull(x, VB) for x in VA), max(dist_to_hull(y, VA) for y in VB))


def gap_bound(V, C, Pi):
    Ch = saturate(C, Pi); extra = sorted(set(Ch) - set(C)); best = 0.0
    for v in np.atleast_2d(V):
        z = v[Ch].sum()
        if z > 0: best = max(best, v[extra].sum() / z)
    return best


# pseudo-links
def arcs_broker(seed=5, draws=600):
    Zv = PP.broker_visit("interleaved"); W = TB.walls(Zv); a01, a12 = W[(0, 1)], W[(1, 2)]
    arc = lambda w: {1: (w[0], w[1]), 0: (w[1], w[0] + TWO_PI)}
    rng = np.random.default_rng(seed); enc = set()
    for o in PP.sample_circular_gaussian(np.eye(3), draws, rng):
        w, init, _, _ = TB.regularised(Zv, o); lk = TB.linking(w, init, 3)
        if all(v == 0 for v in lk) and TB.triple(w, init, 0, 1, 2) == 1: enc.add((TB._sector(o[0] - o[1], a01), TB._sector(o[1] - o[2], a12)))
    assert len(enc) == 1; (s1, s2), = enc
    return Zv, arc(a01)[s1], arc(a12)[s2]


def _mask(a, x):
    return ((x - a[0]) % TWO_PI) < (a[1] - a[0])


def overlap(a1, a2, psi, n=20000):
    """|{theta in a1 : theta - psi in a2}| / 2 pi, i.e. the probability that theta1 in a1 and theta2 = theta1 - psi in a2, theta1 uniform"""
    th = (np.arange(n) + 0.5) * TWO_PI / n
    return float(np.mean(_mask(a1, th) & _mask(a2, th - psi)))


def frechet(s1, s2): return max(0.0, s1 + s2 - 1.0), min(s1, s2)


def envelope_circular(a1, a2, n=721):
    ps = np.linspace(-np.pi, np.pi, n); v = np.array([overlap(a1, a2, p, 4000) for p in ps])
    return float(v.min()), float(v.max()), float(ps[v.argmin()]), float(ps[v.argmax()])


# brokerage credal model
def pos1(h): return sum(t for t, e in enumerate(h) if e[0] == "1")


def laws(H, lam, beta):
    base = np.array([lam ** pos1(h) for h in H], float); mu = np.array([BR.mu123(h) for h in H], float)
    p = base / base.sum(); q = base * np.exp(beta * mu); q = q / q.sum(); return p, q


def projections(H):
    S = [BR.stats(h) for h in H]
    raw = {"counts": [0 for _ in H], "untyped": [s["unsigned_prec"] for s in S],
           "typed": [(s["open_prec"], s["close_prec"], s["unsigned_prec"]) for s in S], "braid": [BR.mu123(h) for h in H], "full": list(range(len(H)))}
    out = {}
    for k, vals in raw.items():
        keys = sorted(set(vals)); idx = {v: i for i, v in enumerate(keys)}; out[k] = ([idx[v] for v in vals], keys)
    return out


def posterior_interval(H, Pi, m, lams_h, lams_c, betas, prior=0.5):
    """for each value y of Pi: (lower, upper) of P(coordinated | y); returns arrays and the component ranges"""
    A = np.array([push(laws(H, l, b)[1], Pi, m)[0] for l in lams_c for b in betas])          # coordinated, all parameters
    B = np.array([push(laws(H, l, 0.0)[0], Pi, m)[0] for l in lams_h])                       # honest
    amin, amax, bmin, bmax = A.min(0), A.max(0), B.min(0), B.max(0)
    lo = prior * amin / (prior * amin + (1 - prior) * bmax); hi = prior * amax / (prior * amax + (1 - prior) * bmin)
    return lo, hi
