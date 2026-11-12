"""The conditional credal field and its braid. New module; it modifies nothing.

A target Y (binary) and k pathways or sources with binary evidence X_1..X_k. The credal set M is the set of joint laws on (X_1..X_k, Y)
(2^{k+1} outcomes) satisfying the data: P(Y=1) = q, P(X_i=1) = p_i(t), and each source's study value a_i(t) = P(X_i=1, Y=1) known up to a
tolerance e_i(t) (a 'star' cover: every context contains Y). Optional relational data add contexts with two pathways.

The field at time t is F(t) = {(P(Y=1 | X_i=1))_i : P in M(t)}; under one law it is a point, and the points of one law over time are the
strands. The order of strands i and j at time t is decided when F(t) lies on one side of the diagonal v_i = v_j.

  outcomes(k)                         all (x_1..x_k, y) tuples
  constraints(k, q, p, a, e, rel)     equality and inequality rows of the LP for one time
  value_range(k, ..., c)              min and max of a linear functional c . (P(X_i=1, Y=1) / p_i)_i over M (two LPs), or None if M is empty
  intervals(...)                      the field's coordinate intervals [L_i, U_i]
  order_decided(..., i, j)            +1 if v_i > v_j for every law, -1 if v_i < v_j for every law, 0 if undecided, None if M is empty
  projection(..., i, j, m)            the projection of F onto (v_i, v_j), as m support points (for drawing)
  MODEL                               the toy pathway model used in the figure and checks
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import linprog


def outcomes(k): return list(itertools.product((0, 1), repeat=k + 1))


def constraints(k, q, p, a, e, rel=()):
    """rel: list of (i, j, x_i, x_j, y, lo, hi): lo <= P(X_i=x_i, X_j=x_j, Y=y) <= hi"""
    O = outcomes(k); n = len(O); Aeq = [np.ones(n)]; beq = [1.0]; Aub = []; bub = []
    Aeq.append(np.array([1.0 if o[-1] == 1 else 0.0 for o in O])); beq.append(q)
    for i in range(k):
        Aeq.append(np.array([1.0 if o[i] == 1 else 0.0 for o in O])); beq.append(p[i])
        row = np.array([1.0 if (o[i] == 1 and o[-1] == 1) else 0.0 for o in O])
        Aub.append(row); bub.append(a[i] + e[i]); Aub.append(-row); bub.append(-(a[i] - e[i]))
    for (i, j, xi, xj, y, lo, hi) in rel:
        row = np.array([1.0 if (o[i] == xi and o[j] == xj and o[-1] == y) else 0.0 for o in O])
        Aub.append(row); bub.append(hi); Aub.append(-row); bub.append(-lo)
    return O, np.array(Aeq), np.array(beq), np.array(Aub), np.array(bub)


def _obj(O, k, p, c):
    return np.array([sum(c[i] / p[i] for i in range(k) if o[i] == 1 and o[-1] == 1) for o in O])


def value_range(k, q, p, a, e, c, rel=()):
    O, Aeq, beq, Aub, bub = constraints(k, q, p, a, e, rel); f = _obj(O, k, p, c); out = []
    for sgn in (1, -1):
        r = linprog(sgn * f, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * len(O), method="highs")
        if r.status != 0: return None
        out.append(sgn * r.fun)
    return tuple(out)


def intervals(k, q, p, a, e, rel=()):
    out = []
    for i in range(k):
        c = np.zeros(k); c[i] = 1.0; out.append(value_range(k, q, p, a, e, c, rel))
    return out


def order_decided(k, q, p, a, e, i, j, rel=(), tol=1e-9):
    c = np.zeros(k); c[i] = 1.0; c[j] = -1.0; r = value_range(k, q, p, a, e, c, rel)
    if r is None: return None
    lo, hi = r
    return 1 if lo > tol else (-1 if hi < -tol else 0)


def projection(k, q, p, a, e, i, j, m=72, rel=()):
    pts = []
    for th in np.linspace(0, 2 * np.pi, m, endpoint=False):
        c = np.zeros(k); c[i] = np.cos(th); c[j] = np.sin(th)
        O, Aeq, beq, Aub, bub = constraints(k, q, p, a, e, rel); f = _obj(O, k, p, c)
        r = linprog(-f, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * len(O), method="highs")
        P = r.x; v = [sum(P[t] for t, o in enumerate(O) if o[s] == 1 and o[-1] == 1) / p[s] for s in (i, j)]; pts.append(v)
    return np.array(pts)


def _sig(x): return 1.0 / (1.0 + np.exp(-x))


# toy model: four pathways over a 120-minute window; centres c_i(t) = P(Y=1 | X_i=1), half-widths w_i(t); p_i = 1/2, q = 1/2
NAMES = ["receptor signal", "light suppression", "sleep pressure", "arousal"]
T = np.linspace(0, 120, 49)
def centres(t):
    return np.array([0.33 + 0.36 * _sig((t - 50) / 11), 0.64 - 0.22 * t / 120, 0.44 + 0.16 * t / 120, 0.62 - 0.30 * _sig((t - 80) / 14)])
def halfwidths(t):
    return np.array([0.035, 0.07, 0.03, 0.05]) * np.ones_like(np.atleast_1d(t))[0]
MODEL = dict(k=4, q=0.5, p=np.full(4, 0.5))


def data_at(t, shift=None):
    c = centres(t); w = halfwidths(t); p = MODEL["p"]; a = c * p; e = w * p
    if shift is not None: a = a + np.asarray(shift) * p
    return a, e
