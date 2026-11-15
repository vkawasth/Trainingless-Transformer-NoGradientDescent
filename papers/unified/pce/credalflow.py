"""Credal sets under evidence flows (section sec:credalflow). New module; it modifies nothing.
States 0..m-1; a law p in the open simplex; soft evidence with log-likelihood a acts by the tilt p -> p e^(t a) / <p, e^(t a)>, a
translation of log p modulo constants. A credal polytope K is given by its vertices; the tilt is linear-fractional with positive
denominator, so it maps K onto the polytope spanned by the tilted vertices, and linear payoffs take their extremes at vertices.
  tilt(p, a, t)                 the tilted law
  hilbert(p, q)                 Hilbert projective distance osc_i log(p_i / q_i)
  levels(a, tol)                the distinct values of a, decreasing, with their state sets
  level_coeffs(p, c, a)         C_s(p) = sum over level s of c_i p_i (the coefficients of the exponential sum)
  sign_changes(xs)              sign changes in a sequence, zeros ignored
  lex_sign(p, c, a)             the sign of the first non-zero C_s: the eventual sign of <c, p_t>
  bracket(V, c, a, t)           [min, max] over K_t of <c, p>
  crossing_bound(V, c, a)       log+(Nbar / kappa) / Delta over the states with c_i != 0: after it, <c, p_t> > 0 on all of K_t
  last_nonpositive(V, c, a, T)  sup{t in [0, T] : min_K_t <c, p> <= 0} (grid and bisection)
  cell_conditionals(p, R)       p(. | R = o) for each observation cell o
"""
from __future__ import annotations
import math
import numpy as np
def tilt(p, a, t):
    l = np.log(np.asarray(p, float)) + t * np.asarray(a, float); l -= l.max(); q = np.exp(l); return q / q.sum()
def hilbert(p, q):
    r = np.log(np.asarray(p, float)) - np.log(np.asarray(q, float)); return float(r.max() - r.min())
def levels(a, tol=1e-12):
    a = np.asarray(a, float); vals = []
    for x in sorted(set(np.round(a, 12)), reverse=True):
        vals.append((float(x), [i for i in range(len(a)) if abs(a[i] - x) <= tol]))
    return vals
def level_coeffs(p, c, a):
    return [float(sum(c[i] * p[i] for i in S)) for _, S in levels(a)]
def sign_changes(xs, tol=1e-15):
    s = [np.sign(x) for x in xs if abs(x) > tol]; return sum(1 for i in range(1, len(s)) if s[i] != s[i - 1])
def lex_sign(p, c, a, tol=1e-13):
    for C in level_coeffs(p, c, a):
        if abs(C) > tol: return int(np.sign(C))
    return 0
def bracket(V, c, a, t):
    g = [float(np.dot(c, tilt(v, a, t))) for v in V]; return min(g), max(g)
def crossing_bound(V, c, a):
    """levels are taken over the states with c_i != 0 only: states with zero payoff do not enter the exponential sum."""
    c = np.asarray(c, float); a = np.asarray(a, float); S = [i for i in range(len(c)) if c[i] != 0]
    V = [np.asarray(v, float)[S] for v in V]; c = c[S]; a = a[S]
    L = levels(a)
    if len(L) < 2: return None
    Delta = L[0][0] - L[1][0]; kappa = min(level_coeffs(v, c, a)[0] for v in V)
    Nbar = max(sum(max(-C, 0.0) for C in level_coeffs(v, c, a)[1:]) for v in V)
    if kappa <= 0: return None
    return max(0.0, math.log(Nbar / kappa)) / Delta if Nbar > 0 else 0.0
def last_nonpositive(V, c, a, T, n=4000):
    ts = np.linspace(0, T, n + 1); low = np.array([bracket(V, c, a, t)[0] for t in ts]); bad = np.where(low <= 0)[0]
    if len(bad) == 0: return None
    k = bad[-1]
    if k == n: return float("inf")
    lo, hi = ts[k], ts[k + 1]
    for _ in range(60):
        mid = (lo + hi) / 2
        if bracket(V, c, a, mid)[0] <= 0: lo = mid
        else: hi = mid
    return (lo + hi) / 2
def cell_conditionals(p, R):
    p = np.asarray(p, float); out = {}
    for o in sorted(set(R)):
        idx = [i for i in range(len(p)) if R[i] == o]; out[o] = p[idx] / p[idx].sum()
    return out
