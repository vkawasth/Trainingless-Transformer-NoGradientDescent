"""One picture: chambers, slices and gauges on the fibre (lab paper, Theorem L). New module; nothing frozen.

A decision between two acts a, b is the linear functional l(P) = sum_x (u_a - u_b)(x) P(x) with wall {l = c} (c = 0 for a = b; for
the requirement, l = U_a and c = u0). The fibre is the line P + s chi (m = 3, pair cover), its model part M the hiding segment.

  wall_functional(ua, ub)           l as a vector on the 8 cells
  chamber_test(P, ua, ub)           does M lie in one chamber? (sign of U_a - U_b at the two walls of the segment)
  set_rule_is_oracle(P, ua, ub)     does the pre-probe set rule choose the oracle act at every truth of M? (grid over M)
  savage_values(P, ua, ub)          R(M) = min_a max_M regret; exact point-slice probe: R_post = 0; plateau probe: R_post = R(M)
  wald_values(P, ua, ub)            W(M) = max_a min_M U_a; exact point-slice probe: W_post = min_M max_a U_a
  fwd_budget(P, l, c, Q)            margin / capacity: delta / (l(Q) - l(P)), for a given injected law Q
  fwd_free(P, l, c)                 over all laws (vertex attack): delta / (max_x l(x) - l(P))
  fwd_line(P, l, c)                 along the pair line (hidden channel): delta / (s_cap |l(chi)|)
  obs_free(O, l, c)                 the observer's block, no model constraint: delta / (c - min_x l(x)); and the LP that checks it
  obs_line(O, l, c)                 the observer's block on the line: |s'| / (cap(O) + |s'|), s' = delta / l(chi)
"""
from __future__ import annotations
import itertools
import numpy as np
from numpy.polynomial import polynomial as Pn
from scipy.optimize import linprog
from .lab import moments, segment, explanation, law, r_of_t
from . import segment_decision as SD
from . import injection as IJ

XS = list(itertools.product((0, 1), repeat=3)); CHI = IJ.chi(3, (0, 1, 2))


def wall_functional(ua, ub): return np.asarray(ua, float) - np.asarray(ub, float)


def _polys(P, ua, ub):
    mu, M, _ = moments(P); v = SD.v_closed(M); lo, hi = segment(mu, v); R = (r_of_t(lo), r_of_t(hi))
    fa = dict(zip(XS, ua)); fb = dict(zip(XS, ub))
    pa = SD.U_poly(mu, v, lambda x: fa[tuple(x)], range(3)); pb = SD.U_poly(mu, v, lambda x: fb[tuple(x)], range(3))
    return pa, pb, R, mu, v


def chamber_test(P, ua, ub):
    pa, pb, R, _, _ = _polys(P, ua, ub); d = [Pn.polyval(e, pa) - Pn.polyval(e, pb) for e in R]
    return bool(d[0] * d[1] > 0), d


def set_rule_is_oracle(P, ua, ub, grid=401):
    pa, pb, R, _, _ = _polys(P, ua, ub); pre = SD.set_rule([pa, pb], R, -1e9)
    return all(SD.oracle([pa, pb], r, -1e9) == pre for r in np.linspace(R[0], R[1], grid))


def _ends(P, ua, ub):
    pa, pb, R, _, _ = _polys(P, ua, ub); A = [Pn.polyval(e, pa) for e in R]; B = [Pn.polyval(e, pb) for e in R]; return A, B, pa, pb, R


def savage_values(P, ua, ub):
    """affine utilities on M: regrets are largest at the endpoints"""
    A, B, *_ = _ends(P, ua, ub)
    ra = max(max(b - a, 0) for a, b in zip(A, B)); rb = max(max(a - b, 0) for a, b in zip(A, B)); return min(ra, rb)


def wald_values(P, ua, ub):
    """W(M) = max_a min_M U_a (before the probe) and W_post = min_M max_a U_a (exact point slices); with the two slopes"""
    A, B, pa, pb, R = _ends(P, ua, ub); W = max(min(A), min(B))
    sa = pa[1] if len(pa) > 1 else 0.0; sb = pb[1] if len(pb) > 1 else 0.0; cands = list(R)
    if abs(sa - sb) > 1e-15:
        rc = (pb[0] - pa[0]) / (sa - sb)
        if R[0] < rc < R[1]: cands.append(rc)
    return W, min(max(Pn.polyval(r, pa), Pn.polyval(r, pb)) for r in cands), sa, sb


def model_part_s(P):
    """the hiding segment's endpoints in the line coordinate s (laws P + s chi)"""
    mu, M, _ = moments(P); v = SD.v_closed(M); lo, hi = segment(mu, v); p = IJ.as_vec(P)
    ss = [float((IJ.as_vec(law(*explanation(mu, v, t))) - p) @ CHI / 8) for t in (lo * (1 + 1e-12), hi * (1 - 1e-12))]
    return min(ss), max(ss)


# --------------------------------------------------------------------------------------------- margin over capacity
def fwd_budget(P, l, c, Q):
    p = IJ.as_vec(P); q = IJ.as_vec(Q); d = c - l @ p; return d / (l @ q - l @ p) if l @ q > l @ p else np.inf


def fwd_free(P, l, c):
    p = IJ.as_vec(P); return (c - l @ p) / (l.max() - l @ p)


def s_caps(P):
    p = IJ.as_vec(P); return float(min(p[CHI < 0])), float(min(p[CHI > 0]))   # s_+ (P + s chi >= 0 for s <= s_+), |s_-|


def fwd_line(P, l, c):
    p = IJ.as_vec(P); d = c - l @ p; lc = l @ CHI
    if abs(lc) < 1e-12 * max(1.0, np.abs(l).max()): return np.inf
    sp, sm = s_caps(P); cap = sp if lc > 0 else sm; return d / (cap * abs(lc))


def obs_free(O, l, c):
    """the observer's block, no model constraint, by water-filling: the adversary's records are taken from the cells with the
    lowest l first. If the lowest cell holds enough mass the radius is delta / (c - min_x l(x)) (margin over capacity)."""
    o = IJ.as_vec(O); d = c - l @ o
    if d <= 0: return 0.0
    removed = 0.0; gain = 0.0                       # gain = sum of (c - l(x)) * mass removed; need gain >= d
    for k in np.argsort(l):
        if c - l[k] <= 0: break
        take = min(o[k], (d - gain) / (c - l[k])); removed += take; gain += take * (c - l[k])
        if gain >= d - 1e-15: return removed
    return np.inf


def obs_free_closed(O, l, c):
    o = IJ.as_vec(O); return (c - l @ o) / (c - l.min())


def obs_free_lp(O, l, c):
    """max lambda = sum y subject to 0 <= y <= o (y = lambda P'), l y >= c sum y; returns 1 - lambda"""
    o = IJ.as_vec(O); n = len(o)
    r = linprog(-np.ones(n), A_ub=np.atleast_2d(c * np.ones(n) - l), b_ub=[0.0], bounds=[(0, oi) for oi in o], method="highs")
    return 1 + r.fun


def obs_line(O, l, c):
    o = IJ.as_vec(O); lc = l @ CHI
    if abs(lc) < 1e-15: return np.inf
    sp_ = (c - l @ o) / lc; sgn = np.sign(sp_); cap = min(o[CHI * sgn > 0]); return abs(sp_) / (cap + abs(sp_))


def fwd_fibre(P, l, c):
    """forward budget when the injected law must keep every single and pair table (any m): margin over the support function of the
    pair fibre polytope in the direction l, max {l(Q) : Q a law with P's pair tables} - l(P) (an LP). Returns (eps, Q)."""
    from .duality import marginal_matrix
    m = P.ndim; A = marginal_matrix(m, 2); p = IJ.as_vec(P)
    r = linprog(-l, A_eq=A, b_eq=A @ p, bounds=[(0, None)] * len(p), method="highs"); q = r.x
    return (c - l @ p) / (l @ q - l @ p), IJ.as_law(q, m)
