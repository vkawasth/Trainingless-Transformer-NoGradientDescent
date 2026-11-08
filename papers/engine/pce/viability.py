"""Trajectories of decision viability (companion note, section 12.25). New module; it modifies nothing and uses pce/margsec.py,
pce/poisonflip.py and pce/voigeom.py.
  q_some(d, w, tau), q_all(d, w, tau, DV)   the viability shares: the least share of the pooled data that a consistent batch (any joint law)
                                            must have so that some rule / every rule puts w.x <= tau. The reachable set is exactly
                                            (1 - q) d + q D, so these are one LP (some) and one LP per dual vertex (all).
  rank_radius(d, U, LAM)                    a certified lower bound on the edit cost needed to change the ranking of actions by robust value
  segment_crossings(...)                    sign changes of L - tau, U - tau and of robust-value differences along a segment of data
  window_stream(laws, W)                    data of a sliding window of the last W batches (each batch a joint law)
  closed_loop(...)                          the same, with the next batch's law chosen by the current decision (finite state)
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from . import margsec as G
from . import poisonflip as P

M = np.array(G.M, float)


def q_some(d, w, tau):
    """min q s.t. some joint law P has pi P = (1 - q) d + pi R, R >= 0, sum R = q (a batch of share q from the law R / q), and w.P <= tau"""
    d = np.asarray(d, float); n = 16
    # variables: q, P (16), R (16)
    Aeq = np.vstack([np.c_[d[:, None], M, -M],                                   # M P - M R + q d = d
                     np.r_[0.0, np.ones(n), np.zeros(n)][None],                   # sum P = 1
                     np.r_[-1.0, np.zeros(n), np.ones(n)][None]])                 # sum R = q
    beq = np.r_[d, 1.0, 0.0]
    r = linprog(np.r_[1.0, np.zeros(2 * n)], A_ub=np.r_[0.0, np.asarray(w, float), np.zeros(n)][None], b_ub=[tau], A_eq=Aeq, b_eq=beq,
                bounds=[(0, 1)] + [(0, None)] * (2 * n), method="highs")
    return r.x[0] if r.status == 0 else None


def q_all(d, w, tau, DV):
    q, _ = P.poison_mix(d, w, tau, DV); return q


def robust_values(d, U):
    return [P.lower_upper(d, u)[0] for u in U]                                   # max-min value of each action: min over the fibre


def lipschitz_action(u, SU):
    return max(P.lipschitz(P.dual_lower(u), SU), 0.0)


def rank_radius(d, U, LAM):
    """min over adjacent pairs in the ranking by robust value of (gap) / (Lam_a + Lam_b): no undetectable edit cheaper than this changes
    the ranking word"""
    v = robust_values(d, U); order = np.argsort(v)[::-1]
    return min((v[order[i]] - v[order[i + 1]]) / (LAM[order[i]] + LAM[order[i + 1]]) for i in range(len(v) - 1)), order


def sign_changes(xs, tol=1e-10):
    s = [np.sign(x) for x in xs if abs(x) > tol]; return sum(1 for a, b in zip(s, s[1:]) if a != b)


def window_stream(batches, W):
    """batches: list of joint laws (arrays of 16); returns the data of the window mean of the last W batches at each step"""
    out = []
    for t in range(len(batches)):
        win = batches[max(0, t - W + 1): t + 1]; out.append(M @ np.mean(win, axis=0))
    return out


def closed_loop(x_act, x_abstain, start, W, T, w, tau, rule):
    """the next batch's law is x_act if the current rule says act (w.rule(d) >= tau), else x_abstain; window of W batches.
    Returns the decision sequence and the data path."""
    hist = list(start); ds, acts = [], []
    for t in range(T):
        d = M @ np.mean(hist[-W:], axis=0); a = int(float(np.asarray(w, float) @ rule(d)) >= tau); ds.append(d); acts.append(a)
        hist.append(x_act if a else x_abstain)
    return acts, ds


def eventual_period(seq, maxp=64, tail=None):
    """the least p such that the last `tail` entries (default: the second half of seq) are p-periodic; None if none <= maxp"""
    tail = len(seq) // 2 if tail is None else tail; t = seq[-tail:]
    for p in range(1, min(maxp, tail // 2) + 1):
        if all(t[i] == t[i + p] for i in range(len(t) - p)): return p
    return None
