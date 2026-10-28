"""The latent-actor lab: two latent types on m binary variables, observed through windows (single, pair and triple tables).

Parametrisation (main note, Prop. pairwise). Weights w1 = w, w2 = 1 - w; harm probabilities a1, a2 in [0, 1]^m; mean mu = w1 a1 + w2 a2;
v = sqrt(w1 w2) (a1 - a2); r = (w2 - w1) / sqrt(w1 w2) = t - 1/t with t = sqrt(w2 / w1). Central moments m_S = v_S s_|S|(r) with
s_2 = 1, s_3 = r, s_(k+1) = r s_k + s_(k-1). Pair tables see mu and v; the weight is the one unseen coordinate (the hiding segment), and
a triple window S with v_S != 0 pins r = m_S / v_S.

Functions
  law(w, a1, a2)                     the joint law on {0,1}^m (array indexed by the outcome tuple)
  moments(P)                         mu, the central pair moments (m x m, diagonal unused), and a function S -> m_S
  sample_moments(P, n, rng, wins)    moments estimated from n draws per window (each window sampled independently)
  fit_v(mu, M)                       v (sign fixed: largest |v_i| positive) by the rank-one completion of the off-diagonal pair moments
  segment(mu, v)                     the feasible interval [t_lo, t_hi] (t = sqrt(w2/w1)) cut out by the box walls; r = t - 1/t
  explanation(mu, v, t)              (w, a1, a2) for a point of the segment
  event_prob(P, event)               probability of an event (a function of the outcome tuple)
  interval(mu, v, event, n=401)      [min, max] of the event probability over the segment, with the argmin / argmax t
  pin_r(mu, v, mS, S)                r from a triple moment
"""
from __future__ import annotations
import itertools
import numpy as np


def law(w, a1, a2):
    a1 = np.asarray(a1, float); a2 = np.asarray(a2, float); m = len(a1); P = np.zeros((2,) * m)
    for x in itertools.product((0, 1), repeat=m):
        x_ = np.array(x)
        P[x] = w * np.prod(np.where(x_ == 1, a1, 1 - a1)) + (1 - w) * np.prod(np.where(x_ == 1, a2, 1 - a2))
    return P


def _marg(P, S):
    m = P.ndim; return P.sum(axis=tuple(i for i in range(m) if i not in S))


def moments(P):
    m = P.ndim; mu = np.array([_marg(P, (i,))[1] for i in range(m)]); M = np.zeros((m, m))
    for i, j in itertools.combinations(range(m), 2):
        M[i, j] = M[j, i] = _marg(P, (i, j))[1, 1] - mu[i] * mu[j]
    def mS(S):
        S = tuple(sorted(S)); T = _marg(P, S); tot = 0.0
        for x in itertools.product((0, 1), repeat=len(S)):
            tot += T[x] * np.prod([x[k] - mu[S[k]] for k in range(len(S))])
        return tot
    return mu, M, mS


def sample_moments(P, n, rng, triples=()):
    """each single, pair and listed triple window is an independent sample of size n; singles are pooled from the pair windows that
    contain them (as a sponsor reporting pair trials would); returns mu, M and a dict of triple moments"""
    m = P.ndim; pairs = list(itertools.combinations(range(m), 2)); tabs = {}
    for S in pairs + list(triples):
        T = _marg(P, S).ravel(); cnt = rng.multinomial(n, T / T.sum()); tabs[S] = (cnt / n).reshape((2,) * len(S))
    mu = np.zeros(m); c = np.zeros(m)
    for (i, j) in pairs:
        T = tabs[(i, j)]; mu[i] += T[1].sum(); mu[j] += T[:, 1].sum(); c[i] += 1; c[j] += 1
    mu = mu / c; M = np.zeros((m, m))
    for (i, j) in pairs:
        T = tabs[(i, j)]; M[i, j] = M[j, i] = T[1, 1] - T[1].sum() * T[:, 1].sum()
    trip = {}
    for S in triples:
        T = tabs[S]; mloc = [T.sum(axis=tuple(k for k in range(3) if k != q))[1] for q in range(3)]; tot = 0.0
        for x in itertools.product((0, 1), repeat=3): tot += T[x] * np.prod([x[q] - mloc[q] for q in range(3)])
        trip[S] = tot
    return mu, M, trip


def fit_v(mu, M, iters=500):
    """rank-one completion of the off-diagonal pair moments: v v^T agrees with M off the diagonal"""
    m = len(mu); A = M.copy(); d = np.zeros(m)
    for _ in range(iters):
        A[np.diag_indices(m)] = d; vals, vecs = np.linalg.eigh(A); k = np.argmax(vals)
        v = vecs[:, k] * np.sqrt(max(vals[k], 0.0)); dn = v ** 2
        if np.max(np.abs(dn - d)) < 1e-15: break
        d = dn
    i = np.argmax(np.abs(v)); return v if v[i] >= 0 else -v


def segment(mu, v):
    lo, hi = 1e-12, np.inf
    for mi, vi in zip(mu, v):
        if vi > 1e-15: hi = min(hi, (1 - mi) / vi); lo = max(lo, vi / mi)
        elif vi < -1e-15: hi = min(hi, mi / (-vi)); lo = max(lo, -vi / (1 - mi))
    return lo, hi


def explanation(mu, v, t):
    w = 1 / (1 + t * t); return w, np.clip(mu + v * t, 0, 1), np.clip(mu - v / t, 0, 1)


def r_of_t(t): return t - 1 / t


def t_of_r(r): return (r + np.sqrt(r * r + 4)) / 2


def event_prob(P, event):
    return float(sum(P[x] for x in itertools.product((0, 1), repeat=P.ndim) if event(x)))


def interval(mu, v, event, n=401):
    lo, hi = segment(mu, v)
    if not np.isfinite(hi) or hi <= lo: return None
    ts = np.exp(np.linspace(np.log(lo), np.log(hi), n)); ps = [event_prob(law(*explanation(mu, v, t)), event) for t in ts]
    return min(ps), max(ps), ts[int(np.argmin(ps))], ts[int(np.argmax(ps))], ts, np.array(ps)


def pin_r(v, mS, S):
    vS = np.prod([v[i] for i in S]); return mS / vS if abs(vS) > 1e-12 else None
