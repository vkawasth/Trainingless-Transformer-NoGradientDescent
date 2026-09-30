"""Phase changes as transformations of presheaves: the natural / non-natural split.

A bias table b(s, t) on the bipartite source-topic graph is a 1-cochain. Two periods give b (before) and b'
(after); the phase change is Delta = b' - b.
  Proposition (naturality). The change is a natural transformation of the gluing data -- a family of
  per-source and per-topic re-calibrations eta_s, eta_t with Delta(s,t) = eta_s + eta_t -- iff Delta is a
  coboundary, iff Delta Phi = 0 on every loop. The non-natural part is the loop residual of Delta.
natural_split(Delta) returns the least-squares coboundary part (eta_s, eta_t), the residual (non-natural
part) and the loop changes Delta Phi on all 4-cycles.
"""
from __future__ import annotations
import itertools
import numpy as np


def natural_split(delta: np.ndarray):
    S, T = delta.shape
    X = np.zeros((S * T, S + T))
    for s in range(S):
        for t in range(T):
            X[s * T + t, s] = 1; X[s * T + t, S + t] = 1
    coef, *_ = np.linalg.lstsq(X, delta.ravel(), rcond=None)
    natural = (X @ coef).reshape(S, T)
    resid = delta - natural
    loops = {}
    for (a, b), (t1, t2) in itertools.product(itertools.combinations(range(S), 2), itertools.combinations(range(T), 2)):
        loops[(a, b, t1, t2)] = float(delta[a, t1] - delta[a, t2] - delta[b, t1] + delta[b, t2])
    return dict(eta_source=coef[:S], eta_topic=coef[S:], natural=natural, non_natural=resid, delta_phi=loops)
