"""Gluing a pseudo-diagram along a chain of pieces (companion note, prop:chainglue). New module; it modifies nothing and uses
pce/boundary.py. Pieces are contiguous blocks of crossings; the cut after piece i carries a state c_i, a function of the crossings to its
left that is recursive: c_i = h_i(c_{i-1}, piece i).
  quotient_states(f, n, cuts)     the left quotient at each cut, as recursive state functions (assignment of crossings 1..cut -> class)
  glue_at(P, cut, state)          G(P) = P(left, c) P(right | c): the two sides made conditionally independent given the cut state
  chain_law(P, cuts, states)      prod_i P(piece_i | c_{i-1}) (c_0 trivial)
  ipf_maxent(P, cuts, states)     the maximum-entropy law with the tables of (c_{i-1}, piece_i) for every i (iterative fitting)
"""
from __future__ import annotations
import itertools
import numpy as np
from . import boundary as B


def quotient_states(f, n, cuts):
    out = []
    for c in cuts:
        TS, _ = B.classes(f, list(range(c)), n); out.append(lambda e, c=c, TS=TS: TS[tuple(e[:c])])
    return out


def _marg(P, key):
    m = {}
    for e, p in P.items():
        k = key(e); m[k] = m.get(k, 0.0) + p
    return m


def glue_at(P, cut, state):
    left = lambda e: tuple(e[:cut]); right = lambda e: tuple(e[cut:])
    pLc = _marg(P, lambda e: (left(e), state(e))); pRc = _marg(P, lambda e: (right(e), state(e))); pc = _marg(P, state)
    out = {}
    for e in P:
        c = state(e); out[e] = pLc.get((left(e), c), 0.0) * pRc.get((right(e), c), 0.0) / pc[c] if pc.get(c, 0) > 0 else 0.0
    return out


def chain_law(P, cuts, states):
    bounds = [0] + list(cuts) + [len(next(iter(P)))]; out = {}
    st = [lambda e: 0] + list(states)
    tabs = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        joint = _marg(P, lambda e, a=a, b=b, i=i: (st[i](e), tuple(e[a:b]))); prev = _marg(P, st[i]); tabs.append((a, b, joint, prev, st[i]))
    for e in P:
        p = 1.0
        for a, b, joint, prev, s in tabs:
            c = s(e); p *= joint.get((c, tuple(e[a:b])), 0.0) / prev[c] if prev.get(c, 0) > 0 else 0.0
        out[e] = p
    return out


def ipf_maxent(P, cuts, states, iters=4000, tol=1e-14):
    bounds = [0] + list(cuts) + [len(next(iter(P)))]; st = [lambda e: 0] + list(states); keys = list(P)
    groups = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]; kf = lambda e, a=a, b=b, i=i: (st[i](e), tuple(e[a:b])); tgt = _marg(P, kf)
        for k, v in tgt.items(): groups.append((np.array([kf(e) == k for e in keys]), v))
    z = np.full(len(keys), 1 / len(keys))
    for _ in range(iters):
        err = 0.0
        for m, v in groups:
            cur = z[m].sum(); err = max(err, abs(cur - v))
            if cur > 0: z[m] *= v / cur
        if err < tol: break
    return dict(zip(keys, z))
