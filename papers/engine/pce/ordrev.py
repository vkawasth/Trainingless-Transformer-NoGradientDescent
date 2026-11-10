"""Ordered co-revision histories (companion note, 'Ordered co-revision'). New module; it modifies nothing.

Events. Sources sit at positions 1..k on a line. An event is a co-revision of an adjacent pair (i, i+1) with a sign: +1 when the pair
moves towards each other's view (adoption; repeated, reinforcement), -1 when it moves back (withdrawal; after adoption, reversal). A
history is a word of such events.
Geometric braid. Each event is realised as one full turn of the two strands about their midpoint (+1 clockwise, which the reading convention of
pce/bundlebraid, front = larger imaginary part, records as sigma_i^2), strands at unit
spacing, piecewise in time; it is the pure braid A_{i,i+1} = sigma_i^2 or its inverse. Closure: the standard closure of the braid.
Independence. Events on disjoint pairs commute as braids (sigma_i^2 and sigma_j^2 commute for |i - j| >= 2), events on overlapping pairs
do not; so the braid of a history depends only on its trace (order up to swapping adjacent independent events) and on free cancellation.
Functions
  arrangements(counts)           all distinct words with the given multiset of events (events are (pair, sign))
  braid_word(h)                  the sigma word of a history;  braid_key(h, k): Artin images (equal iff same braid)
  free_reduce(h)                 cancel adjacent inverse events
  signature(h, npairs, level)    truncated path signature of the cumulative count path (one coordinate per pair, unit steps), as a tuple of
                                 arrays per level (Chen: product of exp(step))
  milnor3(h)                     for k = 3: linking numbers and second-order Magnus coefficient (pce/braidknot.milnor)
  magnus_key(h, deg)             Magnus coefficients up to degree deg of the three longitudes (k = 3)
  alexander3(h), jones3(h)       closure invariants for k = 3: Alexander polynomial (reduced Burau) and Jones polynomial
                                 V = t (-sqrt t)^(e - 2) (t + 1/t + tr Burau_r)  (validated against the state sum)
  trace_key(h)                   Cartier-Foata normal form of the history under the independence of disjoint pairs (positive words)
  realise(h)                     the planar strands over time (for checking the geometric braid against the algebraic one)
"""
from __future__ import annotations
import itertools
import numpy as np
import sympy as sp
from . import braidknot as BK, bundlebraid as BB


def arrangements(counts):
    items = [e for e, c in counts.items() for _ in range(c)]
    return sorted(set(itertools.permutations(items)))


def braid_word(h):
    out = []
    for (i, s) in h: out += [i * s, i * s]
    return out


def braid_key(h, k): img = BK.artin(braid_word(h), k); return tuple(tuple(img[j]) for j in sorted(img))


def free_reduce(h):
    out = []
    for e in h:
        if out and out[-1][0] == e[0] and out[-1][1] == -e[1]: out.pop()
        else: out.append(e)
    return tuple(out)


def signature(h, npairs, level):
    S = [np.ones(())] + [np.zeros((npairs,) * m) for m in range(1, level + 1)]
    for (i, s) in h:
        x = np.zeros(npairs); x[i - 1] = s
        E = [np.ones(())] + [x.copy()]
        for m in range(2, level + 1): E.append(np.multiply.outer(E[-1], x) / m)
        N = [np.ones(())]
        for m in range(1, level + 1):
            acc = np.zeros((npairs,) * m)
            for j in range(m + 1):
                a, b = S[j], E[m - j]
                acc = acc + (np.multiply.outer(a, b) if a.ndim and b.ndim else a * b)
            N.append(acc)
        S = N
    return S


def sig_key(h, npairs, level, nd=9):
    S = signature(h, npairs, level); return tuple((np.round(a, nd) + 0.0).tobytes() for a in S[1:])


def milnor3(h): return BK.milnor(braid_word(h))


def magnus_key(h, deg=3):
    L = BK.longitudes(braid_word(h), 3); out = []
    for i in (1, 2, 3):
        M = BK.magnus(L[i], deg); out.append(tuple(sorted((kk, round(v)) for kk, v in M.items() if abs(v) > 1e-9)))
    return tuple(out)


_t = sp.Symbol("t")


def alexander3(h):
    w = braid_word(h)
    return sp.factor(sp.cancel((sp.eye(2) - BB._burau(w, 3, _t)).det() / (1 + _t + _t ** 2))) if w else sp.Integer(0)


def jones3(h):
    w = braid_word(h); M = BB._burau(w, 3, _t) if w else sp.eye(2); e = sum(1 if g > 0 else -1 for g in w)
    return sp.expand(sp.simplify(_t * (-sp.sqrt(_t)) ** (e - 2) * (_t + 1 / _t + M.trace())))


def trace_key(h):
    """Cartier-Foata normal form: events on pairs i, j are independent iff |i - j| >= 2"""
    steps = []
    for e in h:
        lvl = 0
        for d in range(len(steps) - 1, -1, -1):
            if any(abs(f[0] - e[0]) < 2 for f in steps[d]): lvl = d + 1; break
        if lvl == len(steps): steps.append([])
        steps[lvl].append(e)
    return tuple(tuple(sorted(s)) for s in steps)


def realise(h, k, N=60):
    z = np.arange(k, dtype=complex); P = [z.copy()]
    for (i, s) in h:
        a, b = i - 1, i; z0 = P[-1]; c = (z0[a] + z0[b]) / 2; r = (z0[a] - z0[b]) / 2
        for u in np.linspace(0, 1, N)[1:]:
            w = z0.copy(); ph = np.exp(-2j * np.pi * s * u); w[a] = c + r * ph; w[b] = c - r * ph; P.append(w)
    return np.array(P)


def _cf(word):
    """Cartier-Foata normal form of a word of pair indices (0-based) under the independence |i - j| >= 2"""
    steps = []
    for e in word:
        lvl = 0
        for d in range(len(steps) - 1, -1, -1):
            if any(abs(f - e) < 2 for f in steps[d]): lvl = d + 1; break
        if lvl == len(steps): steps.append([])
        steps[lvl].append(e)
    return tuple(tuple(sorted(s)) for s in steps)


def pc_signature(h, npairs, level):
    """partially commutative signature: the signature's image in the algebra where X_i X_j = X_j X_i for independent pairs; returned as
    a dict per level from trace classes of index words to coefficients"""
    S = signature(h, npairs, level); out = []
    for m in range(1, level + 1):
        acc = {}
        for idx in itertools.product(range(npairs), repeat=m):
            c = S[m][idx]
            if abs(c) > 1e-12: key = _cf(idx); acc[key] = acc.get(key, 0.0) + c
        out.append(acc)
    return out


def pc_sig_key(h, npairs, level, nd=9):
    return tuple(tuple(sorted((k, round(v, nd) + 0.0) for k, v in d.items() if abs(round(v, nd)) > 0)) for d in pc_signature(h, npairs, level))
