"""An executable algebra of in-context learning (ICL) on finite models.

Setting. Outcomes form a finite alphabet Y. A hypothesis is a law p on Y, optionally with a structural descriptor (rule name,
provenance). A context state is a finite law Pi over hypotheses -- level two of the posterior tower. Its one-step predictive is
rho = mu(Pi) (level one). A richer state Omega (a law over such Pi) flattens to Pi = mu(Omega) (level three).

Operations.
  update(Pi, e)            Bayes update by an example e in Y (conditioning)
  update_seq(Pi, seq)      repeated update
  mix(Pi1, Pi2, w)         mixture of alternatives (keeps branches and their descriptors)
  prob(Pi, seq)            probability of a continuation seq
  canonical(Pi)            merge hypotheses with the same law (observational equivalence), drop zero weights: the minimal state
  equivalent(Pi1, Pi2)     behavioral equivalence: equal canonical states (Theorem: iff all continuations have equal probability)
  abstract(Pi, key)        structural abstraction: merge hypotheses with the same key; valid iff the key refines observational equality
  hankel(fn, contexts, continuations)   the matrix H[c, f] = P(c f) of any black-box sequence model
  state_dimension(H, tol)  numerical rank of H: the minimal state dimension of the model that produced it (Carlyle-Paz / PSR rank)
"""
from __future__ import annotations
import itertools
import numpy as np

Hyp = tuple   # (law as a tuple of floats over Y, descriptor)


def state(hyps, weights):
    w = np.asarray(weights, float); w = w / w.sum()
    return [(tuple(float(x) for x in np.asarray(p, float) / np.sum(p)), d, float(wi)) for (p, d), wi in zip(hyps, w)]


def predictive(Pi):
    return sum(w * np.array(p) for p, _, w in Pi)


def prob(Pi, seq):
    return float(sum(w * np.prod([p[y] for y in seq]) for p, _, w in Pi))


def update(Pi, e):
    z = sum(w * p[e] for p, _, w in Pi)
    return [(p, d, w * p[e] / z) for p, d, w in Pi]


def update_seq(Pi, seq):
    for e in seq: Pi = update(Pi, e)
    return Pi


def mix(Pi1, Pi2, w):
    return [(p, d, w * a) for p, d, a in Pi1] + [(p, d, (1 - w) * a) for p, d, a in Pi2]


def canonical(Pi, decimals=12):
    acc = {}
    for p, _, w in Pi:
        if w <= 0: continue
        key = tuple(round(x, decimals) for x in p); acc[key] = acc.get(key, 0.0) + w
    return sorted((k, round(v, decimals)) for k, v in acc.items())


def equivalent(Pi1, Pi2, decimals=10):
    a, b = canonical(Pi1, decimals), canonical(Pi2, decimals)
    return len(a) == len(b) and all(k1 == k2 and abs(v1 - v2) < 10 ** -(decimals - 2) for (k1, v1), (k2, v2) in zip(a, b))


def abstract(Pi, key):
    """merge hypotheses by key(descriptor, law); the merged law is the weight-average (this is where information can be lost)."""
    acc = {}
    for p, d, w in Pi:
        k = key(d, p); P, W = acc.get(k, (np.zeros(len(p)), 0.0)); acc[k] = (P + w * np.array(p), W + w)
    return [(tuple(P / W), k, W) for k, (P, W) in acc.items() if W > 0]


def flatten(Omega):
    """level three to level two: Omega = [(weight, Pi), ...] -> mu(Omega)."""
    out = []
    for a, Pi in Omega: out += [(p, d, a * w) for p, d, w in Pi]
    return out


def sequences(Y, max_len):
    return [s for n in range(max_len + 1) for s in itertools.product(range(Y), repeat=n)]


def hankel(fn, contexts, continuations):
    return np.array([[fn(tuple(c) + tuple(f)) for f in continuations] for c in contexts])


def state_dimension(H, tol=1e-9):
    s = np.linalg.svd(H, compute_uv=False)
    return int((s > tol * s[0]).sum()) if s[0] > 0 else 0
