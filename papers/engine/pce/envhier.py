"""Structured environments (layers paper, Section "Structured environments"). New module. Exact arithmetic.

Two structures beyond a flat environment set:
  * a hierarchy of environments (sites -> regions -> all), given by surjective coarsening maps; each level is a product comonad, and
    coarsening is a morphism of comonads compatible with the distributive laws;
  * a neighbourhood context (the store comonad S_E X = X^E x E), where a record sees the states of all sites; with the independent
    coupling it satisfies three of the four axioms of a distributive law over D, and fails comultiplication.

  coarsen(Lam, parent)               push a law on L x E x X forward along e -> parent[e]
  orders_up(m, r, parent, r_up)      nested pooled orders: v at the parent level = min over children of (r_child + v_child)
  survivors(v)                       sources of least order at each environment
  store_* functions                  the store comonad on finite sets and the independent coupling lambda
"""
from __future__ import annotations
import itertools
from fractions import Fraction as Fr


def coarsen(Lam, parent):
    out = {}
    for (l, e, x), v in Lam.items():
        k = (l, parent[e], x); out[k] = out.get(k, 0) + v
    return out


def orders_up(v, r, parent):
    """v[(l, e)] orders at the child level, r[e] the child environment's own order within its parent; returns v at the parent level"""
    out = {}
    for (l, e), val in v.items():
        k = (l, parent[e]); out[k] = min(out.get(k, val + r[e]), val + r[e])
    return out


def survivors(v):
    envs = {e for _, e in v}; out = {}
    for e in envs:
        m = min(val for (l, e2), val in v.items() if e2 == e); out[e] = {l for (l, e2), val in v.items() if e2 == e and val == m}
    return out


# ------------------------------------------------------------------------------------------------ the store comonad and the independent coupling
def store_elements(X, E):
    """elements (g, e) of S X = X^E x E, with g a tuple indexed by E"""
    return [(g, e) for g in itertools.product(X, repeat=len(E)) for e in E]


def lam_indep(f, e):
    """lambda(f, e) for f: E -> D X (a tuple of dicts): the law of (g, e) with g drawn independently, g(e') ~ f(e')"""
    out = {}
    for g in itertools.product(*[list(d) for d in f]):
        p = Fr(1)
        for d, x in zip(f, g): p *= d[x]
        out[(g, e)] = out.get((g, e), 0) + p
    return out


def push(law, fn):
    out = {}
    for k, v in law.items():
        y = fn(k); out[y] = out.get(y, 0) + v
    return out
