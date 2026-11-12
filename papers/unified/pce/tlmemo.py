"""Temperley-Lieb memos for a chain of random crossings (rem:twolayer and prop:tlmemo of the companion note). New module; it modifies
nothing.
A 4-ended tangle is built from the [0] tangle by adding crossings one at a time, each either horizontally (tangle addition, H) or
vertically (tangle multiplication, V), as in Conway's rational tangles. Each crossing is a pseudo-crossing that resolves r = +1 or -1.
By the Kauffman bracket skein relation the left part of the chain is a vector alpha [0] + beta [inf] in the 2-dimensional
Temperley-Lieb module (basis: the two crossingless matchings [0] = NW-NE, SW-SE and [inf] = NW-SW, NE-SE), and each crossing acts by a
2 x 2 matrix. The memo records connectivity: its two coordinates are the weights of the two ways the four ends can be joined.
  A, DELTA                     the variable and the loop value -A^2 - A^-2 (unknot normalised to 1)
  step(kind, r)                the 2 x 2 matrix of one crossing (kind 'H' or 'V', resolution r)
  memo(kinds, rs)              the TL vector of the left part after the given crossings, from [0]
  closure(v, which)            numerator ('N': NW-NE, SW-SE) or denominator ('D': NW-SW, NE-SE) closure, a linear functional
  bracket_chain(kinds, rs, which)   the bracket of the closed resolved diagram via the memo
  bracket_brute(kinds, rs, which)   the same by a direct state sum: all 2^k smoothings, loops counted by union-find (independent of the
                               matrices)
  expected_memo(kinds, ps)     the expected memo under independent crossings, P(r_i = +1) = p_i (product of expected matrices)
"""
from __future__ import annotations
import itertools
import sympy as sp

A = sp.Symbol("A"); DELTA = -A ** 2 - A ** -2


def step(kind, r):
    a = A if r == 1 else A ** -1                                      # resolution -1 swaps A and A^-1
    if kind == "H":                                                     # T + X = a T+[0] + a^-1 T+[inf]
        return sp.Matrix([[a, 0], [a ** -1, a + a ** -1 * DELTA]])
    if kind == "V":                                                     # T * X = a T*[inf] + a^-1 T*[0]
        return sp.Matrix([[a ** -1 * DELTA + a, a ** -1], [0, a]])
    raise ValueError(kind)


def memo(kinds, rs):
    v = sp.Matrix([1, 0])
    for k, r in zip(kinds, rs): v = step(k, r) * v
    return v.applyfunc(sp.expand)


def closure(v, which="N"):
    n = sp.Matrix([[DELTA, 1]]) if which == "N" else sp.Matrix([[1, DELTA]])
    return sp.expand((n * v)[0])


def bracket_chain(kinds, rs, which="N"):
    return closure(memo(kinds, rs), which)


def _components(edges, nodes):
    parent = {u: u for u in nodes}
    def find(u):
        while parent[u] != u: parent[u] = parent[parent[u]]; u = parent[u]
        return u
    for u, v in edges: parent[find(u)] = find(v)
    return len({find(u) for u in nodes})


def bracket_brute(kinds, rs, which="N"):
    """Direct state sum. Each crossing is smoothed to [0] or [inf]; the A-smoothing of an H crossing with r = +1 is [0], of a V
    crossing with r = +1 is [inf], and r = -1 swaps them. Pieces are joined end to end and the closure is added; every node then has
    degree 2, so the components are the loops."""
    k = len(kinds); total = sp.Integer(0)
    for state in itertools.product((0, 1), repeat=k):                  # 0: the A-smoothing
        nodes, edges = [], []
        def piece(tag, smooth):
            nw, ne, sw, se = (tag, "nw"), (tag, "ne"), (tag, "sw"), (tag, "se"); nodes.extend([nw, ne, sw, se])
            edges.extend([(nw, ne), (sw, se)] if smooth == "0" else [(nw, sw), (ne, se)])
            return {"NW": nw, "NE": ne, "SW": sw, "SE": se}
        T = piece("start", "0"); power = 0
        for i, (kind, r, s) in enumerate(zip(kinds, rs, state)):
            a_smooth = ("0" if kind == "H" else "inf") if r == 1 else ("inf" if kind == "H" else "0")
            b_smooth = "inf" if a_smooth == "0" else "0"
            P = piece(i, a_smooth if s == 0 else b_smooth); power += 1 if s == 0 else -1
            if kind == "H":
                edges.extend([(T["NE"], P["NW"]), (T["SE"], P["SW"])]); T = {"NW": T["NW"], "SW": T["SW"], "NE": P["NE"], "SE": P["SE"]}
            else:
                edges.extend([(T["SW"], P["NW"]), (T["SE"], P["NE"])]); T = {"NW": T["NW"], "NE": T["NE"], "SW": P["SW"], "SE": P["SE"]}
        edges.extend([(T["NW"], T["NE"]), (T["SW"], T["SE"])] if which == "N" else [(T["NW"], T["SW"]), (T["NE"], T["SE"])])
        total += A ** power * DELTA ** (_components(edges, nodes) - 1)
    return sp.expand(total)


def expected_memo(kinds, ps):
    v = sp.Matrix([1, 0])
    for k, p in zip(kinds, ps): v = (p * step(k, 1) + (1 - p) * step(k, -1)) * v
    return v.applyfunc(sp.expand)
