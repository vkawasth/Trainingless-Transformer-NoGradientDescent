"""The Novikov-graded order watch (section sec:novorder). New module; it modifies nothing; uses pce/orderwatch.py.
Events are (edge, sign, round, weight); the weight w >= 0 is the information an event carries (for evidence x bearing on a decision
wall with circuit a: w = |Lambda_a(x)| / N, the circuit log-likelihood ratio of sec:likval, on the valuation scale).
The graded Magnus map sends an event to 1 + s T^(-w) X_e. The coefficient of the chain word X_g1 ... X_gL is a Novikov polynomial
  mu_gamma^Lambda = sum over chain embeddings of (product of signs) T^(-(sum of weights)),
stored as {grade: integer coefficient}. Its valuation is the largest grade with a non-zero coefficient (the heaviest surviving chain).
  graded_count(word, path)        {grade: signed count}; grades rounded to 1e-9
  valuation(coef)                 the leading grade (None if the coefficient is 0)
  heaviest_chain(word, path)      max-plus dynamic programme: the largest total weight of an embedding (signs ignored)
"""
from __future__ import annotations
from . import orderwatch as OW
def graded_count(word, path):
    below = OW.poset([(ev[0], ev[1]) for ev in word]); L = len(path)
    occ = [[i for i, ev in enumerate(word) if OW._matches(ev, e)] for e in path]
    # dynamic programme over links: for each event of link j, the Novikov polynomial of chains ending there
    prev = {i: {round(word[i][3], 9): word[i][1]} for i in occ[0]}
    for j in range(1, L):
        cur = {}
        for i in occ[j]:
            acc = {}
            for k, poly in prev.items():
                if k not in below[i]: continue
                for gr, c in poly.items():
                    g2 = round(gr + word[i][3], 9); acc[g2] = acc.get(g2, 0) + c * word[i][1]
            if acc: cur[i] = acc
        prev = cur
    out = {}
    for poly in prev.values():
        for gr, c in poly.items(): out[gr] = out.get(gr, 0) + c
    return {g: c for g, c in out.items() if c != 0}
def valuation(coef): return max(coef) if coef else None
def heaviest_chain(word, path):
    below = OW.poset([(ev[0], ev[1]) for ev in word]); occ = [[i for i, ev in enumerate(word) if OW._matches(ev, e)] for e in path]
    best = {i: word[i][3] for i in occ[0]}
    for j in range(1, len(path)):
        cur = {}
        for i in occ[j]:
            cand = [b for k, b in best.items() if k in below[i]]
            if cand: cur[i] = max(cand) + word[i][3]
        best = cur
    return max(best.values()) if best else None
