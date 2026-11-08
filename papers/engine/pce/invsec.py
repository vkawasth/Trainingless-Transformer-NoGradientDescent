"""Bayesian inversion and sections of the marginal projection (companion note, section 12.21). New module; it modifies nothing and uses
pce/selnat.py and pce/selcomp.py. (The lab's inversion module, pce/inversion.py, treats inversion of kernels and the conditioning operator;
this module treats inversion against composition rules.)
  kernels(g)                 (p, K, q, Kdag) of a coupling g: K = g / p by rows, Kdag = g^T / q by rows; rows of null events are None
  coupling(p, K)             p(x) K(y | x)
  composite(rule, g1, g2)    the end marginal of the joint law a rule selects on the chain fibre: 'markov', 'sel' (facet leximin) or 'lex'
                             (the lexicographic minimum in cell order (i, j, k)) or 'tilt' (the vertex minimising a fixed objective that
                             treats the two ends differently; not reversal-equivariant)
  inverts(rule, g1, g2)      composite(rule, g1, g2)^T == composite(rule, g2^T, g1^T)
"""
from __future__ import annotations
import itertools
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import linprog
from . import selnat as N
from . import selcomp as S


def kernels(g):
    p = N.rows(g); q = N.cols(g)
    K = [[g[i][j] / p[i] for j in range(len(q))] if p[i] else None for i in range(len(p))]
    Kd = [[g[i][j] / q[j] for i in range(len(p))] if q[j] else None for j in range(len(q))]
    return p, K, q, Kd


def coupling(p, K): return [[p[i] * (K[i][j] if K[i] is not None else 0) for j in range(len(K[0] if K[0] is not None else K[1]))] for i in range(len(p))]


def lexfirst(g1, g2):
    """lexicographic minimum of the compatible 3-way tables in cell order (i, j, k): minimise cell (0,0,0), fix it, then the next cell,
    ... (linear programs, each value made exact and the final table verified exactly)"""
    shape, cells, A, b = S.fibre_constraints([g1, g2]); n = len(cells); Af = np.array(A, float); bf = np.array([float(x) for x in b])
    fixed = {}
    for c in range(n):
        Ae = np.vstack([Af] + [np.eye(n)[k][None] for k in fixed]) if fixed else Af; be = np.r_[bf, [float(v) for v in fixed.values()]]
        r = linprog(np.eye(n)[c], A_eq=Ae, b_eq=be, bounds=[(0, None)] * n, method="highs"); fixed[c] = Fr(r.fun).limit_denominator(10 ** 6)
    x = [fixed[c] for c in range(n)]
    assert all(v >= 0 for v in x) and all(sum(A[r][k] * x[k] for k in range(n)) == b[r] for r in range(len(A))), "inexact lexicographic point"
    out = [[Fr(0)] * shape[2] for _ in range(shape[0])]
    for c, v in zip(cells, x): out[c[0]][c[2]] += v
    return out


GT, HT = (0, 2, 1, 3), (0, 1, 2, 3)


def tilted(g1, g2):
    """the vertex of the chain fibre minimising sum G[i] H[k] x_ijk with G = (0, 2, 1, 3) non-monotone and H = (0, 1, 2, 3) monotone: a fixed
    rule that treats the two ends differently, so it is not reversal-equivariant. (Any objective acts on a binary-end fibre only through the
    2 x 2 interaction contrast, which transposition preserves; with 3 states at an end the order of the states can differ.)"""
    shape, cells, A, b = S.fibre_constraints([g1, g2]); n = len(cells)
    c = np.array([GT[i] * HT[k] for (i, j, k) in cells])
    r = linprog(c, A_eq=np.array(A, float), b_eq=np.array([float(x) for x in b]), bounds=[(0, None)] * n, method="highs-ds")
    x = [Fr(v).limit_denominator(10 ** 6) for v in r.x]
    assert all(v >= 0 for v in x) and all(sum(A[q][k] * x[k] for k in range(n)) == b[q] for q in range(len(A))), "inexact vertex"
    out = [[Fr(0)] * shape[2] for _ in range(shape[0])]
    for cc, v in zip(cells, x): out[cc[0]][cc[2]] += v
    return out


def composite(rule, g1, g2):
    g1 = [[Fr(v) for v in r] for r in g1]; g2 = [[Fr(v) for v in r] for r in g2]
    if rule == "markov": return N.markov(g1, g2)
    if rule == "sel": return N.sel(g1, g2)
    if rule == "lex": return lexfirst(g1, g2)
    if rule == "tilt": return tilted(g1, g2)
    raise ValueError(rule)


def inverts(rule, g1, g2):
    return N.eq(N.T(composite(rule, g1, g2)), composite(rule, N.T(g2), N.T(g1)))


def random_chain(rng, a, b, c, den=12, zero=None):
    """a rational chain (g1 in Pi(p, q), g2 in Pi(q, r)) with denominators dividing den^2: g1 from integer weights, g2 = diag(q) K"""
    w = [[rng.randint(1, 6) for _ in range(b)] for _ in range(a)]
    if zero: w[zero[0]][zero[1]] = 0
    t = sum(map(sum, w)); g1 = [[Fr(x, t) for x in r] for r in w]; q = N.cols(g1)
    K = []
    for j in range(b):
        v = [rng.randint(1, 6) for _ in range(c)]; s = sum(v); K.append([Fr(x, s) for x in v])
    g2 = [[q[j] * K[j][k] for k in range(c)] for j in range(b)]
    return g1, g2
