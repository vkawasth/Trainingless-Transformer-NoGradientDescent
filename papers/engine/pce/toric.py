"""Several experts as toric families on a shared evidence space, and how their lattice polytopes act when the experts are combined.

An expert has outcomes a in a finite set A of lattice points in Z^d and weights c_a. Given evidence u in R^d (log-odds of d observed
features) it says  p(a | u) ∝ exp(beta (c_a + a.u)): a log-linear (toric) family whose partition function is a Laurent polynomial in
z = e^{beta u} with Newton polytope P = conv(A). Its decision (most probable outcome) does not depend on beta, and changes on its
*indecision locus*: the evidence where its top two outcomes tie, which is the tropical hypersurface of max_a (c_a + a.u).

  ToricExpert(A, c)                    one expert
  winners(e)                           the outcomes that are most probable for some evidence (vertices of the regular subdivision of P)
  mixed_volume(Ps)                     MV(P_1..P_d) by inclusion-exclusion of volumes of Minkowski sums (MV(P,..,P) = d! vol P)
  joint_ties(experts)                  d experts in R^d: the evidence points where every expert is tied, each with multiplicity |det| of
                                       its tie directions (generic weights). Tropical Bernstein: the total is MV(P_1..P_d).
  pair_regions(e1, e2)                 two experts on separate questions, decided jointly (product of their laws): the pairs (a, b) that
                                       are jointly most probable somewhere; they are the cells of the mixed subdivision of P_1 + P_2
  pool_winners(e1, e2, rule)           two experts on the same question (common outcomes, each with its own exponents and weights):
                                       'log' = product of experts (scores add: polytope conv{A1(x) + A2(x)}), 'mix' = linear pool
                                       (beta -> infinity: scores max). The outcomes that win somewhere.
  mix_decision(e1, e2, u, beta)        linear pool at finite beta (log-sum-exp of the two experts' scores, per outcome)
  sum_law(e1, e2, u, beta)             two independent experts, decision about the sum statistic a + b (generating functions multiply:
                                       Newton polytope P_1 + P_2); law of the sum at finite beta
  kernel_moves(A_rows, max_entry)      integer moves (kernel vectors with small entries) of a marginal design matrix, and their degrees
  fibre_connected(A_rows, n, moves)    whether the moves connect every fibre of tables with n counts (a Markov basis up to n)
"""
from __future__ import annotations
import itertools, math
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull


@dataclass
class ToricExpert:
    A: np.ndarray          # (n, d) integer exponents
    c: np.ndarray          # (n,) weights

    def scores(self, u): return self.c + self.A @ u


def _region_lp(G, h, d, R=1e3):
    """max t s.t. G u + h >= t (componentwise), |u| <= R. Returns t (> 0: open region non-empty)."""
    if len(G) == 0: return np.inf
    nv = d + 1; c = np.zeros(nv); c[-1] = -1.0
    Aub = np.hstack([-G, np.ones((len(G), 1))]); bub = h
    r = linprog(c, A_ub=Aub, b_ub=bub, bounds=[(-R, R)] * d + [(None, 1.0)], method="highs")
    return -r.fun if r.status == 0 else -np.inf


def winners(e, tol=1e-7):
    out = []
    for a in range(len(e.A)):
        G = np.array([e.A[a] - e.A[b] for b in range(len(e.A)) if b != a]); h = np.array([e.c[a] - e.c[b] for b in range(len(e.A)) if b != a])
        if _region_lp(G, h, e.A.shape[1]) > tol: out.append(a)
    return out


def _vol(points):
    P = np.unique(np.asarray(points, float), axis=0); d = P.shape[1]
    if len(P) <= d: return 0.0
    if np.linalg.matrix_rank(P - P[0]) < d: return 0.0
    return ConvexHull(P).volume


def mixed_volume(Ps):
    d = len(Ps); tot = 0.0
    for k in range(1, d + 1):
        for S in itertools.combinations(range(d), k):
            pts = [np.sum(x, axis=0) for x in itertools.product(*[Ps[i] for i in S])]
            tot += (-1) ** (d - k) * _vol(pts)
    return tot


def joint_ties(experts, tol=1e-9):
    """returns (sum of multiplicities, list of (point, multiplicity))."""
    d = len(experts); pts = []; tot = 0
    edges = [list(itertools.combinations(range(len(e.A)), 2)) for e in experts]
    for combo in itertools.product(*edges):
        M = np.array([experts[k].A[i] - experts[k].A[j] for k, (i, j) in enumerate(combo)], float)
        det = np.linalg.det(M)
        if abs(det) < 1e-9: continue
        rhs = [experts[k].c[j] - experts[k].c[i] for k, (i, j) in enumerate(combo)]
        u = np.linalg.solve(M, rhs)
        good = True
        for k, (i, j) in enumerate(combo):
            s = experts[k].scores(u)
            if s[i] < s.max() - tol: good = False; break
        if good: m = int(round(abs(det))); tot += m; pts.append((u, m))
    return tot, pts


def pair_regions(e1, e2, tol=1e-7):
    d = e1.A.shape[1]; out = []
    w1, w2 = winners(e1), winners(e2)
    for a in w1:
        for b in w2:
            G = [e1.A[a] - e1.A[x] for x in range(len(e1.A)) if x != a] + [e2.A[b] - e2.A[y] for y in range(len(e2.A)) if y != b]
            h = [e1.c[a] - e1.c[x] for x in range(len(e1.A)) if x != a] + [e2.c[b] - e2.c[y] for y in range(len(e2.A)) if y != b]
            if _region_lp(np.array(G), np.array(h), d) > tol: out.append((a, b))
    return out


def pool_winners(e1, e2, rule, tol=1e-7):
    """common outcomes: row x of e1 and of e2 refer to the same outcome."""
    n, d = e1.A.shape; out = []
    if rule == "log":
        e = ToricExpert(e1.A + e2.A, e1.c + e2.c); return winners(e, tol)
    for x in range(n):                                                    # mix: x wins iff for some i its score beats every score
        for i, e in enumerate((e1, e2)):
            G, h = [], []
            for f in (e1, e2):
                for y in range(n):
                    if f is e and y == x: continue
                    G.append(e.A[x] - f.A[y]); h.append(e.c[x] - f.c[y])
            if _region_lp(np.array(G), np.array(h), d) > tol: out.append(x); break
    return out


def mix_decision(e1, e2, u, beta):
    s = np.logaddexp(beta * e1.scores(u), beta * e2.scores(u)); return int(np.argmax(s))


def sum_law(e1, e2, u, beta):
    law = {}
    for a in range(len(e1.A)):
        for b in range(len(e2.A)):
            key = tuple(e1.A[a] + e2.A[b]); law[key] = np.logaddexp(law.get(key, -np.inf), beta * (e1.c[a] + e2.c[b] + (e1.A[a] + e2.A[b]) @ u))
    return law


# -------------------------------------------------------------------------- gluing: moves of a marginal design matrix
def kernel_moves(A_rows, max_entry=1):
    A = np.asarray(A_rows); n = A.shape[1]; out = []
    for v in itertools.product(range(-max_entry, max_entry + 1), repeat=n):
        v = np.array(v)
        if not v.any() or (A @ v).any(): continue
        first = v[np.nonzero(v)[0][0]]
        if first < 0: continue                                            # one of each +-pair
        out.append((v, int(v[v > 0].sum())))
    return out


def fibre_connected(A_rows, n, moves):
    A = np.asarray(A_rows); m = A.shape[1]
    tables = [np.array(t) for t in itertools.product(range(n + 1), repeat=m) if sum(t) == n]
    fib = {}
    for t in tables: fib.setdefault(tuple(A @ t), []).append(tuple(t))
    for key, T in fib.items():
        if len(T) == 1: continue
        S = set(T); seen = {T[0]}; stack = [T[0]]
        while stack:
            x = np.array(stack.pop())
            for v in moves:
                for s in (1, -1):
                    y = x + s * v
                    if (y >= 0).all() and tuple(y) in S and tuple(y) not in seen: seen.add(tuple(y)); stack.append(tuple(y))
        if len(seen) != len(S): return False
    return True
