"""p-adic loop test (sum claims) and its hierarchy version (ultrametric depth claims). Exact, no gradients.

(1) Sum claims  x_i + x_j = r_ij  over Z_p (r given to K digits).
    The edge acts by x -> -x + r (linear part -1), so the obstruction lives in H^1 with SIGN-TWISTED coefficients:
    even cycles need the alternating sum r_12 - r_23 + ... to vanish; odd cycles are always solvable when 2 is a
    unit (p odd) and carry the parity obstruction when p = 2.
    Consistency DEPTH m* = the largest m such that the system is solvable mod p^m (Smith normal form over Z).
    Hull reading: there is an x with |x_i + x_j - r_ij|_p <= rho for every claim iff rho >= p^(-m*); the local
    hulls glue exactly at radius p^(-m*). On a single even cycle m* = v_p(alternating loop sum).
(2) Hierarchy claims: "i and j meet at depth exactly d_ij" (depth of their lowest common ancestor; root = 0).
    Realisable by a tree (an ultrametric) iff no claim (i, j, d) has i and j already connected by claims all
    DEEPER than d (the depth is then forced > d): on every cycle the shallowest depth is attained at least twice.
    Processed by union-find in decreasing depth; each violation comes with its witness path.
    Theorem (node-wise distortions are invisible): if every claim's depth is changed by a function of its
    true meeting node only, no violation can appear -- the two shallowest claims on a cycle meet at the same
    node and move together. Detectable failures need item-level disagreement (e.g. one source misplaces a leaf).
"""
from __future__ import annotations
import collections
import numpy as np


# ------------------------------------------------------------------ (1) sum claims over Z_p
def smith(A):
    """Smith normal form over Z with transforms: returns (U, D, V) with U A V = D (python ints)."""
    A = [list(map(int, r)) for r in A]
    m, n = len(A), len(A[0])
    U = [[int(i == j) for j in range(m)] for i in range(m)]
    V = [[int(i == j) for j in range(n)] for i in range(n)]

    def swap_rows(M, i, j): M[i], M[j] = M[j], M[i]
    def swap_cols(M, i, j):
        for r in M: r[i], r[j] = r[j], r[i]
    def add_row(M, src, dst, k):
        M[dst] = [a + k * b for a, b in zip(M[dst], M[src])]
    def add_col(M, src, dst, k):
        for r in M: r[dst] += k * r[src]

    t = 0
    while t < min(m, n):
        piv = [(abs(A[i][j]), i, j) for i in range(t, m) for j in range(t, n) if A[i][j] != 0]
        if not piv:
            break
        _, i, j = min(piv)
        swap_rows(A, t, i); swap_rows(U, t, i); swap_cols(A, t, j); swap_cols(V, t, j)
        done = False
        while not done:
            done = True
            for i in range(t + 1, m):
                if A[i][t]:
                    q = A[i][t] // A[t][t]
                    add_row(A, t, i, -q); add_row(U, t, i, -q)
                    if A[i][t]:
                        swap_rows(A, t, i); swap_rows(U, t, i); done = False
            for j in range(t + 1, n):
                if A[t][j]:
                    q = A[t][j] // A[t][t]
                    add_col(A, t, j, -q); add_col(V, t, j, -q)
                    if A[t][j]:
                        swap_cols(A, t, j); swap_cols(V, t, j); done = False
            if done:                                     # divisibility of the rest by the pivot
                for i in range(t + 1, m):
                    for j in range(t + 1, n):
                        if A[i][j] % A[t][t]:
                            add_row(A, i, t, 1); add_row(U, i, t, 1); done = False; break
                    if not done:
                        break
        if A[t][t] < 0:
            A[t] = [-a for a in A[t]]; U[t] = [-a for a in U[t]]
        t += 1
    return U, A, V


def vp(x, p):
    if x == 0:
        return float("inf")
    k = 0
    while x % p == 0:
        x //= p; k += 1
    return k


def sum_claim_depth(n, claims, p, K):
    """claims: list of (i, j, r) meaning x_i + x_j = r (r an integer standing for an element of Z_p mod p^K).
    Returns m* in {0..K} (K means consistent to full given precision)."""
    A = [[0] * n for _ in claims]; r = []
    for row, (i, j, rr) in zip(A, claims):
        row[i] += 1; row[j] += 1; r.append(int(rr))
    U, D, _ = smith(A)
    c = [sum(U[i][k] * r[k] for k in range(len(r))) for i in range(len(r))]
    depth = float("inf")
    for i in range(len(c)):
        d = D[i][i] if i < min(len(D), len(D[0])) else 0
        if d != 0:
            if vp(d, p) > vp(c[i], p):
                depth = min(depth, vp(c[i], p))
        else:
            depth = min(depth, vp(c[i], p))
    return int(min(depth, K))


# ------------------------------------------------------------------ (2) hierarchy claims
class DSU:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def hierarchy_violations(n, claims):
    """claims: list of (i, j, depth, meta). Returns list of violated claims (i, j, depth, meta, witness_path)."""
    by = collections.defaultdict(list)
    for c in claims:
        by[c[2]].append(c)
    dsu = DSU(n); adj = collections.defaultdict(list); bad = []
    for d in sorted(by, reverse=True):
        for c in by[d]:
            if dsu.find(c[0]) == dsu.find(c[1]):
                bad.append(c + (_path(adj, c[0], c[1]),))
        for c in by[d]:
            dsu.union(c[0], c[1]); adj[c[0]].append((c[1], d)); adj[c[1]].append((c[0], d))
    return bad


def _path(adj, a, b):
    prev = {a: None}; q = collections.deque([a])
    while q:
        x = q.popleft()
        if x == b:
            break
        for y, _ in adj[x]:
            if y not in prev:
                prev[y] = x; q.append(y)
    out, x = [], b
    while x is not None and x in prev:
        out.append(x); x = prev[x]
    return out[::-1]


# ------------------------------------------------------------------ synthetic trees with graded arity
def random_tree(depth, rng, arities=(2, 3, 4, 5)):
    """Complete tree of given depth; each internal node draws an arity. Leaves get p-adic-style digit codes
    (child index at each level). Returns codes (list of tuples) and node arity map {prefix: arity}."""
    codes, arity = [], {}
    def grow(prefix):
        if len(prefix) == depth:
            codes.append(prefix); return
        k = int(rng.choice(arities)); arity[prefix] = k
        for c in range(k):
            grow(prefix + (c,))
    grow(())
    return codes, arity


def lca_depth(a, b):
    d = 0
    while d < len(a) and a[d] == b[d]:
        d += 1
    return d
