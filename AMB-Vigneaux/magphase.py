"""Magnitude (Layer 2) and phase (holonomy) — estimators and statistics.

Part 1  Tsallis-2 estimators (unbiased), Shannon plug-in, and the algebraic CF formula
        for binary cycles with uniform marginals (Prop. "CF from moduli and one bit").
Part 2  Dimension deficit for affine operation systems over F_p: from estimated edge
        supports, the loop map in homogeneous coordinates M = [[a, b], [0, 1]] and
        rank_Fp(M − I); plus the affine global-section count.
Part 3  Finite-field cocycle check for dimension (log_q of the support) under the
        Shannon (conditional-expectation) action, and for S_0 under the α = 0 action.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .holonomy import Graph


# ---------------------------------------------------------------------------
# Part 1: Tsallis-2 and Shannon estimators
# ---------------------------------------------------------------------------
def sum_sq_unbiased(counts) -> float:
    """Unbiased estimator of Σ p_i² from multinomial counts: Σ n(n−1) / N(N−1)."""
    c = np.asarray(counts, float).ravel()
    N = c.sum()
    return float((c * (c - 1)).sum() / (N * (N - 1)))


def s2_unbiased(counts) -> float:
    return 1.0 - sum_sq_unbiased(counts)


def i2_unbiased(table) -> float:
    """I_2(a;b) = S_2(a) + S_2(b) − S_2(ab) = S_2(b) − a·_2 S_2(b); each Σp² term unbiased."""
    T = np.asarray(table, float)
    return s2_unbiased(T.sum(1)) + s2_unbiased(T.sum(0)) - s2_unbiased(T)


def _H(p) -> float:
    p = np.asarray(p, float).ravel()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def mi_plugin(table, miller_madow: bool = False) -> float:
    """Shannon mutual information (nats), plug-in, optionally Miller–Madow corrected."""
    T = np.asarray(table, float)
    N = T.sum()
    P = T / N
    I = _H(P.sum(1)) + _H(P.sum(0)) - _H(P)
    if miller_madow:
        k = lambda x: int((np.asarray(x) > 0).sum())
        # plug-in entropies are biased down by (k−1)/2N, so plug-in MI is biased up by
        # ((k_ab−1) − (k_a−1) − (k_b−1))/2N; subtract it
        I -= ((k(T) - 1) - (k(T.sum(1)) - 1) - (k(T.sum(0)) - 1)) / (2 * N)
    return I


def modulus_from_i2(I2: float) -> float:
    """|s| = sqrt(4 I_2 − 1) for a binary edge with uniform marginals."""
    return float(np.sqrt(max(0.0, min(1.0, 4 * I2 - 1))))


def modulus_from_mi(I: float) -> float:
    """Invert I = log 2 − h((1+|s|)/2) (nats) by bisection."""
    h = lambda p: -p * np.log(p) - (1 - p) * np.log(1 - p) if 0 < p < 1 else 0.0
    target = np.log(2) - I
    if target <= 0:
        return 1.0
    if target >= np.log(2):
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        m = (lo + hi) / 2
        if h((1 + m) / 2) > target:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def cf_cycle_from_moduli(moduli: Sequence[float], odd: bool) -> float:
    """CF of a binary n-cycle with uniform marginals from edge moduli and the holonomy bit."""
    m = np.asarray(moduli, float)
    n = len(m)
    if odd:
        return float(max(0.0, (m.sum() - (n - 2)) / 2))
    return float(max(0.0, (m.sum() - 2 * m.min() - (n - 2)) / 2))


def sample_binary_edge(s: float, N: int, rng: np.random.Generator) -> np.ndarray:
    """N draws from the 2×2 table (1 + s·(±1))/4; returns counts."""
    p = np.array([1 + s, 1 - s, 1 - s, 1 + s]) / 4
    return rng.multinomial(N, p).reshape(2, 2)


# ---------------------------------------------------------------------------
# Part 2: dimension deficit for affine operation systems over F_p
# ---------------------------------------------------------------------------
def affine_matrix(a: int, b: int, p: int) -> np.ndarray:
    return np.array([[a % p, b % p], [0, 1]], dtype=np.int64)


def rank_mod_p(M: np.ndarray, p: int) -> int:
    A = np.array(M, dtype=np.int64) % p
    r, (rows, cols) = 0, A.shape
    for c in range(cols):
        piv = next((i for i in range(r, rows) if A[i, c] % p), None)
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        inv = pow(int(A[r, c]), p - 2, p)
        A[r] = (A[r] * inv) % p
        for i in range(rows):
            if i != r and A[i, c]:
                A[i] = (A[i] - A[i, c] * A[r]) % p
        r += 1
        if r == rows:
            break
    return r


def sample_affine_edges(graph: Graph, ops: Sequence[Tuple[int, int]], p: int, N: int, eps: float,
                        rng: np.random.Generator) -> List[np.ndarray]:
    """Edge (u,v) with op (a,b): x ~ U(F_p), y = a x + b w.p. 1−ε else uniform. Returns p×p counts."""
    out = []
    for (a, b) in ops:
        x = rng.integers(0, p, N)
        y = (a * x + b) % p
        flip = rng.random(N) < eps
        y[flip] = rng.integers(0, p, flip.sum())
        T = np.zeros((p, p), dtype=np.int64)
        np.add.at(T, (x, y), 1)
        out.append(T)
    return out


def support_from_counts(T: np.ndarray, tau: float) -> np.ndarray:
    """Cells whose conditional frequency P(y|x) exceeds τ."""
    rows = T.sum(1, keepdims=True).clip(min=1)
    return (T / rows) > tau


def affine_from_support(S: np.ndarray, p: int) -> Optional[Tuple[int, int]]:
    """If S is the graph of an affine bijection y = a x + b, return (a, b); else None."""
    if not (S.sum(1) == 1).all() or not (S.sum(0) == 1).all():
        return None
    f = S.argmax(1)
    b = int(f[0])
    a = int((f[1] - b) % p)
    if a == 0 or any((a * x + b) % p != f[x] for x in range(p)):
        return None
    return a, b


def loop_matrix(graph: Graph, ops: Sequence[Tuple[int, int]], p: int, row: np.ndarray) -> np.ndarray:
    """Compose edge maps around a basis loop (signed row of graph.C), homogeneous coordinates.
    Edges are traversed in the cyclic order of the loop starting from its cotree edge."""
    idx = [k for k in range(graph.E) if row[k] != 0]
    # order the loop edges into a walk
    k0 = next(k for k in idx if k in graph.cotree)
    a0, b0 = graph.edges[k0]
    walk, cur, used = [(k0, +1 if row[k0] > 0 else -1)], (b0 if row[k0] > 0 else a0), {k0}
    start = a0 if row[k0] > 0 else b0
    while cur != start:
        k = next(k for k in idx if k not in used and cur in graph.edges[k])
        used.add(k)
        u, v = graph.edges[k]
        walk.append((k, +1 if u == cur else -1))
        cur = v if u == cur else u
    M = np.eye(2, dtype=np.int64)
    for k, sgn in walk:
        a, b = ops[k]
        A = affine_matrix(a, b, p)
        if sgn < 0:
            ai = pow(a, p - 2, p)
            A = affine_matrix(ai, -ai * b, p)
        M = (A @ M) % p
    return M


def dimension_deficit(graph: Graph, ops: Sequence[Tuple[int, int]], p: int) -> Tuple[int, int]:
    """(Σ_loops rank(M_loop − I), number of affine global sections).
    Global sections: colourings c with c_v = a c_u + b on every edge (brute force)."""
    deficit = sum(rank_mod_p((loop_matrix(graph, ops, p, r) - np.eye(2, dtype=np.int64)) % p, p)
                  for r in graph.C)
    n_glob = 0
    for c in itertools.product(range(p), repeat=graph.n_vertices):
        if all((a * c[u] + b - c[v]) % p == 0 for (u, v), (a, b) in zip(graph.edges, ops)):
            n_glob += 1
    return deficit, n_glob


def estimated_deficit(graph: Graph, tables: Sequence[np.ndarray], p: int, tau: float):
    ops = [affine_from_support(support_from_counts(T, tau), p) for T in tables]
    if any(o is None for o in ops):
        return None, None, ops
    d, g = dimension_deficit(graph, ops, p)
    return d, g, ops


# ---------------------------------------------------------------------------
# Part 3: finite-field cocycle checks
# ---------------------------------------------------------------------------
def _pushforward(P: Dict[tuple, float], X: np.ndarray, p: int) -> Dict[tuple, float]:
    out: Dict[tuple, float] = {}
    for v, w in P.items():
        k = tuple((X @ np.array(v)) % p)
        out[k] = out.get(k, 0.0) + w
    return out


def _condition(P: Dict[tuple, float], X: np.ndarray, x: tuple, p: int) -> Dict[tuple, float]:
    sub = {v: w for v, w in P.items() if tuple((X @ np.array(v)) % p) == x}
    Z = sum(sub.values())
    return {v: w / Z for v, w in sub.items()}


def dim_functional(P: Dict[tuple, float], p: int) -> float:
    """log_p |supp P|  (= dimension when the support is an affine subspace)."""
    return float(np.log(sum(1 for w in P.values() if w > 0)) / np.log(p))


def s0_functional(P: Dict[tuple, float]) -> float:
    return float(sum(1 for w in P.values() if w > 0) - 1)


def cocycle_defect(F, P: Dict[tuple, float], X: np.ndarray, Y: np.ndarray, p: int, alpha: float = 1.0) -> float:
    """F[XY](P) − F[X](P) − Σ_x P(x)^α F[Y](P|X=x)  (α = 1: Shannon action, α = 0: counting)."""
    XY = np.vstack([X, Y])
    PX = _pushforward(P, X, p)
    lhs = F(_pushforward(P, XY, p))
    rhs = F(PX)
    for x, w in PX.items():
        if w > 0:
            rhs += (w ** alpha) * F(_pushforward(_condition(P, X, x, p), Y, p))
    return float(lhs - rhs)


def uniform_on_affine(basis: np.ndarray, offset: np.ndarray, p: int) -> Dict[tuple, float]:
    pts = {tuple((offset + np.array(c) @ basis) % p) for c in itertools.product(range(p), repeat=len(basis))}
    return {v: 1.0 / len(pts) for v in pts}


def random_law_with_affine_support(basis, offset, p, rng) -> Dict[tuple, float]:
    U = uniform_on_affine(basis, offset, p)
    w = rng.dirichlet(np.ones(len(U)))
    return {v: float(x) for v, x in zip(U, w)}
