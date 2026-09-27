"""Lossy operations on edges: testing the Layer-2 visibility conjecture.

Each edge (a, b) carries a doubly-stochastic channel K_ab on Z/n (row x = law
of c_b given c_a = x).  Colours are uniform at every vertex, which every
doubly-stochastic channel preserves, so the Bayesian inverse of K is exactly
Kᵀ and loops can be traversed in either direction.  Group-valued operations
are the special case K = T_g (a shift permutation).

Three candidate invariants of route dependence are compared.

  Layer 2   per-edge Shannon functionals of the context law (1/n)K_ab, and
            the loop-composite information I(X; M_L X).
  Layer 1   the loop operator M_L = Π K (Kᵀ backwards) and its SPECTRUM.
            M_L depends on the base point only through cyclic rotation
            (AB vs BA), and AB and BA share their nonzero spectrum, so the
            spectrum is a base-point-free holonomy invariant for any channels,
            lossy or not.  It lives in a monoid, not a group, so it is not a
            cohomology class.
  Layer 3   the edge-context empirical model e_K (contexts = edges, law
            (1/n)K_ab): its contextual fraction CF and AMB obstruction γ.

Results (tests/test_channels.py, examples/channels.py):

 (1) Per-edge Shannon functionals never see which permutation an edge applies:
     (1/n)·N·T_g has the row entropies of N.  This holds for lossy N as well,
     so "lossy" does not make per-context information see holonomy.
 (2) Covariant noise, K = (1−ε)T_g + ε·J/n (commutes with shifts):
       - loop-composite information is also independent of g;
       - the CHARACTER-LABELLED eigenvalue separates cleanly: on χ_k it is
         (1−ε)^{|L|} · exp(2πi k h / n), so the modulus carries the loss and the
         phase carries the holonomy exactly, for every ε < 1.  The unlabelled
         spectrum (a multiset) only sees the order of h: on Z/5 every h ≠ 0 has
         the same multiset.  For general covariant noise μ the value is
         Π μ̂_e(χ)·χ(h): the phase is χ(h) only if μ̂(χ) is real (symmetric noise);
         asymmetric noise adds its own phase;
       - CF(e_K) = max(0, 1 − |L|ε/2) on a single cycle, independent of n and h
         (dual certificate + explicit packing; see cf_noisy_cycle).  On any
         graph, CF = max(0, 1 − ℓ_min ε/2) with ℓ_min the shortest simple cycle
         of non-zero holonomy (≥ proved; = measured, see cf_noisy_graph);
       - all of the above holds verbatim for A = ⊕ Z/m_j (tested on Z/2², Z/2×Z/4, Z/3²);
       - γ = 0 for every ε > 0 (full support).
     Robustness ordering: γ (dies at ε = 0⁺) < CF (dies at 2/|L|) < spectrum
     (survives to ε → 1).
 (3) Non-covariant noise (N a Birkhoff mixture of arbitrary permutations):
     noise and holonomy entangle.  A FLAT system already has CF > 0 (the noise
     permutations themselves have non-trivial loops), the spectral phase is no
     longer exactly h, and loop-composite information varies with g.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .holonomy import Graph
from .outcome import contextual_fraction
from .scenario import EmpiricalModel, Scenario


# ---------------------------------------------------------------- channels
def shift(n: int, g: int) -> np.ndarray:
    """Permutation matrix of x ↦ x + g (row x has its 1 in column x+g)."""
    return np.roll(np.eye(n), int(g) % n, axis=1)


def noisy_shift(n: int, g: int, eps: float) -> np.ndarray:
    """Covariant lossy operation: (1−ε)·T_g + ε·uniform."""
    return (1 - eps) * shift(n, g) + eps / n


def birkhoff_noise(n: int, r: int, rng: np.random.Generator) -> np.ndarray:
    """Random doubly-stochastic matrix: a Dirichlet mixture of r permutations."""
    w = rng.dirichlet(np.ones(r))
    return sum(wi * np.eye(n)[rng.permutation(n)] for wi in w)


# ---------------------------------------------------------------- loops
def loop_walk(graph: Graph, i: int) -> List[int]:
    """Closed vertex walk a → b → … → a for basis loop i (cotree edge a→b, then
    the tree path back)."""
    k = graph.cotree[i]
    a, b = graph.edges[k]

    def up(v):
        out = [v]
        while graph.parent[v] is not None:
            v = graph.parent[v][0]
            out.append(v)
        return out

    pb, pa = up(b), up(a)
    lca = next(v for v in pb if v in pa)
    return [a, b] + pb[1:pb.index(lca) + 1] + list(reversed(pa[:pa.index(lca)]))


def edge_operator(graph: Graph, K: Sequence[np.ndarray], u: int, v: int) -> np.ndarray:
    for k, (a, b) in enumerate(graph.edges):
        if (a, b) == (u, v):
            return K[k]
        if (a, b) == (v, u):
            return K[k].T               # Bayesian inverse under uniform colours
    raise KeyError((u, v))


def loop_operator(graph: Graph, K: Sequence[np.ndarray], walk: Sequence[int]) -> np.ndarray:
    M = np.eye(K[0].shape[0])
    for u, v in zip(walk[:-1], walk[1:]):
        M = M @ edge_operator(graph, K, u, v)
    return M


def loop_spectrum(M: np.ndarray) -> np.ndarray:
    lam = np.linalg.eigvals(M)
    return lam[np.lexsort((np.angle(lam), -np.round(np.abs(lam), 12)))]


def character_eigenvalue(M: np.ndarray, k: int = 1) -> complex:
    """⟨χ_k, M χ_k⟩/n for the Fourier character χ_k(x) = e^{2πikx/n}.  For a
    shift-covariant M this IS the eigenvalue on χ_k."""
    n = M.shape[0]
    w = np.exp(2j * np.pi * k * np.arange(n) / n)
    return complex(w.conj() @ M @ w / n)


def holonomy_from_spectrum(M: np.ndarray) -> float:
    """Read h from the phase of the k = 1 character eigenvalue (in units of Z/n).
    Exact for covariant channels whenever that eigenvalue is nonzero."""
    n = M.shape[0]
    return float(np.angle(character_eigenvalue(M, 1)) * n / (2 * np.pi)) % n


# ---------------------------------------------------------------- Layer 2
def _H(p):
    p = np.asarray(p, float)
    p = p[p > 1e-15]
    return float(-(p * np.log2(p)).sum())


def channel_information(K: np.ndarray) -> float:
    """I(X; Y) in bits for X uniform, Y ~ K(X, ·)."""
    return float(np.log2(K.shape[0]) - np.mean([_H(r) for r in K]))


# ---------------------------------------------------------------- Layer 3
def edge_family(graph: Graph, K: Sequence[np.ndarray]) -> EmpiricalModel:
    """Contexts = edges, law (1/n)·K_ab.  No-signalling holds automatically
    (doubly stochastic ⇒ uniform marginals)."""
    n = K[0].shape[0]
    outs = {f"c{v}": tuple(range(n)) for v in range(graph.n_vertices)}
    ctx = tuple((f"c{a}", f"c{b}") for a, b in graph.edges)
    return EmpiricalModel(Scenario(outs, ctx), {C: (Kk / n).ravel() for C, Kk in zip(ctx, K)})


def cf_noisy_cycle(L: int, eps: float) -> float:
    """Closed form for a single cycle of noisy shifts with holonomy h ≠ 0.

    Upper bound on the non-contextual fraction (dual certificate): weight each
    edge cell by its discrepancy d = y − x − g,  w(0) = 0, w(−h) = 1, w = 1/2
    otherwise.  Every global colouring has Σ d ≡ −h ≠ 0 around the loop, so its
    weights sum to ≥ 1; the certificate's value is L·(ε/n)(1 + (n−2)/2) = Lε/2.
    Achievability: colourings with one violation of discrepancy −h, or two
    violations (a, −h−a), spread uniformly over translations, exhaust the
    off-graph mass and fit the on-graph capacity (tight exactly at ε = 2/L).
    Hence CF = max(0, 1 − Lε/2)."""
    return max(0.0, 1.0 - L * eps / 2.0)


def dual_certificate(n: int, h: int) -> np.ndarray:
    """w(d) for d ∈ Z/n used in cf_noisy_cycle."""
    w = np.full(n, 0.5)
    w[0] = 0.0
    w[(-h) % n] = 1.0
    return w


def cycle(L: int) -> Graph:
    return Graph(L, [(i, (i + 1) % L) for i in range(L)])


def cf_of(graph: Graph, K: Sequence[np.ndarray]) -> float:
    return contextual_fraction(edge_family(graph, K)).value


# ---------------------------------------------------------------- graphs with several loops
def simple_cycles(graph: Graph):
    """All simple cycles as signed edge vectors (enumerated through the cycle
    space; fine for β₁ ≲ 10)."""
    import itertools
    out = []
    for co in itertools.product((-1, 0, 1), repeat=graph.beta1):
        v = np.array(co) @ graph.C
        if not v.any() or np.abs(v).max() > 1:
            continue
        deg = {}
        for k in np.nonzero(v)[0]:
            for x in graph.edges[k]:
                deg[x] = deg.get(x, 0) + 1
        if all(d == 2 for d in deg.values()):
            out.append(v)
    return out


def holonomy_girth(graph: Graph, g, n: int) -> float:
    """ℓ_min: length of the shortest simple cycle with non-zero holonomy (inf if flat)."""
    ls = [int(np.abs(v).sum()) for v in simple_cycles(graph) if int(v @ np.asarray(g)) % n]
    return min(ls) if ls else float("inf")


def cf_noisy_graph(graph: Graph, g, n: int, eps: float) -> float:
    """CF = max(0, 1 − ℓ_min·ε/2) for noisy shifts on any graph.
    Proved: CF ≥ this (restrict the cycle certificate to the shortest non-flat
    cycle).  Measured (not proved): equality, in 44 configurations covering
    shared edges, shared vertices, disjoint non-flat loops and ε up to 0.8."""
    lm = holonomy_girth(graph, g, n)
    return 0.0 if lm == float("inf") else max(0.0, 1.0 - lm * eps / 2.0)


# ---------------------------------------------------------------- finite abelian groups
def group_elements(orders):
    import itertools
    return list(itertools.product(*[range(m) for m in orders]))


def translation(orders, g) -> np.ndarray:
    """Permutation matrix of x ↦ x + g on A = ⊕ Z/m_j (elements in lexicographic order)."""
    els = group_elements(orders)
    idx = {e: i for i, e in enumerate(els)}
    P = np.zeros((len(els), len(els)))
    for i, e in enumerate(els):
        P[i, idx[tuple((a + b) % m for a, b, m in zip(e, g, orders))]] = 1
    return P


def convolution(orders, mu: np.ndarray) -> np.ndarray:
    """Covariant channel y = x + ξ, ξ ~ μ (μ indexed like group_elements)."""
    els = group_elements(orders)
    return sum(mu[k] * translation(orders, e) for k, e in enumerate(els))


def character(orders, k) -> np.ndarray:
    """χ_k(x) = exp(2πi Σ_j k_j x_j / m_j) evaluated on every element."""
    return np.array([np.exp(2j * np.pi * sum(kj * xj / m for kj, xj, m in zip(k, x, orders)))
                     for x in group_elements(orders)])


def character_value(orders, k, M: np.ndarray) -> complex:
    """⟨χ_k, M χ_k⟩/|A|: the eigenvalue of a covariant M on χ_k.  For a loop of
    convolutions then translations it equals Π_e μ̂_e(χ_k)^{(*)} · χ_k(h)
    (conjugate for edges traversed backwards), so the phase is χ_k(h) exactly
    when every μ̂_e(χ_k) is real and non-zero — e.g. symmetric noise."""
    w = character(orders, k)
    return complex(w.conj() @ M @ w / len(w))


# ---------------------------------------------------------------- closed forms (Layer 2)
def noisy_shift_information(n: int, eps: float) -> float:
    """I(c_a; c_b) in bits for K = (1−ε)T_g + ε·J/n and uniform input:
    log₂n + (1−ε+ε/n)log₂(1−ε+ε/n) + (n−1)(ε/n)log₂(ε/n).  Independent of g.
    (The row puts 1−ε+ε/n on one point and ε/n on each of the other n−1; the
    two-point form log₂n + (1−ε)log₂(1−ε) + ε log₂(ε/n) is wrong.)"""
    p0, p1 = 1 - eps + eps / n, eps / n
    val = np.log2(n)
    if p0 > 0:
        val += p0 * np.log2(p0)
    if p1 > 0:
        val += (n - 1) * p1 * np.log2(p1)
    return float(val)


def loop_information(n: int, eps: float, L: int) -> float:
    """Loop-composite information: the same formula with ε_L = 1 − (1−ε)^L."""
    return noisy_shift_information(n, 1 - (1 - eps) ** L)
