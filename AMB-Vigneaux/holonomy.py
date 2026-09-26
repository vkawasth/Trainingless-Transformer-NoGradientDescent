"""Layer 1 with ESTIMATED operations — the handoff's Problem 26.

Setting (holonomy paper, Thm 3): a finite connected graph Γ = (V, E), an
abelian group A, and operations g : E -> A with g_ba = -g_ab.  A global field
c : V -> A with c(b) - c(a) = g_ab exists iff the holonomy h(L) = ℓ_L · g
vanishes on a cycle basis; the obstruction is [g] ∈ H¹(Γ, A) ≅ A^{β₁}.

Here g is not known.  It is observed through a DESIGN: a list of integer
functionals a_k ∈ Z^E, each observation being  y = a_k · g + noise  in A
(a_k an edge indicator = measure one surface; a_k a signed path = send light
along a route and read what arrives).  Three questions:

1. IDENTIFIABILITY (exact).  h(L) is a function of the noiseless observation
   law iff  ℓ_L ∈ R-span{a_k},  where R is the ring acting on A:
       A = R    : R = Q      (rational span)
       A = R/Z  : R = Z      (integer span — R/Z is divisible, but a_k·g is
                               only known mod 1, so it cannot be halved)
       A = Z/n  : R = Z/n    (span mod n)
   Sufficiency: h = Σ λ_k (a_k·g).  Necessity: A is an injective module over R
   in all three cases (R/Z and Q are divisible; Z/n is self-injective), so if
   ℓ ∉ span there is a homomorphism g' : Z^E -> A killing every a_k with
   ℓ·g' ≠ 0, and the worlds g and g + g' produce identical observations.
   The criterion is ring-dependent in exactly the way γ is: observing only the
   loop traversed twice identifies h over R and Z/odd, not over R/Z or Z/even.

2. THE T0 ANALOGUE.  A design that reads each vertex through ONE route (a
   spanning-tree section) has span ∩ cycle space = 0: no loop is identifiable,
   and the fitted field reproduces the data exactly — "a fitted system supplies
   its own consistent field".  Likewise an ESTIMATOR that fits a field and sets
   ĝ = δĉ returns ĥ ≡ 0 whatever the data.  Detection needs edge-local (or
   multi-route) estimates, i.e. a second source per loop.

3. DETECTION.  With edge-local estimates ĝ_e and standard errors s_e,
   ĥ = C ĝ has covariance Σ = C diag(s²) Cᵀ, and T = ĥᵀ Σ⁻¹ ĥ ~ χ²_{β₁} under
   [g] = 0 (small-noise regime), non-central with λ = hᵀ Σ⁻¹ h otherwise.
   Calibration is also done the V6 way: parametric bootstrap from the NEAREST
   FLAT operation system g₀ (weighted projection onto ker C).  For A = Z/n with a
   symmetric channel, the mode estimator gives an exact class with probability
   ≥ 1 − |E|(|A|−1)(1 − (√p_t − √p_w)²)^m  (Chernoff), i.e. a derived sample-size
   threshold.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats

from .zlinalg import solve_integer, solve_rational

GroupSpec = Union[str, int]   # "R", "R/Z", or n for Z/n


# ---------------------------------------------------------------------------
# groups
# ---------------------------------------------------------------------------
def reduce(x, group: GroupSpec):
    """Canonical representative: R as is; R/Z in (-1/2, 1/2]; Z/n in {0..n-1}."""
    x = np.asarray(x)
    if group == "R":
        return x.astype(float)
    if group == "R/Z":
        r = np.asarray(x, float) % 1.0
        return np.where(r > 0.5, r - 1.0, r)
    n = int(group)
    return np.asarray(np.rint(x), dtype=np.int64) % n


def is_zero(x, group: GroupSpec, tol: float = 1e-9) -> np.ndarray:
    r = reduce(x, group)
    if group in ("R", "R/Z"):
        return np.abs(r) < tol
    return r == 0


# ---------------------------------------------------------------------------
# graph and cycle basis
# ---------------------------------------------------------------------------
@dataclass
class Graph:
    n_vertices: int
    edges: List[Tuple[int, int]]

    def __post_init__(self):
        self.edges = [tuple(e) for e in self.edges]
        self.E = len(self.edges)
        self.tree, self.cotree = self._spanning_tree()
        self.C = self._cycle_matrix()                 # β₁ × E, entries in {-1,0,1}
        self.beta1 = self.C.shape[0]
        assert self.beta1 == self.E - self.n_vertices + 1, "graph must be connected"

    def _spanning_tree(self):
        adj = {v: [] for v in range(self.n_vertices)}
        for k, (a, b) in enumerate(self.edges):
            adj[a].append((b, k))
            adj[b].append((a, k))
        seen, tree = {0}, []
        q = deque([0])
        self.parent = {0: None}
        while q:
            u = q.popleft()
            for w, k in adj[u]:
                if w not in seen:
                    seen.add(w)
                    tree.append(k)
                    self.parent[w] = (u, k)
                    q.append(w)
        cotree = [k for k in range(self.E) if k not in set(tree)]
        return tree, cotree

    def root_path(self, v: int) -> np.ndarray:
        """Signed edge vector of the tree path root -> v, so path·g = c(v) - c(root)."""
        vec = np.zeros(self.E, dtype=np.int64)
        while self.parent[v] is not None:
            u, k = self.parent[v]
            vec[k] += 1 if self.edges[k] == (u, v) else -1
            v = u
        return vec

    def edge_vector(self, k: int) -> np.ndarray:
        e = np.zeros(self.E, dtype=np.int64)
        e[k] = 1
        return e

    def _cycle_matrix(self) -> np.ndarray:
        rows = []
        for k in self.cotree:
            a, b = self.edges[k]
            # traverse a -> b by edge k, then back b -> a along the tree
            rows.append(self.edge_vector(k) + self.root_path(a) - self.root_path(b))
        return np.array(rows, dtype=np.int64).reshape(len(rows), self.E)

    # --- known operations ------------------------------------------------
    def holonomy(self, g, group: GroupSpec) -> np.ndarray:
        return reduce(self.C @ np.asarray(g), group)

    def field_by_propagation(self, g, group: GroupSpec, tol: float = 1e-9) -> Optional[np.ndarray]:
        """Brute-force check independent of C: integrate along the tree, then test
        every edge.  Returns the field or None."""
        g = np.asarray(g)
        c = np.zeros(self.n_vertices, dtype=float if group in ("R", "R/Z") else np.int64)
        for v in range(1, self.n_vertices):
            c[v] = self.root_path(v) @ g
        for k, (a, b) in enumerate(self.edges):
            if not is_zero(c[b] - c[a] - g[k], group, tol):
                return None
        return reduce(c, group)

    def coboundary(self, c) -> np.ndarray:
        c = np.asarray(c)
        return np.array([c[b] - c[a] for a, b in self.edges])


# ---------------------------------------------------------------------------
# identifiability of loop holonomy from a design
# ---------------------------------------------------------------------------
def loop_identifiable(design: np.ndarray, loop: np.ndarray, group: GroupSpec) -> bool:
    """Is ℓ·g determined by the noiseless observations {a_k·g}?  Exact.
    design: K × E integer matrix of observation functionals."""
    A = np.asarray(design, dtype=np.int64).reshape(-1, len(loop))
    AT = A.T.tolist()
    ell = [int(v) for v in loop]
    if group == "R":
        return solve_rational(AT, ell) is not None
    if group == "R/Z":
        return solve_integer(AT, ell) is not None
    n = int(group)
    aug = [row + [n if i == j else 0 for j in range(len(ell))] for i, row in enumerate(AT)]
    return solve_integer(aug, ell) is not None


def identifiable_loops(graph: Graph, design: np.ndarray, group: GroupSpec) -> List[bool]:
    return [loop_identifiable(design, graph.C[i], group) for i in range(graph.beta1)]


def edge_design(graph: Graph) -> np.ndarray:
    """Measure every surface on its own: the multi-source design."""
    return np.eye(graph.E, dtype=np.int64)


def tree_section_design(graph: Graph) -> np.ndarray:
    """Read each vertex once, along its tree route from the root: the
    single-source (T0) design.  Its span meets the cycle space only in 0."""
    return np.array([graph.root_path(v) for v in range(1, graph.n_vertices)], dtype=np.int64)


def route_design(graph: Graph, extra_routes: Sequence[np.ndarray]) -> np.ndarray:
    """Tree section plus additional routes (signed edge vectors) to some vertices."""
    return np.vstack([tree_section_design(graph)] + [np.asarray(r, np.int64)[None] for r in extra_routes])


# ---------------------------------------------------------------------------
# sampling and edge-local estimation
# ---------------------------------------------------------------------------
def observe_edges(g, n_obs, group: GroupSpec, noise: float, rng: np.random.Generator) -> List[np.ndarray]:
    """n_obs[e] noisy readings of each operation.
    R, R/Z: additive Gaussian of sd `noise` (wrapped for R/Z).
    Z/n   : symmetric channel — the true value w.p. 1-noise, else uniform on Z/n."""
    g = np.asarray(g)
    n_obs = np.broadcast_to(np.asarray(n_obs), g.shape)
    out = []
    for ge, m in zip(g, n_obs):
        if group in ("R", "R/Z"):
            out.append(reduce(ge + noise * rng.standard_normal(int(m)), group))
        else:
            n = int(group)
            y = np.full(int(m), int(ge) % n)
            flip = rng.random(int(m)) < noise
            y[flip] = rng.integers(0, n, flip.sum())
            out.append(y)
    return out


@dataclass
class EdgeEstimate:
    g: np.ndarray        # point estimates
    se: np.ndarray       # standard errors (continuous groups)


def estimate_edges(obs: List[np.ndarray], group: GroupSpec, se_kind: str = "delta") -> EdgeEstimate:
    if group == "R":
        g = np.array([o.mean() for o in obs])
        se = np.array([o.std(ddof=1) / np.sqrt(len(o)) for o in obs])
        return EdgeEstimate(g, se)
    if group == "R/Z":
        z1 = np.array([np.exp(2j * np.pi * o).mean() for o in obs])
        z2 = np.array([np.exp(4j * np.pi * o).mean() for o in obs])
        g = reduce(np.angle(z1) / (2 * np.pi), group)
        m = np.array([len(o) for o in obs], float)
        R1 = np.clip(np.abs(z1), 1e-6, 1.0)
        # delta-method variance of the circular mean (Fisher 1993, eq. 4.21):
        # var(θ̂) ≈ (1 − ρ₂) / (2 m ρ₁²) in rad², ρ₂ taken along the mean direction.
        # The small-noise form σ/√m is liberal once the noise wraps.
        rho2 = np.real(z2 * np.exp(-2j * np.angle(z1)))
        var = np.maximum(1 - rho2, 1e-15) / (2 * m * R1 ** 2)
        se = np.sqrt(var) / (2 * np.pi)
        if se_kind == "small":        # the naive σ/√m form, kept to show why it fails
            se = np.sqrt(-2 * np.log(np.clip(R1, 1e-12, 1 - 1e-15))) / (2 * np.pi) / np.sqrt(m)
        return EdgeEstimate(g, se)
    n = int(group)
    g = np.array([np.bincount(o, minlength=n).argmax() for o in obs])
    return EdgeEstimate(g, np.zeros(len(obs)))


def noise_sd(obs: List[np.ndarray], group: GroupSpec) -> np.ndarray:
    """Per-edge noise scale for the parametric bootstrap (wrapped-normal fit on R/Z)."""
    if group == "R":
        return np.array([o.std(ddof=1) for o in obs])
    R1 = np.array([np.clip(abs(np.exp(2j * np.pi * o).mean()), 1e-12, 1 - 1e-15) for o in obs])
    return np.sqrt(-2 * np.log(R1)) / (2 * np.pi)


def field_fit_estimate(graph: Graph, est: EdgeEstimate, group: GroupSpec) -> np.ndarray:
    """The estimator that shares a model class with a field: integrate ĝ along the
    tree to ĉ and return δĉ.  Its holonomy is identically zero (T0 control)."""
    c = np.array([graph.root_path(v) @ est.g for v in range(graph.n_vertices)])
    return reduce(graph.coboundary(c), group)


# ---------------------------------------------------------------------------
# the test
# ---------------------------------------------------------------------------
@dataclass
class HolonomyTest:
    h_hat: np.ndarray        # estimated holonomy per basis loop
    se_loop: np.ndarray      # its standard error
    T: float                 # Wald statistic ĥᵀ Σ⁻¹ ĥ
    p_chi2: float            # χ²_{β₁} p-value
    p_boot: Optional[float]  # parametric-bootstrap p-value (null = nearest flat system)
    g_flat: np.ndarray       # nearest flat operation system


def nearest_flat(graph: Graph, est: EdgeEstimate, group: GroupSpec) -> np.ndarray:
    """Weighted projection of ĝ onto ker C (flat systems): g₀ = ĝ − W Cᵀ(CWCᵀ)⁻¹ĥ."""
    C = graph.C.astype(float)
    W = np.diag(np.maximum(est.se, 1e-12) ** 2)
    h = graph.holonomy(est.g, group).astype(float)
    return est.g - W @ C.T @ np.linalg.solve(C @ W @ C.T, h)


def wald(graph: Graph, est: EdgeEstimate, group: GroupSpec):
    C = graph.C.astype(float)
    S = C @ np.diag(np.maximum(est.se, 1e-12) ** 2) @ C.T
    h = graph.holonomy(est.g, group).astype(float)
    return h, np.sqrt(np.diag(S)), float(h @ np.linalg.solve(S, h))


def holonomy_test(graph: Graph, obs: List[np.ndarray], group: GroupSpec,
                  n_boot: int = 0, rng: Optional[np.random.Generator] = None,
                  se_kind: str = "delta") -> HolonomyTest:
    """Test [g] = 0 from edge-local observations (continuous groups)."""
    assert group in ("R", "R/Z"), "use discrete_class for Z/n"
    est = estimate_edges(obs, group, se_kind)
    h, se, T = wald(graph, est, group)
    p = float(stats.chi2.sf(T, graph.beta1))
    g0 = nearest_flat(graph, est, group)
    pb = None
    if n_boot:
        rng = rng or np.random.default_rng(0)
        sd = noise_sd(obs, group)
        Tn = []
        for _ in range(n_boot):
            ob = [reduce(g0[e] + sd[e] * rng.standard_normal(len(obs[e])), group) for e in range(graph.E)]
            Tn.append(wald(graph, estimate_edges(ob, group), group)[2])
        pb = float((1 + np.sum(np.array(Tn) >= T)) / (n_boot + 1))
    return HolonomyTest(h, se, T, p, pb, g0)


def power_theory(graph: Graph, h_true: np.ndarray, se_edges: np.ndarray, alpha: float = 0.05) -> float:
    """Non-central χ² power of the Wald test at the true holonomy."""
    C = graph.C.astype(float)
    S = C @ np.diag(se_edges ** 2) @ C.T
    lam = float(h_true @ np.linalg.solve(S, h_true))
    crit = stats.chi2.ppf(1 - alpha, graph.beta1)
    return float(stats.ncx2.sf(crit, graph.beta1, lam)) if lam > 0 else alpha


def discrete_class(graph: Graph, obs: List[np.ndarray], n: int) -> np.ndarray:
    """Z/n: mode-estimated class [ĝ] as the holonomy vector."""
    return graph.holonomy(estimate_edges(obs, n).g, n)


def chernoff_class_error(graph: Graph, n: int, q: float, m: int) -> float:
    """Upper bound on P([ĝ] ≠ [g] or any edge wrong) for the symmetric channel with
    flip probability q and m readings per edge (ties counted as errors)."""
    pt, pw = 1 - q + q / n, q / n
    return float(min(1.0, graph.E * (n - 1) * (1 - (np.sqrt(pt) - np.sqrt(pw)) ** 2) ** m))


def chernoff_threshold(graph: Graph, n: int, q: float, delta: float) -> int:
    """Readings per edge that guarantee an exact class with probability ≥ 1 − δ."""
    pt, pw = 1 - q + q / n, q / n
    rate = -np.log(1 - (np.sqrt(pt) - np.sqrt(pw)) ** 2)
    return int(np.ceil(np.log(graph.E * (n - 1) / delta) / rate))


def circular_mean_var(sigma: float) -> float:
    """m · var(circular mean) in turns², wrapped normal of sd σ turns:
    (1 − ρ₂)/(2ρ₁²)/(2π)² with ρ_k = exp(−2π²k²σ²).  Grows like exp(4π²σ²)."""
    r1, r2 = np.exp(-2 * np.pi ** 2 * sigma ** 2), np.exp(-8 * np.pi ** 2 * sigma ** 2)
    return float((1 - r2) / (2 * r1 ** 2) / (2 * np.pi) ** 2)


def n_required(graph: Graph, h_true: np.ndarray, sigma: float, alpha: float = 0.05,
               power: float = 0.8, n_max: int = 10 ** 9) -> int:
    """Readings per edge for the Wald test to reach `power` against holonomy h_true
    (R/Z, wrapped-normal noise σ).  Identifiability never fails for σ < ∞ with the
    edge design; what diverges is this number, like exp(4π²σ²)."""
    v = circular_mean_var(sigma)
    lo, hi = 1, 2
    while power_theory(graph, h_true, np.full(graph.E, np.sqrt(v / hi)), alpha) < power:
        lo, hi = hi, hi * 2
        if hi > n_max:
            return n_max
    while lo < hi:
        mid = (lo + hi) // 2
        if power_theory(graph, h_true, np.full(graph.E, np.sqrt(v / mid)), alpha) >= power:
            hi = mid
        else:
            lo = mid + 1
    return lo
