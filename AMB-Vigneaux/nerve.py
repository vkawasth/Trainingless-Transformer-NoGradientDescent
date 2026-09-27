"""Decorated KL-Čech nerve: orientation and holonomy carried on a Bregman nerve.

Points are TOKENS.  Each token carries a distribution over a small state set
(Z/3: T, F, U;  Z/5: T, F, could-T, could-F, U — the order is the user's) and a
TIME (its position).  The construction:

 1. KL-Čech filtration.  Radius of a simplex = min_c max_i D_KL(p_i ‖ c) (the
    smallest enclosing ball {p : D(p‖c) ≤ r}; KL is convex in its first
    argument, so balls and their intersections are convex and the nerve lemma
    applies: the nerve has the homotopy type of the union of balls).
 2. DECORATION.  An edge {i, j} is oriented by time (A before B).  It carries
    the group element a_ij ∈ G that best transports i's distribution onto j's:
    a_ij = argmin_g  radius(g_* p_i, p_j),  and its radius is that minimum
    (so the base filtration lives on the quotient by G).  Reading the edge
    backwards applies a_ij⁻¹.
 3. TRIANGLES are filled with the lift fixed by the edge labels,
    {p_k, (a_jk)_* p_j, (a_jk a_ij)_* p_i} for i < j < k in time, and carry the
    boundary holonomy a_ik⁻¹ a_jk a_ij.  A triangle whose boundary holonomy is
    not the identity is a NON-FLAT EVENT.
 4. Each H₁ bar is labelled by the holonomy of its birth cycle, read from the
    cycle's earliest token, leaving along the edge to the later-time neighbour
    (reversing the reading inverts the label).

What is invariant (and what is not), for G = Aff(Z/p) = {x ↦ ux + t}:
  * The loop label depends on the base point only by conjugation, so the
    CONJUGACY CLASS is intrinsic.  Classes of Aff(Z/p): identity; all
    non-trivial translations (one class!); and, for each u ≠ 1, the maps
    x ↦ ux + t (one class per u).  Hence the intrinsic content is the
    multiplier u, plus "translation or not" when u = 1.
  * The translation's value t (the Z/p phase) and its sign (orientation) are
    readable only with FIXED state labels and a fixed reading convention: a
    change of frame by x ↦ −x sends t to −t.
  * Over homology the label passes through G^ab = (Z/p)^× (translations are
    the commutator subgroup for odd p), so only u is a function of the H₁
    class; the full element is a property of the birth cycle.
  * With every triangle flat, holonomy factors through π₁ of the complex.  For
    an ABELIAN group and INTEGER coefficients this means a cycle with a
    non-identity label cannot become a boundary of flat triangles.  With Z/2
    persistence or a non-abelian group this fails (a Z/2 boundary need not be
    null-homotopic), and measured bars do die at flat triangles.
  * IDENTIFIABILITY.  An element g fixes a face of the simplex of codimension
    n − #cycles(g).  Codimension-1 fixed sets are walls: for Z/3 the affine
    reflections are transpositions, the quotient is a simply connected chamber,
    and every planted holonomy is absorbed into a flat labelling.  All
    non-identity elements of Aff(Z/5) have codimension ≥ 2.  Near any fixed set
    the best alignment is not identified; each edge records its margin and a
    bar whose cycle uses an ambiguous edge is flagged.
"""
from __future__ import annotations

import itertools
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

STATE_NAMES = {3: ["T", "F", "U"], 5: ["T", "F", "could-T", "could-F", "U"]}

Perm = Tuple[int, ...]


# ---------------------------------------------------------------- groups
def compose(a: Perm, b: Perm) -> Perm:
    """a ∘ b : first b, then a."""
    return tuple(a[b[i]] for i in range(len(b)))


def inverse(a: Perm) -> Perm:
    inv = [0] * len(a)
    for i, ai in enumerate(a):
        inv[ai] = i
    return tuple(inv)


def identity(p: int) -> Perm:
    return tuple(range(p))


def affine(p: int, u: int, t: int) -> Perm:
    return tuple((u * x + t) % p for x in range(p))


def affine_group(p: int) -> List[Perm]:
    return [affine(p, u, t) for u in range(1, p) for t in range(p)]


def affine_params(a: Perm, p: int) -> Optional[Tuple[int, int]]:
    t = a[0]
    u = (a[1] - t) % p
    return (u, t) if a == affine(p, u, t) else None


def conjugacy_classes(G: Sequence[Perm]) -> Dict[Perm, int]:
    cls, k = {}, 0
    for g in G:
        if g in cls:
            continue
        for h in G:
            cls[compose(compose(h, g), inverse(h))] = k
        k += 1
    return cls


def order(a: Perm) -> int:
    e, x, k = identity(len(a)), a, 1
    while x != e:
        x, k = compose(a, x), k + 1
    return k


def describe(a: Perm, p: int, names: Optional[List[str]] = None) -> str:
    names = names or STATE_NAMES.get(p, [str(i) for i in range(p)])
    if a == identity(p):
        return "identity (flat)"
    ut = affine_params(a, p)
    if ut is None:
        return "non-affine permutation " + str(a)
    u, t = ut
    if u == 1:
        return f"translation x↦x+{t}  (phase ω^{t}, ω = e^(2πi/{p}); read backwards: ω^{(-t) % p})"
    xs = (t * pow((1 - u) % p, -1, p)) % p
    kind = "reflection x↦−x+t (reverses the state cycle)" if u == p - 1 else f"multiplier u={u} (order {order(a)})"
    return f"{kind}, x↦{u}x+{t}, fixes state {names[xs]}"


def push(a: Perm, dist: np.ndarray) -> np.ndarray:
    """(a_* p)(a(x)) = p(x)."""
    out = np.empty_like(dist)
    out[..., list(a)] = dist
    return out


# ---------------------------------------------------------------- KL-Čech radii
def _kl(P, c):
    return (P * (np.log(P) - np.log(c)[..., None, :])).sum(-1)


def batch_radius(points: np.ndarray, iters: int = 1500) -> np.ndarray:
    """min_c max_i D(p_i ‖ c) for a batch of point sets, shape (B, m, p).
    Badoiu–Clarkson iteration toward the farthest point (the right-sided KL
    centroid is the arithmetic mean, so convex steps stay on the right side)."""
    P = np.clip(points, 1e-300, None)
    c = P.mean(1)
    best = _kl(P, c).max(1)
    idx = np.arange(len(P))
    for i in range(1, iters + 1):
        d = _kl(P, c)
        j = d.argmax(1)
        c = (1 - 1 / (i + 1)) * c + (1 / (i + 1)) * P[idx, j]
        c /= c.sum(1, keepdims=True)
        best = np.minimum(best, _kl(P, c).max(1))
    return best


def pair_radius(P: np.ndarray, Q: np.ndarray, iters: int = 60) -> np.ndarray:
    """Exact min_c max(D(p‖c), D(q‖c)) for batches of pairs (rows).  The optimum
    lies on the segment c = λp + (1−λ)q, where the two divergences are equal;
    found by bisection (checked against direct minimisation to 1e-15)."""
    P, Q = np.clip(P, 1e-300, None), np.clip(Q, 1e-300, None)
    lo, hi = np.zeros(len(P)), np.ones(len(P))
    for _ in range(iters):
        m = (lo + hi) / 2
        c = m[:, None] * P + (1 - m[:, None]) * Q
        f = (P * np.log(P / c)).sum(1) - (Q * np.log(Q / c)).sum(1)
        lo, hi = np.where(f > 0, m, lo), np.where(f > 0, hi, m)
    m = (lo + hi) / 2
    c = m[:, None] * P + (1 - m[:, None]) * Q
    return np.maximum((P * np.log(P / c)).sum(1), (Q * np.log(Q / c)).sum(1))


# ---------------------------------------------------------------- the decorated filtration
@dataclass
class Bar:
    dim: int
    birth: float
    death: float
    cycle: Optional[List[int]] = None       # token indices, closed walk, canonical reading
    label: Optional[Perm] = None
    label_desc: str = ""
    killed_by_nonflat: bool = False
    ambiguous: bool = False           # cycle uses an edge whose alignment is not identified


@dataclass
class DecoratedNerve:
    dists: np.ndarray                  # (N, p) token distributions
    times: np.ndarray                  # (N,) token positions
    group: List[Perm]
    names: Optional[List[str]] = None
    iters: int = 1500
    margin_tol: float = 1e-3          # edges whose best and second-best alignments are closer are AMBIGUOUS
    edges: Dict[Tuple[int, int], dict] = field(default_factory=dict)
    triangles: Dict[Tuple[int, int, int], dict] = field(default_factory=dict)
    bars: List[Bar] = field(default_factory=list)

    def __post_init__(self):
        self.p = self.dists.shape[1]
        self.N = len(self.dists)
        self.order = sorted(range(self.N), key=lambda i: self.times[i])   # time order
        self.rank = {i: r for r, i in enumerate(self.order)}
        self._edges()
        self._triangles()
        self._persistence()

    # ---- decoration --------------------------------------------------------
    def _edges(self):
        G = self.group
        pairs = [(i, j) for i, j in itertools.combinations(range(self.N), 2)]
        pairs = [(i, j) if self.rank[i] < self.rank[j] else (j, i) for i, j in pairs]
        A = np.array([push(g, self.dists[i]) for (i, j) in pairs for g in G])
        B = np.array([self.dists[j] for (i, j) in pairs for g in G])
        rad = pair_radius(A, B).reshape(len(pairs), len(G))
        for k, (i, j) in enumerate(pairs):
            o = np.argsort(rad[k])
            self.edges[(i, j)] = dict(label=G[o[0]], r=float(rad[k, o[0]]),
                                      margin=float(rad[k, o[1]] - rad[k, o[0]]) if len(G) > 1 else np.inf)

    def transport(self, x: int, y: int) -> Perm:
        """Group element carrying x's frame to y's along edge {x, y}."""
        if (x, y) in self.edges:
            return self.edges[(x, y)]["label"]
        return inverse(self.edges[(y, x)]["label"])

    def _triangles(self):
        trip = [tuple(sorted(t, key=lambda i: self.rank[i])) for t in itertools.combinations(range(self.N), 3)]
        lifts, meta = [], []
        for (i, j, k) in trip:
            aij, ajk, aik = self.transport(i, j), self.transport(j, k), self.transport(i, k)
            lifts.append([self.dists[k], push(ajk, self.dists[j]), push(compose(ajk, aij), self.dists[i])])
            hol = compose(inverse(aik), compose(ajk, aij))
            meta.append(hol)
        rad = batch_radius(np.array(lifts), self.iters)
        for t, hol, r in zip(trip, meta, rad):
            r_edges = max(self.edges[e]["r"] for e in [(t[0], t[1]), (t[1], t[2]), (t[0], t[2])])
            self.triangles[t] = dict(r=float(max(r, r_edges)), hol=hol, flat=hol == identity(self.p))

    @property
    def first_nonflat(self) -> float:
        rs = [v["r"] for v in self.triangles.values() if not v["flat"]]
        return min(rs) if rs else np.inf

    # ---- persistence over Z/2 with labelled H1 bars ----------------------------
    def _persistence(self):
        simp = [(0.0, 0, (i,)) for i in range(self.N)]
        simp += [(v["r"], 1, e) for e, v in self.edges.items()]
        simp += [(v["r"], 2, t) for t, v in self.triangles.items()]
        simp.sort(key=lambda s: (s[0], s[1], s[2]))
        index = {s[2]: n for n, s in enumerate(simp)}
        cols, pivot_of = [], {}
        for n, (r, d, s) in enumerate(simp):
            if d == 0:
                col = set()
            elif d == 1:
                col = {index[(s[0],)], index[(s[1],)]}
            else:
                col = set()
                for f in itertools.combinations(s, 2):
                    f = f if self.rank[f[0]] < self.rank[f[1]] else (f[1], f[0])
                    col.add(index[f])
            while col:
                l = max(col)
                if l not in pivot_of:
                    break
                col ^= cols[pivot_of[l]]
            cols.append(col)
            if col:
                pivot_of[max(col)] = n
        # simple, explicit pass: rebuild forest in filtration order with union-find
        parent = list(range(self.N))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        forest = defaultdict(list)
        births = {}
        for n, (r, d, s) in enumerate(simp):
            if d != 1:
                continue
            i, j = s
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
                forest[i].append(j)
                forest[j].append(i)
            else:
                births[n] = self._cycle(i, j, forest)
        for l, n in pivot_of.items():
            if simp[l][1] == 1:                               # H1 bar born at edge l, dies at triangle n
                cyc = births.get(l)
                b = Bar(1, simp[l][0], simp[n][0], cyc)
                self._label(b)
                b.killed_by_nonflat = not self.triangles[simp[n][2]]["flat"]
                self.bars.append(b)
            elif simp[l][1] == 0 and simp[l][0] < simp[n][0]:
                self.bars.append(Bar(0, simp[l][0], simp[n][0]))
        for n, (r, d, s) in enumerate(simp):
            if d == 1 and n in births and n not in pivot_of and not cols[n]:
                b = Bar(1, r, np.inf, births[n])
                self._label(b)
                self.bars.append(b)
        self.bars.sort(key=lambda b: (b.dim, b.birth))

    def _cycle(self, i, j, forest) -> List[int]:
        prev, q = {j: None}, deque([j])
        while q:
            x = q.popleft()
            if x == i:
                break
            for y in forest[x]:
                if y not in prev:
                    prev[y] = x
                    q.append(y)
        path, x = [], i
        while x is not None:
            path.append(x)
            x = prev[x]
        cyc = path                                # i ... j, then edge j→i closes it
        # canonical reading: start at the earliest token, leave toward the later-time neighbour
        s = min(range(len(cyc)), key=lambda k: self.rank[cyc[k]])
        cyc = cyc[s:] + cyc[:s]
        if len(cyc) > 2 and self.rank[cyc[-1]] > self.rank[cyc[1]]:
            cyc = [cyc[0]] + cyc[1:][::-1]
        return cyc + [cyc[0]]

    def loop_holonomy(self, walk: Sequence[int]) -> Perm:
        h = identity(self.p)
        for x, y in zip(walk[:-1], walk[1:]):
            h = compose(self.transport(x, y), h)
        return h

    def _label(self, b: Bar):
        if b.cycle is None:
            return
        b.label = self.loop_holonomy(b.cycle)
        b.label_desc = describe(b.label, self.p, self.names)
        b.ambiguous = any(self.edge(x, y)["margin"] < self.margin_tol for x, y in zip(b.cycle[:-1], b.cycle[1:]))

    def edge(self, x: int, y: int) -> dict:
        return self.edges[(x, y)] if (x, y) in self.edges else self.edges[(y, x)]

    def h1_bars(self, min_persistence: float = 0.0) -> List[Bar]:
        return [b for b in self.bars if b.dim == 1 and b.death - b.birth > min_persistence]


# ---------------------------------------------------------------- synthetic tokens
def generic_distribution(p: int, rng: np.random.Generator, G: Sequence[Perm], min_gap: float = 0.15) -> np.ndarray:
    """A distribution with trivial stabiliser in G, far from its own images."""
    for _ in range(10000):
        q = rng.dirichlet(np.full(p, 2.0))
        imgs = [push(g, q) for g in G if g != identity(p)]
        if min(np.abs(q - im).sum() for im in imgs) > min_gap:
            return q
    raise RuntimeError("no generic distribution found")


def ring_tokens(p: int, h: Perm, n_tokens: int, rng: np.random.Generator, G: Sequence[Perm],
                t0: int = 0, beta: float = 0.8, concentration: float = 0.0,
                q: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Tokens whose context distribution drifts, as time advances, from q to h_* q
    along a curved path in natural parameters.  In the quotient by G this is a
    closed loop with holonomy h.  concentration > 0 adds Dirichlet noise."""
    q = generic_distribution(p, rng, G) if q is None else q
    hq = push(h, q)
    w = rng.standard_normal(p)
    w -= w.mean()
    out = []
    for k in range(n_tokens):
        s = k / n_tokens
        th = (1 - s) * np.log(q) + s * np.log(hq) + beta * np.sin(np.pi * s) * w
        x = np.exp(th - th.max())
        x /= x.sum()
        if concentration > 0:
            x = rng.dirichlet(concentration * x + 1e-3)
            x = np.clip(x, 1e-6, None)
            x /= x.sum()
        out.append(x)
    return np.array(out), np.arange(t0, t0 + n_tokens)


# ---------------------------------------------------------------- ring readout (edge level)
def time_loop_label(dists: np.ndarray, times: np.ndarray, G: Sequence[Perm]) -> Tuple[Perm, float]:
    """Holonomy of the loop that visits tokens in time order and closes from the
    last token back to the first, read in the canonical direction (leave the
    earliest token toward its later-time neighbour).  Edges carry their best
    alignment; returns (label, smallest alignment margin on the loop)."""
    order = list(np.argsort(times))
    pairs = [(order[k], order[k + 1]) for k in range(len(order) - 1)] + [(order[0], order[-1])]
    A = np.array([push(g, dists[i]) for (i, j) in pairs for g in G])
    B = np.array([dists[j] for (i, j) in pairs for g in G])
    rad = pair_radius(A, B).reshape(len(pairs), len(G))
    labs = [G[int(np.argmin(r))] for r in rad]
    margin = float(min(np.sort(r)[1] - np.sort(r)[0] for r in rad)) if len(G) > 1 else np.inf
    hol = labs[-1]
    for l in reversed(labs[:-1]):
        hol = compose(inverse(l), hol)
    return hol, margin


def fixed_codimension(g: Perm) -> int:
    """Codimension (in the simplex) of the face fixed by the permutation g:
    n − number of cycles.  Codimension 1 means g fixes a wall."""
    seen, cycles = set(), 0
    for i in range(len(g)):
        if i not in seen:
            cycles += 1
            while i not in seen:
                seen.add(i)
                i = g[i]
    return len(g) - cycles
