"""Contradiction meter, v0: pairwise parity claims with provenance.

A claim is (unit, i, j, r): unit `unit` asserts x_i XOR x_j = r about binary propositions
x_i, x_j ("i and j have the same truth value" for r = 0, "opposite" for r = 1). A unit
is a provenance block (source, document, turn); claims may be repeated (samples), and
then each (unit, edge) has an estimated correlation s = E[(-1)^r].

The meter keeps the two failures apart (Sections "Not-applicable entries" and
"Support strata" of the write-up):

  within a unit      odd cycles in the unit's own claims            -> self-contradiction
  across units, 1    the same edge asserted with different parity  -> overlap disagreement
                     (d0 != 0; the deficit), RECORDED, not glued
  across units, 2    overlaps agree, but the union has an odd cycle -> contextual across
                     provenance (Liar cycle; H^1 != 0). Reported per cycle with the
                     logical-Bell lower bound on CF (Prop. lbell / modphase).

Blame is assigned only against a declared set of trusted units: an odd cycle whose
edges are all trusted but one names that one edge. Coherent fabrications (consistent
but false claims) are invisible by construction; that is a limit, not a bug.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from math import erf, sqrt
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Edge = Tuple[int, int]


def _norm(i: int, j: int, r: int) -> Tuple[Edge, int]:
    return ((i, j), r) if i < j else ((j, i), r)


def _phi(z: float) -> float:
    return 0.5 * (1 + erf(z / sqrt(2)))


@dataclass
class EdgeStat:
    n0: int = 0
    n1: int = 0

    @property
    def n(self) -> int:
        return self.n0 + self.n1

    @property
    def s(self) -> float:                   # estimated E[(-1)^r]
        return (self.n0 - self.n1) / self.n

    @property
    def parity(self) -> int:                # majority parity
        return 0 if self.n0 >= self.n1 else 1

    deterministic: bool = False
    eps: Optional[float] = None             # pooled flip rate, if estimated

    @property
    def p_wrong(self) -> float:
        """Probability that the majority parity is wrong.
        deterministic: 0 unless the unit contradicts itself on this edge.
        pooled eps   : posterior with a flat prior on the two parities,
                       eps^maj (1-eps)^min / (eps^maj (1-eps)^min + (1-eps)^maj eps^min).
        otherwise    : worst-case binomial (noise 1/2) if unanimous, normal approx. if not."""
        if self.deterministic:
            return 0.0 if (self.n0 == 0 or self.n1 == 0) else 0.5
        if self.eps is not None:
            e = min(max(self.eps, 1e-6), 0.5 - 1e-9)
            mj, mn = max(self.n0, self.n1), min(self.n0, self.n1)
            lr = ((e / (1 - e)) ** (mj - mn))           # odds(minority true : majority true)
            return float(lr / (1 + lr)) if mj != mn else 0.5
        if self.n1 == 0 or self.n0 == 0:
            return 0.5 ** self.n                          # all samples agree
        s = abs(self.s)
        se = sqrt(max(1 - s * s, 1e-12) / self.n)
        return 1 - _phi(s / se)


@dataclass
class CycleReport:
    edges: List[Tuple[Edge, int]]          # (edge, parity used)
    units: Set[str]
    length: int
    cf_lower: float                        # max(0, (sum|s| - (L-2))/2)
    p_all_signs_right: float               # >= 1 - sum p_wrong (union bound)
    blamed: Optional[Edge] = None


@dataclass
class MeterReport:
    self_contradictions: Dict[str, List[CycleReport]] = field(default_factory=dict)
    overlap_disagreements: List[dict] = field(default_factory=list)
    cross_unit_cycles: List[CycleReport] = field(default_factory=list)
    girth_odd: Optional[int] = None
    eps_hat: Optional[float] = None

    def flagged(self, alpha: float = 0.05, bonferroni: bool = True):
        """Cycles whose parities are all right with probability >= 1 - alpha/k, k = number of
        candidate cycles tested (Bonferroni over self and cross cycles)."""
        k = sum(len(c) for c in self.self_contradictions.values()) + len(self.cross_unit_cycles)
        thr = 1 - (alpha / max(k, 1) if bonferroni else alpha)
        selfc = {u: [c for c in cs if c.p_all_signs_right >= thr] for u, cs in self.self_contradictions.items()}
        cross = [c for c in self.cross_unit_cycles if c.p_all_signs_right >= thr]
        return selfc, cross

    def summary(self) -> str:
        return (f"self-contradictory units: {sorted(u for u, c in self.self_contradictions.items() if c)}; "
                f"overlap disagreements: {len(self.overlap_disagreements)}; "
                f"cross-unit odd cycles: {len(self.cross_unit_cycles)} (shortest {self.girth_odd})")


def estimate_flip_rate(stats: Iterable[EdgeStat], iters: int = 200) -> float:
    """Pooled flip rate: EM for eps with the true parity of each edge latent.
    E-step: w = P(majority is true); M-step: eps = expected flips / total samples."""
    st = [x for x in stats if x.n >= 1]
    if not st:
        return 0.0
    eps = 0.1
    for _ in range(iters):
        num = den = 0.0
        for x in st:
            mj, mn = max(x.n0, x.n1), min(x.n0, x.n1)
            e = min(max(eps, 1e-9), 0.5 - 1e-9)
            lr = (e / (1 - e)) ** (mj - mn)
            w = 1 / (1 + lr)                             # P(majority true)
            num += w * mn + (1 - w) * mj
            den += x.n
        new = num / den
        if abs(new - eps) < 1e-12:
            break
        eps = new
    return float(eps)


def _shortest_odd_cycles(edges: Dict[Edge, int], max_len: Optional[int] = None) -> List[List[Edge]]:
    """For every edge e = (a, b) with parity r, the shortest odd cycle through e: a BFS in
    the parity double cover of G - e from (a, 0) to (b, 1 XOR r).  The shortest odd closed
    walk is a simple cycle; results are de-duplicated and sorted by length."""
    adj = defaultdict(list)
    for (a, b), r in edges.items():
        adj[a].append((b, (a, b), r))
        adj[b].append((a, (a, b), r))
    found = {}
    for e0, r0 in edges.items():
        a, b = e0
        start, goal = (a, 0), (b, 1 ^ r0)
        prev = {start: None}
        q = deque([start])
        while q:
            node = q.popleft()
            if node == goal:
                break
            v, par = node
            for w, e, r in adj[v]:
                if e == e0:
                    continue
                nxt = (w, par ^ r)
                if nxt not in prev:
                    prev[nxt] = (node, e)
                    q.append(nxt)
        if goal not in prev:
            continue
        path, node = [e0], goal
        while prev[node] is not None:
            node, e = prev[node]
            path.append(e)
        if len(set(path)) != len(path):
            continue                                     # walk reused an edge: not simple
        if max_len and len(path) > max_len:
            continue
        found[frozenset(path)] = path
    return sorted(found.values(), key=len)


def _odd_cycles(edges: Dict[Edge, int]) -> List[List[Edge]]:
    """Fundamental cycles with odd parity sum, from a BFS spanning forest."""
    adj = defaultdict(list)
    for (a, b), r in edges.items():
        adj[a].append((b, (a, b)))
        adj[b].append((a, (a, b)))
    label, parent, depth = {}, {}, {}
    tree: Set[Edge] = set()
    for root in sorted(adj):
        if root in label:
            continue
        label[root], parent[root], depth[root] = 0, None, 0
        q = deque([root])
        while q:
            u = q.popleft()
            for w, e in adj[u]:
                if w not in label:
                    label[w] = label[u] ^ edges[e]
                    parent[w] = (u, e)
                    depth[w] = depth[u] + 1
                    tree.add(e)
                    q.append(w)
    out = []
    for e, r in edges.items():
        if e in tree:
            continue
        a, b = e
        if label[a] ^ label[b] ^ r == 0:
            continue                                   # even (flat) fundamental cycle
        path, u, v = [e], a, b
        while depth[u] > depth[v]:
            path.append(parent[u][1]); u = parent[u][0]
        while depth[v] > depth[u]:
            path.append(parent[v][1]); v = parent[v][0]
        while u != v:
            path.append(parent[u][1]); u = parent[u][0]
            path.append(parent[v][1]); v = parent[v][0]
        out.append(path)
    return out


def _cycle_report(path: List[Edge], stats: Dict[Edge, EdgeStat], owners: Dict[Edge, Set[str]],
                  trusted: Set[str]) -> CycleReport:
    L = len(path)
    mods = [abs(stats[e].s) for e in path]
    cf = max(0.0, (sum(mods) - (L - 2)) / 2)
    p_ok = max(0.0, 1 - sum(stats[e].p_wrong for e in path))
    units = set().union(*(owners[e] for e in path))
    untrusted = [e for e in path if not (owners[e] & trusted)]
    blamed = untrusted[0] if trusted and len(untrusted) == 1 else None
    return CycleReport([(e, stats[e].parity) for e in path], units, L, cf, p_ok, blamed)


def run_meter(claims: Iterable[Tuple[str, int, int, int]], trusted: Sequence[str] = (),
              z_overlap: float = 3.0, deterministic: bool = False, noise: str = "pooled",
              conf: float = 0.95, cycles: str = "shortest") -> MeterReport:
    """deterministic=True: each claim is an assertion, not a noisy sample; any parity
    mismatch between units is a disagreement and single assertions are certain.
    noise = "pooled" (EM flip rate across all claims) or "worst" (v0 behaviour).
    cycles = "shortest" (shortest odd cycle through every edge) or "basis" (fundamental)."""
    trusted = set(trusted)
    per_unit: Dict[str, Dict[Edge, EdgeStat]] = defaultdict(
        lambda: defaultdict(lambda: EdgeStat(deterministic=deterministic)))
    for unit, i, j, r in claims:
        e, r = _norm(i, j, int(r))
        st = per_unit[unit][e]
        if r:
            st.n1 += 1
        else:
            st.n0 += 1
    rep = MeterReport()
    if not deterministic and noise == "pooled":
        rep.eps_hat = estimate_flip_rate([x for st in per_unit.values() for x in st.values()])
        for st in per_unit.values():
            for x in st.values():
                x.eps = rep.eps_hat
    find = _shortest_odd_cycles if cycles == "shortest" else _odd_cycles

    # 1. within each unit
    for u, st in per_unit.items():
        par = {e: s.parity for e, s in st.items()}
        rep.self_contradictions[u] = [_cycle_report(c, st, {e: {u} for e in st}, set())
                                      for c in find(par)]

    # 2. overlaps: same edge from several units
    by_edge: Dict[Edge, Dict[str, EdgeStat]] = defaultdict(dict)
    for u, st in per_unit.items():
        for e, s in st.items():
            by_edge[e][u] = s
    glued: Dict[Edge, EdgeStat] = {}
    owners: Dict[Edge, Set[str]] = {}
    for e, us in by_edge.items():
        items = list(us.items())
        conflict = False
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                (ua, sa), (ub, sb) = items[a], items[b]
                se = sqrt(max(1 - sa.s ** 2, 1 / sa.n) / sa.n + max(1 - sb.s ** 2, 1 / sb.n) / sb.n)
                z = abs(sa.s - sb.s) / se
                if rep.eps_hat is not None:
                    sure = (1 - sa.p_wrong) * (1 - sb.p_wrong) >= conf
                else:
                    sure = deterministic or z > z_overlap
                if sa.parity != sb.parity and sure:
                    conflict = True
                    rep.overlap_disagreements.append(dict(edge=e, units=(ua, ub), s=(sa.s, sb.s), z=z))
        if not conflict:                                # d0 = 0 on this overlap: glue
            g = EdgeStat(sum(s.n0 for s in us.values()), sum(s.n1 for s in us.values()), deterministic,
                         rep.eps_hat)
            glued[e], owners[e] = g, set(us)

    # 3. across units: odd cycles of the glued graph that no single unit contains
    par = {e: s.parity for e, s in glued.items()}
    for c in find(par):
        units = set().union(*(owners[e] for e in c))
        if any(all(u in owners[e] for e in c) for u in units):
            continue                                    # already a self-contradiction
        rep.cross_unit_cycles.append(_cycle_report(c, glued, owners, trusted))
    lens = [c.length for c in rep.cross_unit_cycles] + \
           [c.length for cs in rep.self_contradictions.values() for c in cs]
    rep.girth_odd = min(lens) if lens else None
    return rep
