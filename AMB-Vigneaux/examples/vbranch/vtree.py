"""Variable right-branching hierarchy grammar (fixed depth, unknown tree shape).

Every production at level l is   b  ->  c_0  c_1 ... c_k       k in {0, ..., KMAX}
  c_0 is the single LEFT child, c_1..c_k are the RIGHT children (k may be 0: a unary node).
Children of a level-l symbol are level-(l-1) symbols; level-0 symbols are leaves (tokens).
All leaves sit at depth L, so the sentence length is the product of the branching along the
tree and varies from sentence to sentence; the learner does not see the tree.

Exact inside by SPAN MATRICES.  For a sentence x of length n let
    B_0[a][i, j] = 1  if j = i + 1 and x_i = a           ((n+1) x (n+1), strictly upper)
    B_l[b]       = sum_r  p_r  B_{l-1}[c_0] B_{l-1}[c_1] ... B_{l-1}[c_k]
because concatenating contiguous spans is a matrix product.  Then
    P(x) = sum_b  pi_b  B_L[b][0, n].
Expected rule counts for EM are p_r * d log P / d p_r (autodiff; JAX).
"""
from __future__ import annotations

import itertools, json
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

Rule = Tuple[int, Tuple[int, ...]]            # (lhs, children)


@dataclass
class VGrammar:
    L: int                                     # depth
    V: int                                     # symbols per internal level
    A: int                                     # leaf alphabet
    rules: Dict[int, List[Tuple[int, Tuple[int, ...], float]]] = field(default_factory=dict)
    # rules[l] = list of (lhs, children, prob); children are level-(l-1) symbols (tokens at l = 1)
    prior: np.ndarray = None

    def card(self, l: int) -> int:
        return self.A if l == 0 else self.V

    def to_json(self):
        return dict(L=self.L, V=self.V, A=self.A, prior=self.prior.tolist(),
                    rules={str(l): [[b, list(c), p] for b, c, p in rs] for l, rs in self.rules.items()})

    @staticmethod
    def from_json(d):
        g = VGrammar(d["L"], d["V"], d["A"])
        g.rules = {int(l): [(b, tuple(c), p) for b, c, p in rs] for l, rs in d["rules"].items()}
        g.prior = np.array(d["prior"])
        return g


def random_grammar(L=3, V=8, A=12, m=3, kmax=5, k_probs=None, seed=0) -> VGrammar:
    """m productions per symbol; each production draws its own right arity k in 0..kmax and
    distinct child tuples.  Productions of a symbol are equiprobable."""
    rng = np.random.default_rng(seed)
    k_probs = np.full(kmax + 1, 1 / (kmax + 1)) if k_probs is None else np.asarray(k_probs)
    g = VGrammar(L, V, A, {}, np.full(V, 1 / V))
    for l in range(1, L + 1):
        nc = A if l == 1 else V
        rs, used = [], set()
        for b in range(V):
            for _ in range(m):
                while True:
                    k = int(rng.choice(kmax + 1, p=k_probs))
                    ch = tuple(int(c) for c in rng.integers(0, nc, 1 + k))
                    if ch not in used:                 # no two symbols share a production
                        used.add(ch); break
                rs.append((b, ch, 1.0 / m))
        g.rules[l] = rs
    return g


def sample(g: VGrammar, n: int, rng: np.random.Generator, return_trees=False):
    by = {l: {} for l in g.rules}
    for l, rs in g.rules.items():
        for b, c, p in rs:
            by[l].setdefault(b, []).append((c, p))
    out, trees = [], []
    for _ in range(n):
        def expand(l, b):
            opts = by[l][b]
            i = rng.choice(len(opts), p=np.array([p for _, p in opts]) / sum(p for _, p in opts))
            ch = opts[i][0]
            if l == 1:
                return list(ch), (b, [("tok", c) for c in ch])
            parts = [expand(l - 1, c) for c in ch]
            return [t for s, _ in parts for t in s], (b, [t for _, t in parts])
        root = int(rng.choice(g.V, p=g.prior))
        s, t = expand(g.L, root)
        out.append(s); trees.append(t)
    return (out, trees) if return_trees else out


# ---------------------------------------------------------------------------- exact inside (numpy)
def inside_matrices(g: VGrammar, x: Sequence[int]):
    n = len(x)
    B = np.zeros((g.A, n + 1, n + 1))
    for i, a in enumerate(x):
        B[a, i, i + 1] = 1.0
    mats = {0: B}
    for l in range(1, g.L + 1):
        Bl = np.zeros((g.V, n + 1, n + 1))
        for b, ch, p in g.rules[l]:
            M = mats[l - 1][ch[0]]
            for c in ch[1:]:
                M = M @ mats[l - 1][c]
            Bl[b] += p * M
        mats[l] = Bl
    return mats


def loglik(g: VGrammar, x: Sequence[int]) -> float:
    Bl = inside_matrices(g, x)[g.L]
    return float(np.log(g.prior @ Bl[:, 0, len(x)]))


# ---------------------------------------------------------------------------- brute force (tests)
def brute_force_prob(g: VGrammar, x: Sequence[int]) -> float:
    """Sum over all derivations whose yield is x (exponential; tiny grammars only)."""
    x = tuple(x)
    by = {l: {} for l in g.rules}
    for l, rs in g.rules.items():
        for b, c, p in rs:
            by[l].setdefault(b, []).append((c, p))
    memo = {}

    def gen(l, b, i, j):                        # P(symbol b at level l yields x[i:j])
        key = (l, b, i, j)
        if key in memo:
            return memo[key]
        tot = 0.0
        for ch, p in by[l].get(b, []):
            if l == 1:
                if j - i == len(ch) and tuple(x[i:j]) == ch:
                    tot += p
                continue
            # all ways to cut [i, j) into len(ch) non-empty contiguous parts
            k = len(ch)
            for cuts in itertools.combinations(range(i + 1, j), k - 1):
                bounds = (i,) + cuts + (j,)
                pr = p
                for t in range(k):
                    pr *= gen(l - 1, ch[t], bounds[t], bounds[t + 1])
                    if pr == 0:
                        break
                tot += pr
        memo[key] = tot
        return tot
    return sum(g.prior[b] * gen(g.L, b, 0, len(x)) for b in range(g.V))


# ---------------------------------------------------------------------------- JAX: batched log-lik + EM
def compile_rules(g: VGrammar):
    """Group rules by (level, arity) for batched products. Returns a list per level of
    (lhs idx, children idx matrix, rule ids) and the flat rule table."""
    flat = []
    groups = {}
    for l in range(1, g.L + 1):
        for b, ch, p in g.rules[l]:
            rid = len(flat); flat.append((l, b, ch))
            groups.setdefault(l, {}).setdefault(len(ch), []).append(rid)
    comp = {}
    for l, byk in groups.items():
        comp[l] = [(np.array([flat[r][1] for r in rids]), np.array([flat[r][2] for r in rids]),
                    np.array(rids)) for k, rids in sorted(byk.items())]
    probs = np.array([p for l in range(1, g.L + 1) for _, _, p in g.rules[l]])
    return comp, flat, probs


def make_batch_loglik(g: VGrammar, comp, nmax: int):
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    def one(x, n, logp):                       # x padded with -1 to nmax
        p = jnp.exp(logp)
        idx = jnp.arange(nmax)
        B0 = jnp.zeros((g.A, nmax + 1, nmax + 1))
        valid = x >= 0
        B0 = B0.at[jnp.where(valid, x, 0), idx, idx + 1].add(valid.astype(B0.dtype))
        mats = B0
        for l in range(1, g.L + 1):
            Bl = jnp.zeros((g.V, nmax + 1, nmax + 1))
            for lhs, C, rids in comp[l]:
                M = mats[C[:, 0]]
                for t in range(1, C.shape[1]):
                    M = M @ mats[C[:, t]]
                Bl = Bl.at[lhs].add(p[rids][:, None, None] * M)
            mats = Bl
        prior = jnp.asarray(g.prior)
        return jnp.log(prior @ mats[:, 0, n])

    batched = jax.vmap(one, in_axes=(0, 0, None))
    f = jax.jit(lambda X, N, logp: batched(X, N, logp).sum())
    return f, jax.jit(jax.grad(lambda logp, X, N: batched(X, N, logp).sum()))


def buckets(S: List[List[int]], width: int = 16):
    by = {}
    for s in S:
        nm = int(np.ceil(len(s) / width) * width)
        by.setdefault(nm, []).append(s)
    out = []
    for nm, ss in sorted(by.items()):
        X = np.full((len(ss), nm), -1, dtype=np.int64)
        for i, s in enumerate(ss):
            X[i, :len(s)] = s
        out.append((nm, X, np.array([len(s) for s in ss])))
    return out


class EM:
    """EM for a VGrammar with a fixed rule inventory (the E-step is exact)."""

    def __init__(self, g: VGrammar, batch: int = 64):
        self.g, self.batch = g, batch
        self.comp, self.flat, _ = compile_rules(g)
        self.fns = {}
        self.lhs_key = np.array([(l, b) for l, b, _ in self.flat])

    def _fn(self, nm):
        if nm not in self.fns:
            self.fns[nm] = make_batch_loglik(self.g, self.comp, nm)
        return self.fns[nm]

    def loglik(self, B, logp):
        tot, n = 0.0, 0
        for nm, X, N in B:
            f, _ = self._fn(nm)
            for s in range(0, len(X), self.batch):
                tot += float(f(X[s:s + self.batch], N[s:s + self.batch], logp)); n += len(X[s:s + self.batch])
        return tot / n

    def step(self, B, logp):
        counts = np.zeros(len(logp))
        for nm, X, N in B:
            _, gr = self._fn(nm)
            for s in range(0, len(X), self.batch):
                counts += np.asarray(gr(logp, X[s:s + self.batch], N[s:s + self.batch]))  # = p * dlogP/dp
        new = np.zeros_like(counts)
        for key in {tuple(k) for k in self.lhs_key}:
            m = (self.lhs_key == key).all(1)
            new[m] = counts[m] / max(counts[m].sum(), 1e-300)
        return np.log(np.clip(new, 1e-300, None))


# ---------------------------------------------------------------------------- sparse chart inside-outside
def chart_io(g: VGrammar, x: Sequence[int], probs: Dict[int, Dict[int, float]] = None):
    """Exact inside-outside on the sparse chart.  Returns (log P(x), expected rule counts dict
    {(l, rule_index): count}).  probs[l][rule_index] overrides g's probabilities."""
    x = tuple(x); n = len(x)
    rules = {l: [(b, ch, (probs[l][ri] if probs else p)) for ri, (b, ch, p) in enumerate(rs)]
             for l, rs in g.rules.items()}
    # level 0: tokens -> constituents (a, i, i+1) with inside 1
    ins = {0: {}}
    starts = {0: {}}                      # starts[l][(sym, i)] -> list of (j, key)
    for i, a in enumerate(x):
        k = (a, i, i + 1); ins[0][k] = 1.0
        starts[0].setdefault((a, i), []).append(i + 1)
    edges = {}
    for l in range(1, g.L + 1):
        ins[l], starts[l], edges[l] = {}, {}, []
        S = starts[l - 1]; I = ins[l - 1]
        for ri, (b, ch, p) in enumerate(rules[l]):
            if p == 0.0:
                continue
            for i in range(n):
                # extend chains child by child: list of (end, [bounds])
                partial = [(i, [i])]
                for c in ch:
                    nxt = []
                    for pos, bounds in partial:
                        for j in S.get((c, pos), ()):
                            nxt.append((j, bounds + [j]))
                    partial = nxt
                    if not partial:
                        break
                for j, bounds in partial:
                    pr = p
                    for t, c in enumerate(ch):
                        pr *= I[(c, bounds[t], bounds[t + 1])]
                    key = (b, i, j)
                    if key not in ins[l]:
                        ins[l][key] = 0.0
                        starts[l].setdefault((b, i), []).append(j)
                    ins[l][key] += pr
                    edges[l].append((ri, key, bounds, pr))
    P = sum(g.prior[b] * ins[g.L].get((b, 0, n), 0.0) for b in range(g.V))
    if P <= 0:
        return -np.inf, {}
    out = {g.L: {(b, 0, n): g.prior[b] / P for b in range(g.V)}}
    counts = {}
    for l in range(g.L, 0, -1):
        out[l - 1] = {}
        O = out[l]; I = ins[l - 1]
        for ri, key, bounds, pr in edges[l]:
            o = O.get(key, 0.0)
            if o == 0.0:
                continue
            w = o * pr                               # posterior mass of this hyperedge
            counts[(l, ri)] = counts.get((l, ri), 0.0) + w
            ch = rules[l][ri][1]
            for t, c in enumerate(ch):
                ck = (c, bounds[t], bounds[t + 1])
                out[l - 1][ck] = out[l - 1].get(ck, 0.0) + w / I[ck]
    return float(np.log(P)), counts


def _io_worker(args):
    gj, S, probs = args
    g = VGrammar.from_json(gj)
    tot, C = 0.0, {}
    for s in S:
        ll, c = chart_io(g, s, probs)
        tot += ll
        for k, v in c.items():
            C[k] = C.get(k, 0.0) + v
    return tot, C


def em_sweep(g: VGrammar, S, probs, pool=None, chunks=2):
    """One exact EM sweep; returns (mean log-lik under the OLD probs, new probs)."""
    gj = g.to_json()
    parts = [S[i::chunks] for i in range(chunks)]
    res = pool.map(_io_worker, [(gj, p, probs) for p in parts]) if pool else [_io_worker((gj, S, probs))]
    tot = sum(r[0] for r in res); C = {}
    for _, c in res:
        for k, v in c.items():
            C[k] = C.get(k, 0.0) + v
    new = {}
    for l, rs in g.rules.items():
        z = {}
        for ri, (b, _, _) in enumerate(rs):
            z[b] = z.get(b, 0.0) + C.get((l, ri), 0.0)
        new[l] = {ri: (C.get((l, ri), 0.0) / z[b] if z[b] > 0 else probs[l][ri]) for ri, (b, _, _) in enumerate(rs)}
    return tot / len(S), new


def probs_of(g: VGrammar):
    return {l: {ri: p for ri, (_, _, p) in enumerate(rs)} for l, rs in g.rules.items()}
