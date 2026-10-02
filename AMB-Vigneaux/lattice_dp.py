"""Dynamic programming on the gluing lattice: sample regions that satisfy every rule, using second-order jets.

Setting. Sources s = 1..S, topics x = 1..X, a logit table y[s, x] with weights w[s, x] (binomial counts). A region
is a set U of topics, encoded by indicators z_x in {0, 1}. Its consistency is the deviance of the additive (glued,
loop-free) model on U:
        D(z) = min_{alpha, beta} sum_x z_x sum_s w_sx (y_sx - alpha_s - beta_x)^2 .
Jets of D in the indicators (closed form, envelope theorem; profile out beta_x, so each topic's term is L_x(alpha)):
        first order   D'_x   = L_x(alpha_hat)                                (topic x's loop residual sum of squares)
        second order  D''_xy = - g_x^T H^+ g_y,  g_x = grad L_x(alpha_hat),  H = sum_x hess L_x(alpha_hat)
        D(U) ~ D(full) - sum_{x not in U} D'_x + 1/2 sum_{x, y not in U} D''_xy        ('jets fast path')
Policy. The Gibbs measure pi(U) ~ exp(-[D~(U) - 2 df(U)] / 2) * [U satisfies the local rules] -- the soft-optimal
policy that a policy-gradient method would approach, obtained here exactly by sum-product (no gradient iterations).
exp(-D/2) is a likelihood ratio; the df term (AIC rate) makes a clean topic worth keeping and a looped one worth
dropping. Global rules that are smooth in the region (a target outcome's value) are made local by their own jets
and carried as DP state; the rest are checked exactly on each sample.
Structure. Second-order terms couple topics; keep the strong ones, take connected components ('bags', capped in size)
as the clusters of a tree decomposition, chain the bags, and carry the counting rules (region size, group quotas) as
DP state. Forward filtering over the bag chain, backward sampling = exact samples from pi under the kept terms.
Cross-bag terms dropped by the cap and the GLOBAL rules (the region glues; a target outcome is stable) are checked
exactly on each sample (accept / reject). Loops between bags are what makes it hard: the bag cap is the treewidth.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy import stats
from scipy.special import logsumexp


# ------------------------------------------------------------------------------------------- exact additive fit
def additive_fit(y, w, U):
    """weighted additive fit alpha_s + beta_x on topics U (exact: profile beta, solve the S x S system)."""
    U = np.asarray(U, int); Y = y[:, U]; Wt = w[:, U]
    Wx = Wt.sum(0)                                                   # per topic
    # normal equations for alpha after eliminating beta_x = sum_s w (y - alpha) / W_x
    A = np.diag(Wt.sum(1)) - (Wt / Wx) @ Wt.T
    b = (Wt * Y).sum(1) - (Wt / Wx) @ (Wt * Y).sum(0)
    alpha = np.linalg.lstsq(A + 1e-9 * np.eye(len(A)), b, rcond=None)[0]
    beta = ((Wt * (Y - alpha[:, None])).sum(0)) / Wx
    R = Y - alpha[:, None] - beta[None, :]
    return alpha, beta, float((Wt * R ** 2).sum()), A


def glue_p(y, w, U):
    S = y.shape[0]; _, _, D, _ = additive_fit(y, w, U); df = (S - 1) * (len(U) - 1)
    return float(stats.chi2.sf(D, df)), D, df


def target_stability(y, w, U, s_star, t_star, eta0):
    """outcome = [lean of source s* >= eta0], lean = alpha_s* - mean_s alpha_s on region U (the source's house effect,
    estimated from every topic in U); stability r = |lean_hat - eta0| / se. t_star is a topic the rules require."""
    U = list(U); S = y.shape[0]
    alpha, beta, D, A = additive_fit(y, w, U)
    c = -np.full(S, 1.0 / S); c[s_star] += 1.0
    var = float(c @ np.linalg.pinv(A) @ c); eta = float(c @ alpha)
    return float(abs(eta - eta0) / np.sqrt(var)), eta


# ------------------------------------------------------------------------------------------- jets of D in the indicators
def jets(y, w):
    S, X = y.shape; alpha, beta, D, _ = additive_fit(y, w, np.arange(X))
    E = y - alpha[:, None]; Wx = w.sum(0); ebar = (w * E).sum(0) / Wx; R = E - ebar[None, :]
    d1 = (w * R ** 2).sum(0)                                          # L_x(alpha_hat)
    G = -2 * w * R                                                    # S x X: grad of L_x wrt alpha
    H = sum(2 * (np.diag(w[:, x]) - np.outer(w[:, x], w[:, x]) / Wx[x]) for x in range(X))
    Hp = np.linalg.pinv(H)
    d2 = -G.T @ Hp @ G                                                # X x X
    return dict(D_full=D, d1=d1, d2=d2, alpha=alpha, beta=beta)


def jet_predict(J, removed, order=2):
    r = np.asarray(sorted(removed), int)
    out = J["D_full"] - J["d1"][r].sum()
    if order >= 2 and len(r):
        out += 0.5 * J["d2"][np.ix_(r, r)].sum()
    return float(out)


# ------------------------------------------------------------------------------------------- bags (tree decomposition)
def bags_from_graph(d2, cap=8, forced_pairs=(), quantile=0.98):
    """connected components of the graph of strong second-order couplings |D''_xy| (and forced pairs: exclusion
    rules must sit inside one bag), threshold raised until every component has at most `cap` topics."""
    X = d2.shape[0]; A = np.abs(d2.copy()); np.fill_diagonal(A, 0)
    tau = np.quantile(A[np.triu_indices(X, 1)], quantile)
    while True:
        parent = list(range(X))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i
        for i, j in zip(*np.where(np.triu(A > tau, 1))):
            parent[find(i)] = find(j)
        for i, j in forced_pairs:
            parent[find(i)] = find(j)
        comps = {}
        for i in range(X):
            comps.setdefault(find(i), []).append(i)
        bags = sorted(comps.values(), key=lambda b: min(b))
        if max(len(b) for b in bags) <= cap or tau > A.max():
            return bags, float(tau)
        tau *= 1.25


# ------------------------------------------------------------------------------------------- the sampler
@dataclass
class Rules:
    k_min: int = 0                                  # region size >= k_min
    group: Sequence[int] = ()                       # topics of a quota group
    m_min: int = 0                                  # at least m_min topics of the group
    must: Sequence[int] = ()                        # topics that must be in U
    exclusive: Sequence[tuple] = ()                 # pairs that cannot both be in U


def lean_jets(y, w, bags, s_star, U_full=None):
    """first-order (exact single deletions) and within-bag second-order (exact pair deletions minus the first-order
    parts) changes of the target outcome lean = alpha_s* - mean alpha when topics are removed"""
    S, X = y.shape; c = -np.full(S, 1.0 / S); c[s_star] += 1.0
    full = list(range(X)); base = float(c @ additive_fit(y, w, full)[0])
    d = np.array([float(c @ additive_fit(y, w, [u for u in full if u != x])[0]) - base for x in range(X)])
    pair = {}
    for b in bags:
        for i, j in itertools.combinations(b, 2):
            v = float(c @ additive_fit(y, w, [u for u in full if u not in (i, j)])[0]) - base
            pair[(i, j)] = v - d[i] - d[j]
    A = additive_fit(y, w, full)[3]; se = float(np.sqrt(c @ np.linalg.pinv(A) @ c))
    return dict(base=base, d=d, pair=pair, se_full=se)


@dataclass
class Sampler:
    """exact forward-filtering backward-sampling over a chain of bags.
    State = (region size, quota-group count[, bin of the predicted target outcome]). Log-weight of a region:
    -beta/2 * [D~(U) - c_pen * df(U)], D~ from the jets (order 1 or 2), df(U) = (S - 1)(|U| - 1): c_pen = 2 is the
    AIC-type exchange rate (a topic is worth keeping unless its loop excess exceeds its degrees of freedom).
    With `lean` (from lean_jets) and `target` = (eta0, r0), the stability rule |lean(U) - eta0| / se(U) >= r0 is
    carried in the state through the jet prediction of lean(U), se(U) ~ se_full sqrt(X / |U|)."""
    X: int
    bags: List[List[int]]
    d1: np.ndarray
    d2: np.ndarray
    rules: Rules
    beta: float = 1.0
    order: int = 2
    S: int = 50
    c_pen: float = 2.0
    lean: Optional[dict] = None
    target: Optional[tuple] = None
    nbins: int = 120
    tables: list = field(default_factory=list)

    def __post_init__(self):
        X = self.X; G = set(self.rules.group); must = set(self.rules.must); excl = [tuple(p) for p in self.rules.exclusive]
        use_lean = self.lean is not None and self.target is not None
        if use_lean:
            d = self.lean["d"]; lo = float(np.minimum(d, 0).sum()) - 1e-9; hi = float(np.maximum(d, 0).sum()) + 1e-9
            lo += min(0.0, sum(min(v, 0) for v in self.lean["pair"].values())); hi += max(0.0, sum(max(v, 0) for v in self.lean["pair"].values()))
            self.edges = np.linspace(lo, hi, self.nbins + 1); self.width = (hi - lo) / self.nbins; self.lo = lo
        nb = self.nbins if use_lean else 1
        self.KA = (X + 1, len(G) + 1, nb); self.use_lean = use_lean
        zero_bin = int(np.clip((0.0 - self.lo) / self.width, 0, nb - 1)) if use_lean else 0
        self.configs = []
        for b in self.bags:
            rows = []
            for bits in itertools.product((0, 1), repeat=len(b)):
                inn = [x for x, z in zip(b, bits) if z]; out = [x for x, z in zip(b, bits) if not z]
                if any(x in must for x in out):
                    continue
                if any(i in inn and j in inn for i, j in excl):
                    continue
                dD = -self.d1[out].sum()
                if self.order >= 2 and out:
                    dD += 0.5 * self.d2[np.ix_(out, out)].sum()
                lw = -self.beta * (dD + self.c_pen * (self.S - 1) * len(out)) / 2
                shift = 0
                if use_lean:
                    dl = float(self.lean["d"][out].sum())
                    if self.order >= 2:
                        dl += sum(self.lean["pair"].get((min(i, j), max(i, j)), 0.0) for i, j in itertools.combinations(out, 2))
                    shift = int(round(dl / self.width))
                rows.append((tuple(inn), len(inn), sum(x in G for x in inn), lw, shift))
            self.configs.append(rows)
        F = np.full(self.KA, -np.inf); F[0, 0, zero_bin] = 0.0; self.tables = [F]
        for rows in self.configs:
            Fn = np.full(self.KA, -np.inf)
            for inn, k, a, lw, sh in rows:
                T = np.full(self.KA, -np.inf)
                src = F[:self.KA[0] - k, :self.KA[1] - a]
                if sh >= 0:
                    T[k:, a:, sh:] = src[:, :, :nb - sh] + lw
                else:
                    T[k:, a:, :nb + sh] = src[:, :, -sh:] + lw
                Fn = np.logaddexp(Fn, T)
            self.tables.append(Fn); F = Fn
        Fin = F.copy(); Fin[:self.rules.k_min] = -np.inf; Fin[:, :self.rules.m_min] = -np.inf
        if use_lean:
            eta0, r0 = self.target; centres = self.lo + (np.arange(nb) + 0.5) * self.width + self.lean["base"]
            for k in range(1, X + 1):
                se = self.lean["se_full"] * np.sqrt(X / k)
                Fin[k][:, np.abs(centres - eta0) / se < r0] = -np.inf
        self.final = Fin; self.logZ = float(logsumexp(Fin))

    def sample(self, rng):
        P = np.exp(self.final - self.logZ).ravel(); idx = rng.choice(P.size, p=P / P.sum()); k, a, l = np.unravel_index(idx, self.KA)
        U = []
        for b in range(len(self.bags) - 1, -1, -1):
            prev = self.tables[b]; rows = self.configs[b]
            def ok(r):
                return k - r[1] >= 0 and a - r[2] >= 0 and 0 <= l - r[4] < self.KA[2]
            lw = np.array([r[3] + (prev[k - r[1], a - r[2], l - r[4]] if ok(r) else -np.inf) for r in rows])
            p = np.exp(lw - logsumexp(lw)); c = rows[rng.choice(len(rows), p=p / p.sum())]
            U.extend(c[0]); k -= c[1]; a -= c[2]; l -= c[4]
        return sorted(U)


def check(y, w, U, rules: Rules, target=None, alpha_glue=0.05, r0=2.0):
    """exact global rules: local rules again, the region glues (p >= alpha_glue), and the target outcome is stable"""
    Us = set(U); G = set(rules.group)
    ok_local = (len(U) >= rules.k_min and sum(x in G for x in U) >= rules.m_min and all(m in Us for m in rules.must)
                and not any(i in Us and j in Us for i, j in rules.exclusive))
    p, D, df = glue_p(y, w, U)
    r = target_stability(y, w, U, *target)[0] if target else np.inf
    return dict(local=bool(ok_local), glue_p=p, glues=p >= alpha_glue, r=r, stable=r >= r0,
                feasible=bool(ok_local and p >= alpha_glue and r >= r0))
