"""Probing inside arity: the hull tree of a filled cell, and the p-adic radius profile.

A unit covered by k sources is a filled (k-1)-simplex. Its linear invariants inside the cell vanish: the loop sum of
the differences x_i - x_j over any cycle of sources on the SAME unit is 0 (Proposition 'filled cells are flat').
What is left is the configuration of the k values modulo translation. Its combinatorial shadow is the HULL TREE:
  * non-archimedean values (integers in Z_p): the convex hull of the k points in the Berkovich line is a finite tree;
    its vertices are the discs where the points split. Along it, log-distance is piecewise linear with integer slopes
    (the number of points in the disc) -- the elementary fact behind the continuity / integer-slope theorems for
    radius functions on Berkovich curves. disc_profile gives N(m) = number of discs of radius p^-m that contain the points.
  * real values: the single-linkage dendrogram (the ultrametric closure of |x_i - x_j|), the archimedean shadow.
The first merge is the COALITION of the cell: the sources that agree first. Translation by an event effect does not
change the tree; subtracting source effects makes it invariant under every natural change (u_s + v_e), so coalition
frequencies beyond an additive (lean) null are structure inside arity that holonomy cannot see.

obstruction_profile: for sum claims x_i + x_j = r over Z_p, e(m) = log_p of the ORDER of the obstruction class [r] in
coker(A) (x) Z/p^m (canonical: independent of the Smith basis). e is piecewise linear in m with integer slopes (0 or 1),
e(m) = max_i (min(m, v_p(d_i)) - v_p(c_i))^+ ; it is 0 exactly for m <= m* (the local hulls glue at radius p^-m*),
rises with slope 1 after m*, and flattens at a torsion ceiling v_p(d_i). Finitely many breakpoints: the p-adic
analogue of a radius function that is continuous, piecewise log-linear with integer slopes and a finite controlling set.
"""
from __future__ import annotations
import itertools, collections
import numpy as np
from .padic_loops import smith, vp


def padic_dist(a, b, p):
    return 0.0 if a == b else float(p) ** (-vp(int(a) - int(b), p))


def hull_tree(x, dist=lambda a, b: abs(a - b)):
    """single-linkage merges [(height, frozenset A, frozenset B)] of the points x (indices 0..k-1)."""
    k = len(x); parent = list(range(k))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    members = {i: frozenset([i]) for i in range(k)}; merges = []
    for d, i, j in sorted((dist(x[i], x[j]), i, j) for i, j in itertools.combinations(range(k), 2)):
        a, b = find(i), find(j)
        if a != b:
            merges.append((d, members[a], members[b])); parent[a] = b; members[b] = members[a] | members[b]
    return merges


def coalition(x, tol=0.0):
    """the first-merging pair if the smallest gap is unique (beyond tol), else None."""
    pairs = sorted((abs(x[i] - x[j]), (i, j)) for i, j in itertools.combinations(range(len(x)), 2))
    if len(pairs) > 1 and pairs[1][0] - pairs[0][0] <= tol:
        return None
    return pairs[0][1]


def disc_profile(x, p, mmax):
    """N(m) = number of residue classes mod p^m among the integers x, m = 0..mmax, and the class-size composition."""
    out = []
    for m in range(mmax + 1):
        c = collections.Counter(int(v) % p ** m for v in x)
        out.append((m, len(c), tuple(sorted(c.values(), reverse=True))))
    return out


def obstruction_profile(n, claims, p, mmax):
    A = [[0] * n for _ in claims]; r = []
    for row, (i, j, rr) in zip(A, claims):
        row[i] += 1; row[j] += 1; r.append(int(rr))
    U, D, _ = smith(A)
    c = [sum(U[i][k] * r[k] for k in range(len(r))) for i in range(len(r))]
    vd = [vp(D[i][i], p) if i < min(len(D), len(D[0])) and D[i][i] != 0 else float("inf") for i in range(len(c))]
    vc = [vp(ci, p) for ci in c]
    return [(m, int(max([0] + [min(m, b) - a for a, b in zip(vc, vd) if a < min(m, b)]))) for m in range(mmax + 1)]


def coalition_test(cells, sources, B=2000, seed=0, tol=0.0):
    """cells: list of dicts {source: value} with all `sources` present. Additive null: x = v_e + u_s + e, residuals
    permuted within source across cells. Returns observed / null-mean counts and two-sided p for each coalition."""
    rng = np.random.default_rng(seed)
    Y = np.array([[c[s] for s in sources] for c in cells], float)
    ve = Y.mean(1, keepdims=True); u = (Y - ve).mean(0); res = Y - ve - u
    def counts(Z):
        cnt = collections.Counter()
        for z in Z:
            cc = coalition(z, tol); cnt[(sources[cc[0]], sources[cc[1]]) if cc else "tie"] += 1
        return cnt
    obs = counts(Y); sims = []
    for _ in range(B):
        sims.append(counts(ve + u + np.column_stack([rng.permutation(res[:, j]) for j in range(len(sources))])))
    keys = [(a, b) for a, b in itertools.combinations(sources, 2)] + ["tie"]
    out = {}
    for k in keys:
        a = np.array([s[k] for s in sims])
        out[str(k)] = dict(obs=int(obs[k]), null=float(a.mean()),
                           p=float(min(1, 2 * min((a >= obs[k]).mean(), (a <= obs[k]).mean()))))
    return dict(n=len(cells), lean=dict(zip(sources, u.round(4).tolist())), coalitions=out)


def coalition_test_binomial(cells, sources, B=2000, seed=0):
    """cells: list of dicts {source: (k_positive, n)} (e.g. annotator groups on one sentence). Discreteness-preserving
    additive null: k ~ Binomial(n, clip(p_unit + u_s)), p_unit the pooled rate of the unit, u_s the source lean."""
    rng = np.random.default_rng(seed)
    K = np.array([[c[s][0] for s in sources] for c in cells], float); N = np.array([[c[s][1] for s in sources] for c in cells], float)
    P = K / N; pu = K.sum(1, keepdims=True) / N.sum(1, keepdims=True); u = (P - pu).mean(0)
    def counts(Pm):
        cnt = collections.Counter()
        for z in Pm:
            cc = coalition(z); cnt[(sources[cc[0]], sources[cc[1]]) if cc else "tie"] += 1
        return cnt
    obs = counts(P); q = np.clip(pu + u, 0, 1); sims = [counts(rng.binomial(N.astype(int), q) / N) for _ in range(B)]
    keys = [(a, b) for a, b in itertools.combinations(sources, 2)] + ["tie"]
    out = {}
    for k in keys:
        a = np.array([s[k] for s in sims])
        out[str(k)] = dict(obs=int(obs[k]), null=float(a.mean()),
                           p=float(min(1, 2 * min((a >= obs[k]).mean(), (a <= obs[k]).mean()))))
    return dict(n=len(cells), lean=dict(zip(sources, u.round(4).tolist())), coalitions=out)


def coalition_test_labelperm(cells, sources, B=500, seed=0):
    """cells: list of (values, labels) arrays for the raters/items of one unit. Null: permute the source labels within
    the unit (keeps discreteness, group sizes and rater heterogeneity; removes every source effect)."""
    rng = np.random.default_rng(seed)
    def counts(perm):
        cnt = collections.Counter()
        for v, g in cells:
            if perm:
                g = rng.permutation(g)
            z = [v[g == s].mean() for s in sources]; cc = coalition(z)
            cnt[(sources[cc[0]], sources[cc[1]]) if cc else "tie"] += 1
        return cnt
    obs = counts(False); sims = [counts(True) for _ in range(B)]
    keys = [(a, b) for a, b in itertools.combinations(sources, 2)] + ["tie"]
    return dict(n=len(cells), coalitions={str(k): dict(obs=int(obs[k]), null=float(np.mean([s[k] for s in sims])),
                p=float(min(1, 2 * min(np.mean([s[k] >= obs[k] for s in sims]), np.mean([s[k] <= obs[k] for s in sims])))))
                for k in keys})


def variogram(cells, sources, B=1000, seed=0, noise=None):
    """Lean-invariant agreement geometry of sources on shared units.
    gamma_ij = 1/2 Var_units(x_i - x_j): the within-unit disagreement of i and j after removing their mean offset.
    Invariant under every natural change (u_s + v_e). Its single-linkage tree is the tree-level invariant whose first
    merge is the population coalition. cells: list of dicts {source: value} (missing sources allowed; pairwise).
    noise: optional list of dicts {source: sampling variance of that cell value} (e.g. p(1-p)/(n-1) for a mean of n
    binary raters). Then gamma_ij is corrected by subtracting 1/2 (v_i + v_j), so that groups with fewer raters are
    not pushed apart by sampling noise alone.
    Returns gamma, pair counts, bootstrap se, the tree, and bootstrap support for the first merge."""
    rng = np.random.default_rng(seed); k = len(sources)
    X = np.array([[c.get(s, np.nan) for s in sources] for c in cells], float)
    V = np.array([[nz.get(s, np.nan) for s in sources] for nz in noise], float) if noise is not None else np.zeros_like(X)
    def gam(Xm, Vm):
        G = np.full((k, k), 0.0); N = np.zeros((k, k), int)
        for i, j in itertools.combinations(range(k), 2):
            m = ~np.isnan(Xm[:, i]) & ~np.isnan(Xm[:, j]); d = Xm[m, i] - Xm[m, j]
            G[i, j] = G[j, i] = (0.5 * d.var(ddof=1) - 0.5 * np.nanmean(Vm[m, i] + Vm[m, j])) if m.sum() > 2 else np.nan
            N[i, j] = N[j, i] = int(m.sum())
        return G, N
    G, N = gam(X, V)
    def first(Gm):
        iu = [(Gm[i, j], i, j) for i, j in itertools.combinations(range(k), 2) if not np.isnan(Gm[i, j])]
        _, i, j = min(iu); return (sources[i], sources[j])
    top = first(G); boots = []; sup = 0
    for _ in range(B):
        ib = rng.integers(len(X), size=len(X)); Gb, _ = gam(X[ib], V[ib]); boots.append(Gb); sup += first(Gb) == top
    se = np.nanstd(np.array(boots), axis=0)
    tree = hull_tree(list(range(k)), lambda a, b: G[a, b])
    return dict(sources=list(sources), gamma=G.round(5).tolist(), n=N.tolist(), se=se.round(5).tolist(), first_merge=top,
                support=sup / B, tree=[(round(float(h), 5), sorted(sources[i] for i in A), sorted(sources[i] for i in Bm))
                                        for h, A, Bm in tree])
