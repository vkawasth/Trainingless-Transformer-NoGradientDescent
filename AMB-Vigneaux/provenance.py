"""provenance.py -- shared library for "Precise Attribution of Model Interventions".

Conventions
-----------
* A distribution over a finite outcome set is a list of numbers indexed by outcome position.
  Exact work uses fractions.Fraction; linear programs use floats (scipy HiGHS) and their
  certificates are rationalized and re-verified exactly.
* TV(p, q) = (1/2) * sum |p_b - q_b|.
* A source is either a distribution ("exact") or a support set ("support", only its support is known).
"""
from fractions import Fraction as F
import numpy as np
from scipy.optimize import linprog

# ----------------------------------------------------------------- basics
def tv(p, q):
    return sum(abs(a - b) for a, b in zip(p, q)) / 2

def l1(p, q):
    return sum(abs(a - b) for a, b in zip(p, q))

def mix(weights, dists):
    n = len(dists[0])
    return [sum(w * d[b] for w, d in zip(weights, dists)) for b in range(n)]

def point_mass(n, b):
    return [F(1) if i == b else F(0) for i in range(n)]

def coarsen(p, groups):
    """push p forward along the map outcome -> group index"""
    return [sum(p[b] for b in g) for g in groups]

def rat(v, den=10**6):
    return F(v).limit_denominator(den)

# ----------------------------------------------------------------- attribution sets
def _attribution_lp(p, sources, budget):
    """Variables: one block per source.
    exact source i: a scalar lambda_i (its contribution is lambda_i * A_i);
    support source i with support S_i: a vector s_i on S_i (its weight is sum s_i).
    Constraints: sum of contributions <= p (pointwise), total weight >= 1 - budget, all >= 0.
    Returns (n_vars, A_ub, b_ub, weight_rows, contrib) where weight_rows[i] is the linear form of
    lambda_i and contrib[i][b] is the linear form of source i's contribution at outcome b."""
    n = len(p)
    cols = []          # (source index, outcome or None)
    for i, (kind, data) in enumerate(sources):
        if kind == "exact":
            cols.append((i, None))
        else:
            for b in sorted(data):
                cols.append((i, b))
    nv = len(cols)
    contrib = [[np.zeros(nv) for _ in range(n)] for _ in sources]
    weight = [np.zeros(nv) for _ in sources]
    for j, (i, b) in enumerate(cols):
        kind, data = sources[i]
        if kind == "exact":
            for c in range(n):
                contrib[i][c][j] = float(data[c])
            weight[i][j] = 1.0
        else:
            contrib[i][b][j] = 1.0
            weight[i][j] = 1.0
    A_ub, b_ub = [], []
    for c in range(n):
        A_ub.append(sum(contrib[i][c] for i in range(len(sources)))); b_ub.append(float(p[c]))
    A_ub.append(-sum(weight)); b_ub.append(-(1.0 - float(budget)))
    return nv, np.array(A_ub), np.array(b_ub), weight, contrib

def attribution_report(p, sources, budget=0, outcomes_of_interest=None):
    """Sharp intervals for each weight lambda_i, for mu = 1 - sum lambda, and for the
    observation-level attribution pi_i(b) = contribution_i(b) / p(b).
    Returns None if the attribution set is empty (budget < mu_min)."""
    nv, A, b, weight, contrib = _attribution_lp(p, sources, budget)
    bounds = [(0, None)] * nv
    def opt(c):
        lo = linprog(c, A_ub=A, b_ub=b, bounds=bounds)
        hi = linprog(-c, A_ub=A, b_ub=b, bounds=bounds)
        if lo.status != 0 or hi.status != 0:
            return None
        return (lo.fun, -hi.fun)
    tot = sum(weight)
    t = opt(tot)
    if t is None:
        return None
    rep = {"lambda": [opt(w) for w in weight], "mu": (1 - t[1], 1 - t[0]), "pi": {}}
    for bb in (outcomes_of_interest or []):
        if float(p[bb]) > 0:
            rep["pi"][bb] = [opt(contrib[i][bb] / float(p[bb])) for i in range(len(sources))]
    rep["widths"] = [hi - lo for lo, hi in rep["lambda"]]
    rep["mu_width"] = rep["mu"][1] - rep["mu"][0]
    return rep

# ----------------------------------------------------------------- unmodeled weight
def mu_min_sources(p, A):
    """mu_min(p) = 1 - max{sum lambda : sum lambda_i A_i <= p}, with an exactly verified dual
    certificate y >= 0, <y, A_i> >= 1.  Returns (mu_min_float, lower_bound_exact, y_exact)."""
    k, n = len(A), len(p)
    M = np.array([[float(a[c]) for a in A] for c in range(n)])
    pr = linprog(-np.ones(k), A_ub=M, b_ub=np.array([float(v) for v in p]), bounds=[(0, None)] * k)
    du = linprog(np.array([float(v) for v in p]), A_ub=-M.T, b_ub=-np.ones(k), bounds=[(0, None)] * n)
    y = [max(rat(v), F(0)) for v in du.x]
    s = min(sum(yc * a[c] for c, yc in enumerate(y)) for a in A)
    if s > 0 and s < 1:
        y = [v / s for v in y]
    assert all(v >= 0 for v in y) and all(sum(yc * a[c] for c, yc in enumerate(y)) >= 1 for a in A)
    return 1 + pr.fun, 1 - sum(yc * pc for yc, pc in zip(y, p)), y

# ----------------------------------------------------------------- exponential tilts
def tilt(p, w):
    """T_w(p)(b) = p(b) w(b) / sum_c p(c) w(c)"""
    z = sum(a * b for a, b in zip(p, w))
    return [a * b / z for a, b in zip(p, w)]

def implied_ratio(p, q):
    """K*(p, q) = max_b (q_b/p_b) / min_b (q_b/p_b) over supp p, or None if supp q is not inside supp p.
    q is reachable by a tilt with weight ratio K iff K*(p, q) <= K."""
    if any(q[b] > 0 and p[b] == 0 for b in range(len(p))):
        return None
    r = [q[b] / p[b] for b in range(len(p)) if p[b] > 0]
    if min(r) == 0:
        return None
    return max(r) / min(r)

def tilt_cone_rows(p, K):
    """homogeneous rows G with  cone(Reach_K(p)) = {s >= 0 : G s <= 0}."""
    n = len(p); rows = []
    S = [b for b in range(n) if p[b] > 0]
    for b in S:
        for c in S:
            if b != c:
                r = [F(0)] * n; r[b] = p[c]; r[c] = -K * p[b]; rows.append(r)
    for b in range(n):
        if p[b] == 0:
            r = [F(0)] * n; r[b] = F(1); rows.append(r)
    return rows

def cone_residual(q, G):
    """For the polyhedral cone K = {s >= 0 : G s <= 0}:
       explained mass m* = max{1.s : s in K, s <= q};  mu_min = 1 - m*.
    Returns (mu_min_float, s_float, certified_lower_bound_exact, (y, nu) exact certificate),
    where y >= 0, nu >= 0, y + G^T nu >= 1 certifies mu >= 1 - <y, q>."""
    n = len(q); Gf = np.array([[float(v) for v in r] for r in G]) if G else np.zeros((0, n))
    A_ub = np.r_[Gf, np.eye(n)]; b_ub = np.r_[np.zeros(len(G)), np.array([float(v) for v in q])]
    pr = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * n)
    m = len(G)
    # dual: min <y,q> s.t. y + G^T nu >= 1, y, nu >= 0
    c = np.r_[np.array([float(v) for v in q]), np.zeros(m)]
    A = -np.c_[np.eye(n), Gf.T] if m else -np.eye(n)
    du = linprog(c, A_ub=A, b_ub=-np.ones(n), bounds=[(0, None)] * (n + m))
    y = [max(rat(v), F(0)) for v in du.x[:n]]
    nu = [max(rat(v), F(0)) for v in du.x[n:]]
    # repair rounding so the certificate is exactly valid
    for b in range(n):
        lhs = y[b] + sum(G[j][b] * nu[j] for j in range(m))
        if lhs < 1:
            y[b] += 1 - lhs
    assert all(v >= 0 for v in y + nu)
    assert all(y[b] + sum(G[j][b] * nu[j] for j in range(m)) >= 1 for b in range(n))
    return 1 + pr.fun, pr.x, 1 - sum(a * b for a, b in zip(y, q)), (y, nu)

# ----------------------------------------------------------------- output-level mechanisms
def mixing(a, target):
    """p -> (1 - a) p + a * target"""
    return lambda p: [(1 - a) * x + a * t for x, t in zip(p, target)]

def tilt_map(w):
    return lambda p: tilt(p, w)

def lipschitz_bound_tilt(K):
    return (3 * K - 1) / 2

def dist(F1, F2, states):
    """d_S(F1, F2) = max over states of TV(F1(p), F2(p))"""
    return max(tv(F1(p), F2(p)) for p in states)
