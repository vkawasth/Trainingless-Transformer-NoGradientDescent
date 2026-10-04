"""Grading regions by Blackwell order: the reward-free value of a region, after Bohinen-Perrone (arXiv:2502.14941).

A region U (a set of measurements observed together) is a statistical experiment Theta -> X_U. Its standard measure
(Def. 2.34 there) is the law of the posterior P(Theta | X_U) under the prior: a distribution on the simplex of Theta.
Blackwell order (Thm 2.38 there): U <= V (V is at least as informative) iff the standard measure of V dominates that
of U in the convex order (second-order stochastic dominance / mean-preserving spread). Consequences used here:
  * for EVERY decision problem with reward R[a, theta], the optimal expected reward
        val(U) = E_{x ~ X_U} max_a sum_theta R[a, theta] P(theta | x)
    is the integral of a convex function of the posterior, hence monotone in Blackwell order;
  * so a region dominated by a no-more-expensive region can never be the optimum of any (reward, cost) trade-off:
    the cost-Blackwell Pareto frontier is computed ONCE, reward-free, and contains every optimum (exact pruning);
  * transitivity of the order (partial evaluations compose; their Thm 3.1) is what makes staged gluing coherent.
For a binary Theta the standard measure is a law on [0, 1] and the convex order is checked exactly: equal means and
E(t - X)_+ (the integrated CDF) ordered at every atom.
Everything assumes ONE joint law (a gluable region). For regions whose contexts are sampled separately, the
'experiment' exists only for each consistent gluing: the value becomes an interval over the fibre (robust_value);
when the family is contextual the fibre is empty and the region has no value at all.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linprog


def cells(n):
    return np.array(list(itertools.product((0, 1), repeat=n)))


def standard_measure(joint, U, n, tol=1e-12):
    """joint over (theta, x_1..x_n) in lexicographic order, theta first. Returns (posterior atoms P(theta=1|x_U),
    weights P(x_U)), merged."""
    J = np.asarray(joint).reshape((2,) * (n + 1))
    keep = (0,) + tuple(1 + i for i in sorted(U)); drop = tuple(a for a in range(n + 1) if a not in keep)
    M = J.sum(axis=drop) if drop else J
    M = M.reshape(2, -1); px = M.sum(0); m = px > tol
    post = M[1, m] / px[m]; w = px[m]
    order = np.argsort(post); post, w = post[order], w[order]
    out_p, out_w = [], []
    for p_, w_ in zip(post, w):
        if out_p and abs(out_p[-1] - p_) < 1e-10:
            out_w[-1] += w_
        else:
            out_p.append(p_); out_w.append(w_)
    return np.array(out_p), np.array(out_w)


def _icdf(atoms, weights, t):
    """E (t - X)_+ at each t, vectorised: t F(t) - sum_{x <= t} w x"""
    o = np.argsort(atoms); a, w = atoms[o], weights[o]
    cw = np.concatenate([[0.0], np.cumsum(w)]); cwx = np.concatenate([[0.0], np.cumsum(w * a)])
    idx = np.searchsorted(a, t, side="right")
    return t * cw[idx] - cwx[idx]


def blackwell_leq(mu, nu, tol=1e-10):
    """mu <=_Blackwell nu for binary Theta: nu is a mean-preserving spread of mu"""
    (a, w), (b, v) = mu, nu
    if abs(np.sum(a * w) - np.sum(b * v)) > 1e-9:
        return False
    t = np.union1d(a, b)
    return bool(np.all(_icdf(a, w, t) <= _icdf(b, v, t) + tol))


def value(mu, R):
    """optimal expected reward of the decision problem R[a, theta] (theta in {0, 1}) given standard measure mu"""
    a, w = mu
    post = np.stack([1 - a, a])                                       # 2 x atoms
    return float(np.sum(w * (R @ post).max(0)))


def mutual_information(mu, prior1):
    a, w = mu
    def h(p):
        p = np.clip(p, 1e-15, 1 - 1e-15); return -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return float(h(prior1) - np.sum(w * h(a)))


def frontier(measures, costs, grade=None):
    """the cost-Blackwell Pareto frontier, by one sweep in order of increasing cost: a region is pruned if some region
    already ON the frontier is no more expensive and at least as informative. Checking only against frontier members is
    exact because the order is transitive (a region dominated by a pruned region is dominated by whatever pruned it).
    Returns (frontier indices, number of dominance checks)."""
    n = len(measures); grade = np.zeros(n) if grade is None else np.asarray(grade)
    order = sorted(range(n), key=lambda i: (costs[i], -grade[i])); F = []; checks = 0
    for i in order:
        dominated = False
        for j in F:
            checks += 1
            if costs[j] <= costs[i] + 1e-12 and blackwell_leq(measures[i], measures[j]):
                dominated = True; break
        if not dominated:
            F.append(i)
    return F, checks


# ------------------------------------------------------------------------------------------- obstructions: interval values
def robust_value(n_vars, contexts, margins, target, inputs, R):
    """Contexts sampled separately (margins m_c over the variables of context c). Region = decide `target` from
    `inputs`. The value is an interval over all joints consistent with the margins (the fibre):
        hi = max over joints of the optimal value, lo = max over decision rules of the min over joints of its reward.
    Rules are enumerated (deterministic maps inputs -> action). Returns (lo, hi) or None if the fibre is empty."""
    X = cells(n_vars); K = len(X)
    Aeq = [np.ones(K)]; beq = [1.0]
    for c, m in zip(contexts, margins):
        sub = list(itertools.product((0, 1), repeat=len(c)))
        for j, s in enumerate(sub):
            Aeq.append(np.array([float(tuple(x[list(c)]) == s) for x in X])); beq.append(m[j])
    Aeq = np.array(Aeq); beq = np.array(beq)
    if linprog(np.zeros(K), A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs").status != 0:
        return None
    in_states = list(itertools.product((0, 1), repeat=len(inputs))); A = R.shape[0]
    lo, hi = -np.inf, -np.inf
    for rule in itertools.product(range(A), repeat=len(in_states)):
        r = np.array([R[rule[in_states.index(tuple(x[list(inputs)]))], x[target]] for x in X])
        mn = linprog(r, A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs").fun
        mx = -linprog(-r, A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs").fun
        lo = max(lo, mn); hi = max(hi, mx)
    return float(lo), float(hi)


def ipf_glue(n_vars, contexts, margins, iters=500):
    """the naive maximum-entropy gluing (IPF), which returns SOME joint even when no consistent joint exists"""
    X = cells(n_vars); p = np.full(len(X), 1.0 / len(X))
    for _ in range(iters):
        for c, m in zip(contexts, margins):
            sub = list(itertools.product((0, 1), repeat=len(c)))
            idx = np.array([sub.index(tuple(x[list(c)])) for x in X])
            cur = np.bincount(idx, weights=p, minlength=len(sub))
            p = p * (np.asarray(m)[idx] / np.maximum(cur[idx], 1e-300))
            if p.sum() <= 0:
                return None                                            # the margins exclude every joint cell
    return p / p.sum()


# =========================================================================================== experiments as matrices
# An experiment is a likelihood matrix L (k x m): column x holds P(x | theta) for the k values of Theta. With a prior
# pi, the standard measure has atoms posterior(x) = pi * L[:, x] / (pi . L[:, x]) and weights pi . L[:, x].
# Columns with proportional likelihoods give the same posterior and are merged (this is the coarse-graining that
# leaves the standard measure unchanged).
RESOLUTION = 10                                                         # decimals of posterior merging (10: exact)


def merge_columns(L, tol=1e-12, decimals=None):
    """merge columns with the same posterior direction. decimals < 10 merges NEARBY posteriors too: a coarse-graining
    (a garbling), so values can only drop, by at most the reward range times the merged posterior spread."""
    L = np.asarray(L, float); s = L.sum(0); L = L[:, s > tol]
    if L.shape[1] == 0:
        return L
    D = L / L.sum(0, keepdims=True); keys = np.round(D, RESOLUTION if decimals is None else decimals)
    _, inv = np.unique(keys, axis=1, return_inverse=True)
    inv = np.asarray(inv).ravel()
    out = np.zeros((L.shape[0], inv.max() + 1)); np.add.at(out.T, inv, L.T)
    return out


def experiment(joint, U, n, k=2):
    """likelihood matrix of the region U from a joint over (theta in 0..k-1, x_1..x_n binary); theta first"""
    J = np.asarray(joint).reshape((k,) + (2,) * n)
    keep = (0,) + tuple(1 + i for i in sorted(U)); drop = tuple(a for a in range(n + 1) if a not in keep)
    M = (J.sum(axis=drop) if drop else J).reshape(k, -1)
    prior = M.sum(1)
    return merge_columns(M / prior[:, None]), prior


def product(L1, L2):
    """experiment of two regions that are conditionally independent given Theta"""
    return merge_columns((L1[:, :, None] * L2[:, None, :]).reshape(L1.shape[0], -1))


def measure_k(L, prior):
    W = prior[:, None] * L; w = W.sum(0)
    return (W / w).T, w                                                # atoms (m x k), weights


def value_k(L, prior, R):
    """optimal expected reward of R[a, theta]"""
    W = prior[:, None] * L
    return float((R @ W).max(0).sum())


def leq_k(L1, L2, prior, tol=1e-9):
    """Blackwell order L1 <= L2. Binary Theta: the exact convex-order test. k > 2: Strassen's theorem as an LP --
    a martingale coupling pi_ab >= 0 with row sums w_a, column sums v_b and sum_b pi_ab (nu_b - mu_a) = 0."""
    if L1.shape[0] == 2:
        a, w = measure_k(L1, prior); b, v = measure_k(L2, prior)
        return blackwell_leq((a[:, 1], w), (b[:, 1], v), tol)
    A, w = measure_k(L1, prior); B_, v = measure_k(L2, prior); m1, m2 = len(w), len(v); k = A.shape[1]
    rows, rhs = [], []
    for i in range(m1):
        r = np.zeros(m1 * m2); r[i * m2:(i + 1) * m2] = 1; rows.append(r); rhs.append(w[i])
    for j in range(m2):
        r = np.zeros(m1 * m2); r[j::m2] = 1; rows.append(r); rhs.append(v[j])
    for i in range(m1):
        for c in range(k - 1):
            r = np.zeros(m1 * m2); r[i * m2:(i + 1) * m2] = B_[:, c] - A[i, c]; rows.append(r); rhs.append(0.0)
    res = linprog(np.zeros(m1 * m2), A_eq=np.array(rows), b_eq=np.array(rhs), bounds=(0, None), method="highs")
    return res.status == 0


def pareto(cands, prior):
    """cands: list of (cost, L, label). Sweep in order of increasing cost; keep those not dominated by a kept one."""
    cands = sorted(cands, key=lambda c: (c[0], -value_k(c[1], prior, np.eye(len(prior)))))
    F = []; checks = 0
    for c in cands:
        dom = False
        for f in F:
            checks += 1
            if f[0] <= c[0] + 1e-12 and leq_k(c[1], f[1], prior):
                dom = True; break
        if not dom:
            F.append(c)
    return F, checks


def grow_frontier(blocks, prior):
    """EXACT frontier growth over blocks that are conditionally independent given Theta.
    blocks: list of lists of (cost, L, label) -- each block's candidate sub-regions, the empty one included.
    Stage b: F_b = frontier{ f x g : f in F_{b-1}, g in frontier(block b) }. Exact because (i) the product with a common
    experiment preserves Blackwell order (garble with M x I), so any combination that uses a dominated sub-region is
    dominated by the one that swaps in its dominator; (ii) the order is transitive. Returns the frontier and the
    number of combinations formed (the work), against prod_b |block b| for full enumeration."""
    k = len(prior); F = [(0.0, np.ones((k, 1)), ())]; formed = 0; checks = 0
    for blk in blocks:
        Fb, c1 = pareto(blk, prior); checks += c1
        cands = [(f[0] + g[0], product(f[1], g[1]), f[2] + tuple(g[2])) for f in F for g in Fb]
        formed += len(cands)
        F, c2 = pareto(cands, prior); checks += c2
    return F, formed, checks
