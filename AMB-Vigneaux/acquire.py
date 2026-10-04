"""Identification-aware data acquisition on a cover: which context to buy next, and when to stop.

Problem. n binary measurements, a set of contexts (subsets of measurements) each with a per-sample cost. A decision
'is <f, p> >= t ?' about the unknown joint p must be made with error <= delta at minimum total cost. Cheap contexts
(e.g. pairs) may leave f unidentified (f not in the row space of their restriction maps plus the constant), so their
data alone can only bound <f, p>; expensive contexts (a triple, the full joint) can identify it.

One valid stopping rule for every policy (so policies differ only in WHERE they sample):
  robust bounds  for each sampled context c with n_c draws, the true margin lies in the L1 ball of radius
                 eps_c = sqrt(2 / n_c * log((2^{k_c} - 2) / delta_c)) around the empirical margin (Weissman et al.);
                 delta_c = delta / (|C| * 2 j^2) at the j-th check (union bound over contexts and checks: anytime-valid).
                 The range of <f, p> over joints whose margins lie in every ball is a linear programme [lo, hi];
                 stop with '>=' if lo > t, '<' if hi < t.
Policies (no gradient steps anywhere):
  full-only      sample only the full joint.
  c-opt          cost-weighted c-optimal design over ALL contexts (the Track-and-Stop / optimal-design baseline): the
                 minimum-cost linear unbiased estimator of <f, p>, min_lambda sum_c sqrt(cost_c * lambda_c' S_c lambda_c)
                 s.t. sum_c R_c' lambda_c = f + kappa 1, solved by the eta-trick (closed-form weighted least squares);
                 sample shares ~ sqrt(lambda_c' S_c lambda_c / cost_c), tracked; plug-in p from IPF on the data so far.
  cheap-only     equal shares over the cheap contexts (can stop only if the identified set of the cheap cover excludes t).
  bounds-first   (proposed) cheap contexts first; stop as soon as the robust bounds decide; if the cheap data show that
                 the identified set itself straddles t (futility: the plug-in identified set contains t; switching never
                 affects validity, only cost), switch to the CHEAPEST set of contexts that identifies f (exact search over subsets,
                 rank test) and sample it by c-optimal design, keeping all data in the stopping rule.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
from scipy.optimize import linprog


# ------------------------------------------------------------------------------------------- covers
def cells(n):
    return np.array(list(itertools.product((0, 1), repeat=n)))


def restriction(n, ctx):
    X = cells(n); sub = list(itertools.product((0, 1), repeat=len(ctx)))
    R = np.zeros((len(sub), len(X)))
    for j, x in enumerate(X):
        R[sub.index(tuple(x[list(ctx)])), j] = 1.0
    return R


def identified(n, ctxs, f):
    """f is identified by the contexts iff f lies in span(rows of their restriction maps, 1)"""
    if not ctxs:
        return False
    M = np.vstack([restriction(n, c) for c in ctxs] + [np.ones((1, 2 ** n))])
    res = f - M.T @ np.linalg.lstsq(M.T, f, rcond=None)[0]
    return float(np.abs(res).max()) < 1e-9


def cheapest_identifying(n, contexts, costs, f, have=()):
    """exact: the minimum-cost set of additional contexts S such that have + S identifies f"""
    idx = [i for i in range(len(contexts)) if i not in have]; best = None
    for r in range(0, len(idx) + 1):
        for S in itertools.combinations(idx, r):
            c = sum(costs[i] for i in S)
            if best is not None and c >= best[0]:
                continue
            if identified(n, [contexts[i] for i in list(have) + list(S)], f):
                best = (c, S)
    return best


# ------------------------------------------------------------------------------------------- robust bounds
def eps_l1(n_c, k, delta_c):
    return np.sqrt(2.0 / n_c * np.log((2 ** k - 2) / delta_c)) if n_c > 0 else np.inf


def robust_bounds(n, Rs, ms, eps, f):
    """[min, max] of <f, p> over joints p with ||R_c p - m_c||_1 <= eps_c for every sampled context"""
    K = 2 ** n; use = [i for i in range(len(Rs)) if np.isfinite(eps[i])]
    nv = K + sum(2 * Rs[i].shape[0] for i in use)
    Aeq = [np.concatenate([np.ones(K), np.zeros(nv - K)])]; beq = [1.0]; Aub, bub = [], []; off = K
    for i in use:
        k = Rs[i].shape[0]
        for r in range(k):
            row = np.zeros(nv); row[:K] = Rs[i][r]; row[off + r] = -1; row[off + k + r] = 1; Aeq.append(row); beq.append(ms[i][r])
        row = np.zeros(nv); row[off:off + 2 * k] = 1; Aub.append(row); bub.append(eps[i]); off += 2 * k
    c = np.concatenate([f, np.zeros(nv - K)]); out = []
    for sgn in (1, -1):
        r = linprog(sgn * c, A_ub=np.array(Aub) if Aub else None, b_ub=np.array(bub) if bub else None, A_eq=np.array(Aeq), b_eq=np.array(beq),
                    bounds=(0, None), method="highs")
        out.append(sgn * r.fun if r.status == 0 else np.nan)
    return out[0], out[1]


# ------------------------------------------------------------------------------------------- c-optimal design
def ipf_joint(n, Rs, counts, iters=300):
    p = np.full(2 ** n, 1.0 / 2 ** n)
    for _ in range(iters):
        for R, m in zip(Rs, counts):
            if m.sum() == 0:
                continue
            tgt = (m + 0.5) / (m + 0.5).sum(); cur = R @ p
            p = p * (R.T @ (tgt / np.maximum(cur, 1e-300)))
    return p / p.sum()


def c_optimal(n, Rs, costs, f, p, iters=60):
    """cost shares of the minimum-cost linear unbiased estimator of <f, p> (eta-trick; closed-form steps).
    Returns (shares over the given contexts, variance-cost product) or (None, inf) if f is not identified."""
    ks = [R.shape[0] for R in Rs]; M = np.hstack([R.T for R in Rs] + [np.ones((2 ** n, 1))])
    S = [np.diag(R @ p) - np.outer(R @ p, R @ p) + 1e-9 * np.eye(R.shape[0]) for R in Rs]
    a = np.ones(len(Rs))
    for _ in range(iters):
        Winv = np.zeros((M.shape[1], M.shape[1])); o = 0
        for i, k in enumerate(ks):
            Winv[o:o + k, o:o + k] = np.linalg.inv(S[i]) * a[i] / costs[i]; o += k
        Winv[-1, -1] = 1e6                                             # kappa is free
        G = M @ Winv @ M.T; lam = Winv @ M.T @ np.linalg.lstsq(G, f, rcond=None)[0]
        if np.abs(M @ lam - f).max() > 1e-6:
            return None, np.inf
        v = []; o = 0
        for i, k in enumerate(ks):
            l = lam[o:o + k]; v.append(float(l @ S[i] @ l)); o += k
        a = np.sqrt(np.maximum(np.array(costs) * np.array(v), 1e-18))
    v = np.array(v); share = np.sqrt(v * np.array(costs)); share = share / share.sum()          # share of COST budget
    return share, float(np.sum(np.sqrt(np.array(costs) * v)) ** 2)


# ------------------------------------------------------------------------------------------- the sequential loop
@dataclass
class Problem:
    n: int
    contexts: List[tuple]
    costs: List[float]
    cheap: List[int]
    f: np.ndarray
    t: float


def run(prob: Problem, p_true, policy, delta=0.05, batch_cost=100.0, cap=60000.0, growth=1.25, rng=None):
    rng = rng or np.random.default_rng(0)
    n, C, cost = prob.n, prob.contexts, np.array(prob.costs, float)
    Rs = [restriction(n, c) for c in C]; margins = [R @ p_true for R in Rs]
    counts = [np.zeros(R.shape[0]) for R in Rs]; spent = np.zeros(len(C)); total = 0.0; j = 0; next_check = batch_cost
    phase = "cheap"; active = list(prob.cheap) if policy in ("cheap-only", "bounds-first") else (
        [len(C) - 1] if policy == "full-only" else list(range(len(C))))
    shares = None; switched_at = None
    while total < cap:
        # choose shares for the active set
        cheap_id = policy == "bounds-first" and phase == "cheap" and identified(n, [C[i] for i in prob.cheap], prob.f)
        if policy in ("c-opt",) or (policy == "bounds-first" and (phase == "identify" or cheap_id)):
            p_hat = ipf_joint(n, [Rs[i] for i in range(len(C))], counts)
            sh, _ = c_optimal(n, [Rs[i] for i in active], cost[active], prob.f, p_hat)
            shares = sh if sh is not None else np.full(len(active), 1.0 / len(active))
        else:
            shares = np.full(len(active), 1.0 / len(active))
        # spend one batch by tracking the cost shares (vectorised: one multinomial draw per context)
        want = shares * (spent[active].sum() + batch_cost) - spent[active]
        want = np.maximum(want, 0); want = want / want.sum() * batch_cost if want.sum() > 0 else shares * batch_cost
        for i, wc in zip(active, want):
            k = int(np.floor(wc / cost[i] + rng.random()))
            if k > 0:
                counts[i] += rng.multinomial(k, margins[i]); spent[i] += k * cost[i]; total += k * cost[i]
        if total < next_check:
            continue
        j += 1; next_check *= growth
        sampled = [i for i in range(len(C)) if counts[i].sum() > 0]
        dc = delta / (len(C) * 2 * j * j)
        eps = [eps_l1(counts[i].sum(), Rs[i].shape[0], dc) if i in sampled else np.inf for i in range(len(C))]
        ms = [counts[i] / max(counts[i].sum(), 1) for i in range(len(C))]
        lo, hi = robust_bounds(n, Rs, ms, eps, prob.f)
        if lo > prob.t or hi < prob.t:
            return dict(stopped=True, decision=bool(lo > prob.t), cost=total, checks=j, switched_at=switched_at,
                        spent={str(C[i]): float(spent[i]) for i in range(len(C))})
        if policy == "bounds-first" and phase == "cheap" and not identified(n, [C[i] for i in prob.cheap], prob.f):
            p_hat = ipf_joint(n, [Rs[i] for i in prob.cheap], [counts[i] for i in prob.cheap])               # consistent plug-in
            plo, phi = robust_bounds(n, [Rs[i] for i in prob.cheap], [Rs[i] @ p_hat for i in prob.cheap], [0.0] * len(prob.cheap), prob.f)
            if plo < prob.t < phi:                                     # the estimated identified set straddles t: cheap is futile
                best = cheapest_identifying(n, C, list(cost), prob.f, have=tuple(prob.cheap))
                active = sorted(set(prob.cheap) | set(best[1])); phase = "identify"; switched_at = total
    return dict(stopped=False, decision=None, cost=total, checks=j, switched_at=switched_at,
                spent={str(C[i]): float(spent[i]) for i in range(len(C))})
