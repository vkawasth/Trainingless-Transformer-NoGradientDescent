"""A reward loop with all three layers: policy and value iteration on beliefs built from geometry, under scheme rules.

Stream. Each period t, S sources report on X topics (binomial counts). Each topic carries a hidden contamination state
z_{x,t} in {0, 1} (a Markov chain: clean -> looped with q01, looped -> clean with q10). A looped topic carries a
source-interaction vector (a planted loop) that also biases the target source's lean.
Layers.
  geometry      the evidence per topic and period is its loop statistic Q = weighted residual sum of squares of the
                additive fit (first-order jet D'_x of region consistency). Clean: Q ~ chi2(S - 1). Looped: noncentral
                chi2(S - 1, lam) with lam = the Fisher norm^2 of the loop (twice the holonomy information it carries).
                lam is estimated from a burn-in stream by EM with a moment M-step (closed form).
  probability   a Bayes (hidden-Markov) filter per topic: belief b = P(looped | Q_1..Q_t).
  scheme rules  each period the region (topics kept) must GLUE (additive-model p >= 0.05) and the target outcome must
                be RIGHT (the lean estimated on the region within 2 se of the truth); a violated rule costs P.
Reward per period: + v per clean topic kept, - c per looped topic kept, - kappa per topic switched in or out, - P per
violated rule. (Data kept is valuable; contamination and churn are costly.)
Control. The rules couple the topics; a Lagrange multiplier mu (price per looped topic kept) decouples them (the
restless-bandit / Whittle-style relaxation). Per topic, the state is (belief bin, kept last period), the action is
keep / drop; the belief kernel is computed by quadrature over the chi2 mixture. VALUE ITERATION and POLICY
ITERATION (exact linear solves) give the same policy. mu is chosen by a grid search on training streams. No gradient
steps anywhere.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from . import lattice_dp as L


# ------------------------------------------------------------------------------------------- belief filter
def lik(Q, df, lam):
    return stats.chi2.pdf(Q, df), stats.ncx2.pdf(Q, df, lam)


def predict(b, q01, q10):
    return b * (1 - q10) + (1 - b) * q01


def update(b_pred, Q, df, lam):
    f0, f1 = lik(Q, df, lam)
    num = b_pred * f1; den = num + (1 - b_pred) * f0
    return np.where(den > 0, num / np.maximum(den, 1e-300), b_pred)


def em_lambda(Qs, df, iters=200):
    """mixture pi0 chi2(df) + pi1 ncx2(df, lam): EM with the moment update lam = E[Q | looped] - df"""
    Qs = np.asarray(Qs, float); pi1 = 0.2; lam = max(np.percentile(Qs, 95) - df, 1.0)
    for _ in range(iters):
        f0, f1 = lik(Qs, df, lam); r = pi1 * f1 / np.maximum(pi1 * f1 + (1 - pi1) * f0, 1e-300)
        pi1 = float(r.mean()); lam = max(float((r * Qs).sum() / max(r.sum(), 1e-12) - df), 0.5)
    return lam, pi1


# ------------------------------------------------------------------------------------------- belief MDP per topic
def belief_kernel(grid, df, lam, q01, q10, nq=400):
    """K[i, j] = P(next belief in bin j | current belief grid[i]) after one period (predict, observe Q, update)"""
    B = len(grid); K = np.zeros((B, B)); u = (np.arange(nq) + 0.5) / nq
    Q0 = stats.chi2.ppf(u, df); Q1 = stats.ncx2.ppf(u, df, lam)
    for i, b in enumerate(grid):
        bp = predict(b, q01, q10)
        for Qs, w in ((Q0, 1 - bp), (Q1, bp)):
            nb = update(bp, Qs, df, lam)
            pos = nb * (B - 1); lo = np.floor(pos).astype(int); hi = np.minimum(lo + 1, B - 1); fr = pos - lo
            np.add.at(K[i], lo, w * (1 - fr) / nq); np.add.at(K[i], hi, w * fr / nq)
    return K / K.sum(1, keepdims=True)


def rewards(grid, v, c, kappa, mu):
    """R[i, u_prev, a]: expected reward of action a (1 keep) at belief grid[i] given the previous inclusion u_prev"""
    R = np.zeros((len(grid), 2, 2))
    keep = (1 - grid) * v - grid * (c + mu)
    for up in (0, 1):
        for a in (0, 1):
            R[:, up, a] = a * keep - kappa * (a != up)
    return R


def value_iteration(K, R, gamma=0.95, tol=1e-9, max_iter=10000):
    B = K.shape[0]; V = np.zeros((B, 2)); it = 0
    while it < max_iter:
        it += 1
        Qa = R + gamma * np.stack([np.stack([K @ V[:, a] for a in (0, 1)], 1)] * 2, 1)   # [i, up, a]: next state (j, a)
        Vn = Qa.max(2)
        if np.max(np.abs(Vn - V)) < tol:
            V = Vn; break
        V = Vn
    return Qa.argmax(2), V, it


def policy_iteration(K, R, gamma=0.95, max_iter=100):
    """exact policy evaluation (linear solve over the 2B states) + greedy improvement"""
    B = K.shape[0]; pol = np.zeros((B, 2), int); it = 0
    while it < max_iter:
        it += 1
        # states s = (i, up); next state (j, a) with a = pol[i, up]
        P = np.zeros((2 * B, 2 * B)); r = np.zeros(2 * B)
        for up in (0, 1):
            a = pol[:, up]; rows = np.arange(B) + up * B
            r[rows] = R[np.arange(B), up, a]
            for i in range(B):
                P[rows[i], a[i] * B:(a[i] + 1) * B] = K[i]
        Vflat = np.linalg.solve(np.eye(2 * B) - gamma * P, r); V = np.stack([Vflat[:B], Vflat[B:]], 1)
        Qa = R + gamma * np.stack([np.stack([K @ V[:, a] for a in (0, 1)], 1)] * 2, 1)
        new = Qa.argmax(2)
        if np.array_equal(new, pol):
            break
        pol = new
    return pol, V, it


# ------------------------------------------------------------------------------------------- the stream
class Stream:
    def __init__(self, S=50, X=60, T=100, q01=0.02, q10=0.10, amp=0.6, amp_star=1.0, n_mean=30, seed=0):
        r = np.random.default_rng(seed); self.r = r
        self.S, self.X, self.T, self.q01, self.q10 = S, X, T, q01, q10
        self.a = r.normal(0, 0.5, S); self.b = r.normal(-0.5, 0.6, X)
        self.g = np.zeros((X, S))
        for x in range(X):
            srcs = r.choice(np.arange(1, S), 20, replace=False); self.g[x, srcs] = r.choice([-1, 1], 20) * amp; self.g[x, 0] = amp_star
        z = np.zeros((T, X), int); z[0] = r.random(X) < q01 / (q01 + q10)
        for t in range(1, T):
            u = r.random(X); z[t] = np.where(z[t - 1] == 1, u >= q10, u < q01)
        self.z = z; self.n_mean = n_mean; self.true_lean = self.a[0] - self.a.mean()

    def period(self, t):
        eta = self.a[:, None] + self.b[None, :] + (self.g * self.z[t][:, None]).T
        n = self.r.poisson(self.n_mean, (self.S, self.X)) + 5; k = self.r.binomial(n, 1 / (1 + np.exp(-eta)))
        y = np.log((k + 0.5) / (n - k + 0.5)); w = 1 / (1 / (k + 0.5) + 1 / (n - k + 0.5))
        return y, w


def loop_stats(y, w):
    """per-topic loop statistic Q_x (the first-order jet) from the additive fit on all topics"""
    return L.jets(y, w)["d1"]


def rules_ok(y, w, U, true_lean):
    if len(U) < 2:
        return False, False
    p, D, df = L.glue_p(y, w, U)
    r, lean = L.target_stability(y, w, U, 0, U[0], true_lean)
    return p >= 0.05, r < 2.0                               # glue; lean within 2 se of the truth
