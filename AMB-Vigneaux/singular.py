"""Singular statistics and optimal learning: component [L].

Where a model is singular (the Fisher information degenerates on a set of true parameters), the Bayes free energy is
    F_n = n S_n + lambda log n - (m - 1) log log n + O_p(1),
with lambda the real log canonical threshold (learning coefficient, lambda <= d/2) and m its multiplicity, and the Bayes
generalisation error is lambda / n (Watanabe). Plug-in maximum likelihood is not asymptotically normal there; posterior
averaging is the optimal learner. Tools (no gradient steps; random-walk Metropolis only):
  flop_posterior   Dirichlet posterior on a table -> probability of each forced (even, odd) pair of the support toggle
                   and the posterior of its radius (replaces the argmin, which flips near a node)
  tempered_chain   random-walk Metropolis on a log-posterior at inverse temperature beta
  wbic_lambda      WBIC = E_{beta1}[n L_n] at beta1 = 1/log n, and Watanabe's estimate
                   lambda_hat = (E_{beta1}[n L_n] - E_{beta2}[n L_n]) / (1/beta1 - 1/beta2)
  mix_loglik       two-component shift mixture r ~ (1 - a) N(0, s2) + a N(b, s2)   (singular at a b = 0: K ~ a^2 b^2,
                   lambda = 1/2, m = 2, against the regular d/2 = 1)
  lca_*            latent class model on binary vectors (a secant variety of the independence model; singular)
"""
from __future__ import annotations

import itertools
from typing import Callable

import numpy as np


# ------------------------------------------------------------------------------------------- flop posterior
def flop_posterior(counts, ev, od, B=4000, rng=None, prior=0.5):
    rng = rng or np.random.default_rng(0)
    P = rng.dirichlet(np.asarray(counts, float) + prior, size=B)
    e = np.array(ev)[np.argmin(P[:, ev], 1)]; o = np.array(od)[np.argmin(P[:, od], 1)]
    pairs = {}
    for a, b in zip(e, o):
        pairs[(int(a), int(b))] = pairs.get((int(a), int(b)), 0) + 1
    mass = P[np.arange(B), e] + P[np.arange(B), o]
    rad = -np.log1p(-mass)
    return dict(pairs={k: v / B for k, v in sorted(pairs.items(), key=lambda kv: -kv[1])},
                radius_mean=float(rad.mean()), radius_lo=float(np.quantile(rad, .025)), radius_hi=float(np.quantile(rad, .975)))


# ------------------------------------------------------------------------------------------- tempered sampling
def tempered_chain(loglik: Callable, logprior: Callable, x0, beta, steps=20000, burn=5000, scale=0.1, rng=None, adapt=True):
    rng = rng or np.random.default_rng(0)
    x = np.array(x0, float); ll = loglik(x); lp = logprior(x); out = []; acc = 0; s = scale
    for t in range(steps):
        y = x + s * rng.normal(size=x.shape); lpy = logprior(y)
        if np.isfinite(lpy):
            lly = loglik(y)
            if np.log(rng.random()) < beta * (lly - ll) + (lpy - lp):
                x, ll, lp = y, lly, lpy; acc += 1
        if adapt and t < burn and t % 200 == 199:
            s *= 1.25 if acc / 200 > 0.3 else 0.8; acc = 0
        if t >= burn:
            out.append(-ll)                                   # n L_n = minus total log-likelihood
    return np.array(out)


def wbic_lambda(loglik, logprior, x0, n, steps=20000, burn=5000, scale=0.1, rng=None):
    b1 = 1 / np.log(n); b2 = 1.5 / np.log(n)
    e1 = tempered_chain(loglik, logprior, x0, b1, steps, burn, scale, rng).mean()
    e2 = tempered_chain(loglik, logprior, x0, b2, steps, burn, scale, rng).mean()
    return dict(wbic=float(e1), lam=float((e1 - e2) / (1 / b1 - 1 / b2)))


# ------------------------------------------------------------------------------------------- shift mixture
def mix_loglik(r, s2):
    c = -0.5 * np.log(2 * np.pi * s2)
    def ll(th):
        a, b = th
        if not (0 <= a <= 1):
            return -np.inf
        f0 = np.exp(c - 0.5 * r ** 2 / s2); f1 = np.exp(c - 0.5 * (r - b) ** 2 / s2)
        return float(np.log((1 - a) * f0 + a * f1).sum())
    return ll


def mix_logprior(bscale):
    def lp(th):
        a, b = th
        return -0.5 * (b / bscale) ** 2 if 0 <= a <= 1 else -np.inf
    return lp


# ------------------------------------------------------------------------------------------- latent class model
def lca_probs(th, K, d):
    """th = (K-1 class logits, K*d item logits) -> law on {0,1}^d (2^d cells, lexicographic)"""
    w = np.exp(np.concatenate([[0.0], th[:K - 1]])); w /= w.sum()
    q = 1 / (1 + np.exp(-th[K - 1:].reshape(K, d)))
    X = np.array(list(itertools.product((0, 1), repeat=d)))
    lik = np.prod(np.where(X[None, :, :] == 1, q[:, None, :], 1 - q[:, None, :]), axis=2)   # K x cells
    return w @ lik


def lca_loglik(counts, K, d):
    c = np.asarray(counts, float)
    return lambda th: float(c @ np.log(np.maximum(lca_probs(th, K, d), 1e-300)))


def lca_logprior(th):
    return float(-0.5 * (th / 3.0) @ (th / 3.0))


def lca_em(counts, K, d, starts=20, iters=500, rng=None):
    rng = rng or np.random.default_rng(0)
    X = np.array(list(itertools.product((0, 1), repeat=d))); c = np.asarray(counts, float); best = None
    for _ in range(starts):
        w = rng.dirichlet(np.ones(K)); q = rng.uniform(0.2, 0.8, (K, d))
        for _ in range(iters):
            lik = np.prod(np.where(X[None] == 1, q[:, None], 1 - q[:, None]), axis=2) * w[:, None]
            post = lik / lik.sum(0, keepdims=True)
            wc = post * c[None]
            w = wc.sum(1) / c.sum(); q = np.clip((wc @ X) / wc.sum(1, keepdims=True), 1e-6, 1 - 1e-6)
        p = w @ np.prod(np.where(X[None] == 1, q[:, None], 1 - q[:, None]), axis=2)
        ll = float(c @ np.log(p))
        if best is None or ll > best[0]:
            best = (ll, w, q)
    ll, w, q = best
    th = np.concatenate([np.log(w[1:] / w[0]), np.log(q / (1 - q)).ravel()])
    return dict(loglik=ll, theta=th, probs=lca_probs(th, K, d))


def log_evidence_ti(loglik, logprior, x0, betas=None, steps=8000, burn=2000, scale=0.05, rng=None):
    """log marginal likelihood by thermodynamic integration: log Z = int_0^1 E_beta[log L] d beta (proper prior)"""
    rng = rng or np.random.default_rng(0)
    betas = np.concatenate([[0.0], np.geomspace(1e-4, 1.0, 24)]) if betas is None else np.asarray(betas)
    E = []
    for b in betas:
        E.append(-tempered_chain(loglik, logprior, x0, b, steps, burn, scale, rng).mean())
    E = np.array(E)
    return float(np.trapezoid(E, betas)), betas, E


def wbic_lambda_rep(loglik, logprior, x0, n, chains=4, **kw):
    """wbic_lambda over independent chains: means and Monte-Carlo standard errors"""
    rng = kw.pop("rng", None) or np.random.default_rng(0)
    R = [wbic_lambda(loglik, logprior, x0, n, rng=np.random.default_rng(rng.integers(1 << 31)), **kw) for _ in range(chains)]
    w = np.array([r["wbic"] for r in R]); l = np.array([r["lam"] for r in R])
    return dict(wbic=float(w.mean()), wbic_se=float(w.std(ddof=1) / np.sqrt(chains)), lam=float(l.mean()), lam_se=float(l.std(ddof=1) / np.sqrt(chains)))


# ------------------------------------------------------------------------------------------- categorical latent classes
def cat_cells(d, V):
    return np.array(list(itertools.product(range(V), repeat=d)))


def cat_lca_probs(th, K, d, V):
    """th = (K-1 class logits, K*d*(V-1) item logits) -> law on V^d cells. One parent symbol, d children (items)."""
    w = np.exp(np.concatenate([[0.0], th[:K - 1]])); w /= w.sum()
    L = th[K - 1:].reshape(K, d, V - 1); L = np.concatenate([np.zeros((K, d, 1)), L], axis=2)
    Q = np.exp(L - L.max(2, keepdims=True)); Q /= Q.sum(2, keepdims=True)                 # K x d x V
    X = cat_cells(d, V)
    lik = np.ones((K, len(X)))
    for j in range(d):
        lik *= Q[:, j, X[:, j]]
    return w @ lik


def cat_lca_loglik(counts, K, d, V):
    c = np.asarray(counts, float)
    return lambda th: float(c @ np.log(np.maximum(cat_lca_probs(th, K, d, V), 1e-300)))


def cat_lca_em(counts, K, d, V, starts=10, iters=400, rng=None):
    rng = rng or np.random.default_rng(0); X = cat_cells(d, V); c = np.asarray(counts, float); best = None
    for _ in range(starts):
        w = rng.dirichlet(np.ones(K)); Q = rng.dirichlet(np.ones(V), size=(K, d))
        for _ in range(iters):
            lik = np.ones((K, len(X)))
            for j in range(d):
                lik *= Q[:, j, X[:, j]]
            post = lik * w[:, None]; post /= np.maximum(post.sum(0, keepdims=True), 1e-300); wc = post * c[None]
            w = wc.sum(1) / c.sum()
            for j in range(d):
                for v in range(V):
                    Q[:, j, v] = wc[:, X[:, j] == v].sum(1)
                Q[:, j] = np.clip(Q[:, j] / np.maximum(Q[:, j].sum(1, keepdims=True), 1e-300), 1e-8, 1)
        lik = np.ones((K, len(X)))
        for j in range(d):
            lik *= Q[:, j, X[:, j]]
        p = w @ lik; ll = float(c @ np.log(np.maximum(p, 1e-300)))
        if best is None or ll > best[0]:
            best = (ll, w, Q)
    ll, w, Q = best; w = np.maximum(w, 1e-8)
    th = np.concatenate([np.log(w[1:] / w[0]), (np.log(Q[:, :, 1:]) - np.log(Q[:, :, :1])).ravel()])
    return dict(loglik=ll, theta=th, probs=cat_lca_probs(th, K, d, V))
