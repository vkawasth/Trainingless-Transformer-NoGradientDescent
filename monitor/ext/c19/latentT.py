"""Latent first-order Markov model over T waves (K latent classes, 3 observed categories): initial law, time-varying transitions,
and either one response law for all waves ('shared') or one per wave ('free'). EM over all K^T latent paths; G2 against the
saturated 3^T table with a parametric-bootstrap p-value."""
import itertools, numpy as np
K = 3

def _paths(T): return np.array(list(itertools.product(range(K), repeat=T)))

def joint(th, T):
    pi, Ts, Rs = th; P = np.zeros((K,) * T + (3,) * T)
    lat = _paths(T); obs = _paths(T)
    pl = pi[lat[:, 0]] * np.prod([Ts[t][lat[:, t], lat[:, t + 1]] for t in range(T - 1)], axis=0)            # P(latent path)
    po = np.prod([Rs[t][lat[:, t]][:, obs[:, t]] for t in range(T)], axis=0)                                    # P(obs | latent), shape (L, O)
    return pl, po

def em(N, T, shared, rng, iters=1500):
    obs = _paths(T); lat = _paths(T); n = N.sum(); Nv = N.reshape(-1)
    LO = [np.eye(K)[lat[:, t]] for t in range(T)]; OO = [np.eye(3)[obs[:, t]] for t in range(T)]
    D = lambda *s: rng.dirichlet(np.ones(s[-1]), size=s[:-1])
    pi = rng.dirichlet(np.ones(K)); Ts = [D(K, K) for _ in range(T - 1)]; R0 = D(K, 3); Rs = [R0] * T if shared else [D(K, 3) for _ in range(T)]
    old = -np.inf
    for _ in range(iters):
        pl, po = joint((pi, Ts, Rs), T); J = pl[:, None] * po; P = J.sum(0)
        W = J / np.maximum(P, 1e-300) * Nv                                                                      # expected counts (L, O)
        wl = W.sum(1)
        pi = np.bincount(lat[:, 0], weights=wl, minlength=K) / n
        Ts = []
        for t in range(T - 1):
            M = (LO[t] * wl[:, None]).T @ LO[t + 1]; Ts.append(M / np.maximum(M.sum(1, keepdims=True), 1e-300))
        Rn = [LO[t].T @ W @ OO[t] for t in range(T)]
        if shared: S = sum(Rn); Rs = [S / np.maximum(S.sum(1, keepdims=True), 1e-300)] * T
        else: Rs = [M / np.maximum(M.sum(1, keepdims=True), 1e-300) for M in Rn]
        ll = float((Nv[Nv > 0] * np.log(np.maximum(P[Nv > 0], 1e-300))).sum())
        if ll - old < 1e-8: break
        old = ll
    pl, po = joint((pi, Ts, Rs), T); P = (pl[:, None] * po).sum(0)
    return ll, P, (pi, Ts, Rs)

def fit(N, T, shared=True, seed=0, starts=20):
    rng = np.random.default_rng(seed); best = max((em(N, T, shared, rng) for _ in range(starts)), key=lambda r: r[0]); return best[1], best[2]

def G2(N, P):
    Nv = N.reshape(-1); m = Nv > 0; return 2 * float((Nv[m] * np.log(Nv[m] / (Nv.sum() * P[m]))).sum())

def n_params(T, shared): return (K - 1) + (T - 1) * K * (K - 1) + (1 if shared else T) * K * 2

def test(N, T, shared=True, B=200, seed=0, starts=20, refit_starts=4):
    P, th = fit(N, T, shared, seed, starts); g = G2(N, P); n = int(N.sum()); rng = np.random.default_rng(seed + 1); bs = []
    for _ in range(B):
        Nb = rng.multinomial(n, P).reshape(N.shape).astype(float); Pb, _ = fit(Nb, T, shared, int(rng.integers(1e9)), starts=refit_starts); bs.append(G2(Nb, Pb))
    df = 3 ** T - 1 - n_params(T, shared)
    return dict(G2=round(g, 2), df=df, p_boot=float((np.sum(np.array(bs) >= g) + 1) / (B + 1)), min_class=round(float(th[0].min()), 3), n=n)
