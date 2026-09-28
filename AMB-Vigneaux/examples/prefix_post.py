"""Prefix posteriors of a tree grammar (inside-outside with unobserved leaves = 1).

For every sequence x and position t, given only x_<t:
  pred[n,t,:]      P(x_t | x_<t)
  anc[l][n,t,:]    P(B_l(t) | x_<t) for the level-l ancestor of leaf t, l = 1..L
"""
import numpy as np


def load_true(meta):
    V, A, L = meta["nsym"], meta["nleaf"], meta["depth"]
    P = {}
    for l in range(1, L + 1):
        c = A if l == 1 else V
        t = np.zeros((V, c, c))
        for s, r in meta["rules_all"][str(l)].items():
            for x, y in r:
                t[int(s), x, y] += 1.0 / len(r)
        P[l] = t
    return P


def prefix_posteriors(X, P, prior=None, chunk=2000):
    N, S = X.shape
    L = len(P); A = P[1].shape[1]; Vr = P[L].shape[0]
    prior = np.full(Vr, 1.0 / Vr) if prior is None else prior
    pred = np.zeros((N, S, A)); anc = {l: np.zeros((N, S, P[l].shape[0])) for l in range(1, L + 1)}
    for s0 in range(0, N, chunk):
        Xc = X[s0:s0 + chunk]; n = len(Xc)
        for t in range(S):
            b = np.ones((n, S, A)); b[:, :t, :] = 0
            b[np.arange(n)[:, None], np.arange(t)[None, :], Xc[:, :t]] = 1.0
            beta = {0: b}
            for l in range(1, L + 1):
                lo, hi = beta[l - 1][:, 0::2], beta[l - 1][:, 1::2]
                beta[l] = np.einsum("bxy,njx,njy->njb", P[l], lo, hi)
            alpha = {L: np.broadcast_to(prior, (n, 1, Vr)).copy()}
            for l in range(L, 0, -1):
                lo, hi = beta[l - 1][:, 0::2], beta[l - 1][:, 1::2]
                al = np.einsum("njb,bxy,njy->njx", alpha[l], P[l], hi)
                ar = np.einsum("njb,bxy,njx->njy", alpha[l], P[l], lo)
                a = np.empty((n, 2 * al.shape[1], al.shape[2])); a[:, 0::2] = al; a[:, 1::2] = ar
                alpha[l - 1] = a
            p = alpha[0][:, t]; pred[s0:s0 + n, t] = p / p.sum(1, keepdims=True)
            for l in range(1, L + 1):
                q = alpha[l][:, t >> l] * beta[l][:, t >> l]
                anc[l][s0:s0 + n, t] = q / q.sum(1, keepdims=True)
    return pred, anc


def boundary_level(t):
    """level of the lowest common ancestor of leaves t-1 and t (0 for t = 0)."""
    return 0 if t == 0 else int(t ^ (t - 1)).bit_length()
