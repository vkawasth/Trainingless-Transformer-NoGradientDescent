"""Isotonic conditional law (Arnold & Ziegel 2025; for binary Y it is isotonic regression under a partial order, as in IDR)
on an order JUSTIFIED by the target's structure: s = e m1 + (1 - e)(1 - m0), m_t = P(Y=1 | T=t, x), so s is
increasing in m1 and decreasing in m0. Order on xi = (m1_hat, -m0_hat) (product order).
Indices are out-of-sample for every training unit: leave-one-out kernel-ridge score within its own arm; the other arm's
fit never saw it. Exact weighted L2 projection onto the isotonic cone of a G x G quantile grid by Dykstra's alternating
PAVA over rows and columns (converges to the exact projection; no gradients). G and a blend with the linear backbone by CV."""
import os, sys, numpy as np
from isotonic_link import R, loo_index
from sklearn.isotonic import IsotonicRegression

def pava_w(v, w):
    out = v.copy(); m = w > 0
    if m.sum() > 1: out[m] = IsotonicRegression().fit(np.arange(m.sum()), v[m], sample_weight=w[m]).predict(np.arange(m.sum()))
    return out

def iso2d(Ysum, Wt, iters=500):
    """weighted isotonic (nondecreasing in both axes) projection of cell means; empty cells filled by order bounds"""
    G = Ysum.shape[0]; M = np.where(Wt > 0, Ysum / np.maximum(Wt, 1e-12), 0.0); x = M.copy(); p = np.zeros_like(x); q = np.zeros_like(x)
    for _ in range(iters):
        y = np.array([pava_w((x + p)[i], Wt[i]) for i in range(G)]); p = x + p - y
        xn = np.array([pava_w((y + q)[:, j], Wt[:, j]) for j in range(G)]).T; q = y + q - xn
        if np.abs(xn - x)[Wt > 0].max(initial=0) < 1e-9: x = xn; break
        x = xn
    F = x.copy(); occ = Wt > 0
    for i in range(G):
        for j in range(G):
            if not occ[i, j]:
                lo = x[:i + 1, :j + 1][occ[:i + 1, :j + 1]]; hi = x[i:, j:][occ[i:, j:]]
                a = lo.max() if lo.size else (hi.min() if hi.size else M[occ].mean()); b = hi.min() if hi.size else a
                F[i, j] = 0.5 * (a + b)
    return F

def grid_fit(u, v, y, uq, vq, G):
    eu = np.quantile(u, np.linspace(0, 1, G + 1)[1:-1]); ev = np.quantile(v, np.linspace(0, 1, G + 1)[1:-1])
    iu, iv = np.searchsorted(eu, u), np.searchsorted(ev, v); S = np.zeros((G, G)); Wt = np.zeros((G, G))
    np.add.at(S, (iu, iv), y); np.add.at(Wt, (iu, iv), 1.0); F = iso2d(S, Wt)
    return F[np.searchsorted(eu, uq), np.searchsorted(ev, vq)]

def indices(W, idx_tr, idx_q):
    """m1_hat, m0_hat: per-arm linear kernel-ridge scores; LOO for a unit's own arm, plain prediction for the other arm"""
    X = W["X"]; T = W["T"]; Y = W["Y"].astype(float); m = {}
    for t in (0, 1):
        A = idx_tr[T[idx_tr] == t]; B = np.concatenate([idx_tr[T[idx_tr] != t], idx_q])
        loo, pred = loo_index(X[A], Y[A], X[B]); m[t] = dict(zip(A, loo)); m[t].update(zip(B, pred))
    f = lambda ids: (np.array([m[1][i] for i in ids]), -np.array([m[0][i] for i in ids]))
    return f(idx_tr), f(idx_q)

def icl_s(W, Gs=(4, 6, 8, 12), k=5):
    c, q = W["ctx"], W["qry"]; y = (W["Y"][c] == W["T"][c]).astype(float)
    (u, v), (uq, vq) = indices(W, c, q); zl, zlq = loo_index(W["X"][c], y, W["X"][q])
    f = np.random.default_rng(0).integers(0, k, len(c)); best = None
    for G in Gs:
        for a in np.linspace(0, 1, 6):
            err = 0.0
            for i in range(k):
                tr, te = f != i, f == i; pi = grid_fit(u[tr], v[tr], y[tr], u[te], v[te], G)
                err += np.sum((a * pi + (1 - a) * np.clip(zl[te], 0, 1) - y[te]) ** 2)
            if best is None or err < best[0]: best = (err, G, a)
    _, G, a = best; pq = grid_fit(u, v, y, uq, vq, G)
    return a * pq + (1 - a) * np.clip(zlq, 0, 1), np.clip(zlq, 0, 1), pq, (G, a)

if __name__ == "__main__":
    res = []
    for seed in range(int(os.environ.get("NW", 40))):
        W = R.world(seed); Lo, Uo = R.manski(W["q"].mean(1)); rm = lambda s: float(np.sqrt(np.mean((s - Uo) ** 2)))
        bl, lin, pure, ga = icl_s(W); r = (rm(lin), rm(pure), rm(bl)); res.append(r)
        print(seed, W["family"][:10], "linear %.3f | ICL on (m1, -m0) %.3f | CV blend %.3f" % r, "(G, a) =", ga, flush=True)
    print("mean linear %.4f ICL %.4f blend %.4f" % tuple(np.mean(res, 0)))
