"""Decompose the oracle 0.04: true subspace span(vT, mu_w, a, b) with (i) linear ridge, (ii) RBF kernel on the projections."""
import os, sys, numpy as np
_h = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_h, "..")); sys.path.insert(0, os.path.join(_h, "../../.."))
import run as R
S = R.S
def basis(seed):
    fam = S.CATE_FAMILIES[seed % len(S.CATE_FAMILIES)]; rng = S._rng(seed, "nuisance_parameters")
    mu = S._unit_vector(rng, R.D, 14); vT = S._unit_vector(rng, R.D, 14); cp = S._cate_parameters(seed, R.D, fam)
    V = [vT, mu, np.roll(mu, 1), cp["a"], cp["b"]] + (list(cp["hidden_w"]) if "hidden_w" in cp else [])
    Q, _ = np.linalg.qr(np.array(V).T); return Q
def krr_loo(K, y, Kq, lams=np.logspace(-3, 4, 29)):
    m = y.mean(); yc = y - m; ev, V = np.linalg.eigh(K); ev = np.maximum(ev, 0); yt = V.T @ yc; best = None
    for lam in lams:
        d = ev / (ev + lam); loo = (yc - V @ (d * yt)) / (1 - (V ** 2) @ d); e = np.mean(loo ** 2)
        if best is None or e < best[0]: best = (e, lam)
    return best[0], m + Kq @ (V @ (yt / (ev + best[1])))
def rbf(A, B, g):
    return np.exp(-g * ((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T))
if __name__ == "__main__":
    res = []
    for seed in range(int(os.environ.get("NW", 10))):
        W = R.world(seed); c, q = W["ctx"], W["qry"]; X = W["X"]; y = (W["Y"][c] == W["T"][c]).astype(float)
        Lo, Uo = R.manski(W["q"].mean(1)); Q = basis(seed); Fc, Fq = X[c] @ Q, X[q] @ Q
        rm = lambda s: float(np.sqrt(np.mean((np.clip(s, 0, 1) - Uo) ** 2)))
        lin = rm(krr_loo(Fc @ Fc.T, y, Fq @ Fc.T)[1])
        best = min((krr_loo(rbf(Fc, Fc, g) + Fc @ Fc.T / Q.shape[1], y, rbf(Fq, Fc, g) + Fq @ Fc.T / Q.shape[1]) for g in (0.05, 0.1, 0.2, 0.4)), key=lambda t: t[0])
        nl = rm(best[1]); res.append((lin, nl, Q.shape[1])); print(seed, W["family"][:10], f"dim {Q.shape[1]} subspace-linear {lin:.3f} subspace-RBF {nl:.3f}", flush=True)
    print("mean", np.round(np.mean(res, 0), 4))
