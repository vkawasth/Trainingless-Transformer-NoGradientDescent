"""Treatment beacon + known structure (the attachment's strategies A and C, in the form that applies here).
s(x) = P(Y=T|x) ~ 1/2 + tau(x)/2 + E[(2e - 1)(b - 1/2)]: the propensity e is well estimated (strong label T, instrument
unpenalised), so fit the varying-coefficient model  s = a + x.b1 + g (c + x.b2),  g = 2 e_hat(x, I) - 1,
as kernel ridge with K = XX' + w (gg' o XX') + w0 gg'; w = 0 is the current backbone. Ridge by closed-form LOO, (w, w0) by LOO.
At a query point the prediction is averaged over I in {0, 1} (P(I=1) = 1/2)."""
import os, sys, numpy as np
_h = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_h, "..")); sys.path.insert(0, os.path.join(_h, "../../.."))
import run as R
from oracle_subspace import krr_loo
LAMS = np.logspace(-2, 5, 29)

def beacon_s(W, ws=(0, 0.3, 1, 3, 10), w0s=(0, 10, 100), g_override=None):
    c, q = W["ctx"], W["qry"]; X = W["X"]; T = W["T"][c].astype(float); I = W["I"][c].astype(float); y = (W["Y"][c] == W["T"][c]).astype(float)
    one = lambda n: np.ones((n, 1)); Zc = np.hstack([one(len(c)), X[c], I[:, None]]); fr = (0, Zc.shape[1] - 1)
    bT = R._cv_logistic(Zc, T, free=fr)
    g = 2 * R._sig(Zc @ bT) - 1 if g_override is None else g_override[0]
    gq = [2 * R._sig(np.hstack([one(len(q)), X[q], np.full((len(q), 1), j)]) @ bT) - 1 for j in (0, 1)] if g_override is None else g_override[1]
    G = X[c] @ X[c].T; best = None
    for w in ws:
        for w0 in (w0s if w else (0,)):
            K = G + w * np.outer(g, g) * G + w0 * np.outer(g, g)
            e, _ = krr_loo(K, y, np.zeros((1, len(c))), LAMS)
            if best is None or e < best[0]: best = (e, w, w0)
    _, w, w0 = best; K = G + w * np.outer(g, g) * G + w0 * np.outer(g, g); Gq = X[q] @ X[c].T
    p = np.mean([krr_loo(K, y, Gq + w * np.outer(gj, g) * Gq + w0 * np.outer(gj, g), LAMS)[1] for gj in gq], 0)
    return np.clip(p, 0, 1), (w, w0)

if __name__ == "__main__":
    res = []
    for seed in range(int(os.environ.get("NW", 10))):
        W = R.world(seed); c, q = W["ctx"], W["qry"]; X = W["X"]; Lo, Uo = R.manski(W["q"].mean(1))
        rm = lambda s: float(np.sqrt(np.mean((s - Uo) ** 2))); y = (W["Y"][c] == W["T"][c]).astype(float)
        base = rm(np.clip(R._krr_linear_loo(X[c], y, X[q]), 0, 1)); p, wv = beacon_s(W); b = rm(p)
        res.append((base, b)); print(seed, W["family"][:10], f"base {base:.3f} | beacon varying-coefficient {b:.3f} (w, w0) = {wv}", flush=True)
    print("mean base %.4f beacon %.4f" % tuple(np.mean(res, 0)))
