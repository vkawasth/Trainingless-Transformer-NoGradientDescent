"""Shape, not directions: a monotone link fitted exactly by PAVA (isotonic regression; no gradients).
Index z = linear kernel-ridge score. Training units use their closed-form LEAVE-ONE-OUT score (so the link is fitted on
out-of-sample indices: no double counting); query units use the full-fit score. Then
  iso    : s = m(z), m nondecreasing, by PAVA
  blend  : s = a m(z) + (1 - a) z, a chosen by 5-fold CV of the PAVA step
Also the ceiling of 'shape on one index': PAVA of the ORACLE s on z (best monotone link of this index)."""
import os, sys, numpy as np
_h = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_h, "..")); sys.path.insert(0, os.path.join(_h, "../../.."))
import run as R
from sklearn.isotonic import IsotonicRegression
LAMS = np.logspace(-1, 4, 26)

def loo_index(Xc, y, Xq):
    m = y.mean(); yc = y - m; K = Xc @ Xc.T; ev, V = np.linalg.eigh(K); ev = np.maximum(ev, 0); yt = V.T @ yc; best = None
    for lam in LAMS:
        d = ev / (ev + lam); loo = (yc - V @ (d * yt)) / (1 - (V ** 2) @ d); e = np.mean(loo ** 2)
        if best is None or e < best[0]: best = (e, lam, loo)
    a = V @ (yt / (ev + best[1])); return y - best[2], m + Xq @ Xc.T @ a      # LOO scores at training units, full fit at queries

def iso(z, y, zq):
    return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(z, y).predict(zq)

def blend_cv(z, y, alphas=np.linspace(0, 1, 11), k=5, seed=0):
    f = np.random.default_rng(seed).integers(0, k, len(y)); err = np.zeros(len(alphas))
    for i in range(k):
        tr, te = f != i, f == i; mi = iso(z[tr], y[tr], z[te])
        err += [np.sum((a * mi + (1 - a) * np.clip(z[te], 0, 1) - y[te]) ** 2) for a in alphas]
    return alphas[int(np.argmin(err))]

if __name__ == "__main__":
    res = []
    for seed in range(int(os.environ.get("NW", 40))):
        W = R.world(seed); c, q = W["ctx"], W["qry"]; X = W["X"]; y = (W["Y"][c] == W["T"][c]).astype(float)
        Lo, Uo = R.manski(W["q"].mean(1)); rm = lambda s: float(np.sqrt(np.mean((s - Uo) ** 2)))
        z, zq = loo_index(X[c], y, X[q]); lin = np.clip(zq, 0, 1); m = iso(z, y, zq); a = blend_cv(z, y)
        ceil = IsotonicRegression(out_of_bounds="clip").fit(zq, Uo).predict(zq)       # best monotone link of this index (oracle)
        r = (rm(lin), rm(m), rm(a * m + (1 - a) * lin), a, rm(ceil)); res.append(r)
        print(seed, W["family"][:10], "linear %.3f | PAVA %.3f | blend %.3f (a=%.1f) | oracle monotone link of this index %.3f" % r, flush=True)
    print("mean linear %.4f PAVA %.4f blend %.4f a %.2f oracle-link %.4f" % tuple(np.mean(res, 0)))
