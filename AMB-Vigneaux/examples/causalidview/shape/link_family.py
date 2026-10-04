"""Low-parameter links on the LOO index (honest) vs the oracle ceilings.
  platt  : s = sigmoid(a + b logit z)       (2 parameters, Newton/IRLS)
  cubic  : s = sigmoid(a + b u + c u^2 + d u^3), u = logit z (4 parameters)
  oracle affine / oracle monotone: fitted to the TRUE s at the queries (ceilings, not estimators)"""
import os, sys, numpy as np
from isotonic_link import R, loo_index, IsotonicRegression
def logit(p): p = np.clip(p, 0.02, 0.98); return np.log(p / (1 - p))
def fit_link(z, y, zq, deg):
    u, uq = logit(z), logit(zq); F = lambda v: np.vstack([v ** k for k in range(deg + 1)]).T
    b = R._irls_bin(F(u), y, 1e-6, free=tuple(range(deg + 1))); return R._sig(F(uq) @ b)
if __name__ == "__main__":
    res = []
    for seed in range(int(os.environ.get("NW", 40))):
        W = R.world(seed); c, q = W["ctx"], W["qry"]; X = W["X"]; y = (W["Y"][c] == W["T"][c]).astype(float)
        Lo, Uo = R.manski(W["q"].mean(1)); rm = lambda s: float(np.sqrt(np.mean((s - Uo) ** 2)))
        z, zq = loo_index(X[c], y, X[q]); lin = np.clip(zq, 0, 1)
        A = np.c_[np.ones(len(zq)), zq]; aff = A @ np.linalg.lstsq(A, Uo, rcond=None)[0]
        mono = IsotonicRegression(out_of_bounds="clip").fit(zq, Uo).predict(zq)
        r = (rm(lin), rm(fit_link(z, y, zq, 1)), rm(fit_link(z, y, zq, 3)), rm(aff), rm(mono)); res.append(r)
        print(seed, W["family"][:10], "linear %.3f | platt %.3f | cubic %.3f | oracle affine %.3f | oracle monotone %.3f" % r, flush=True)
    print("mean linear %.4f platt %.4f cubic %.4f | oracle affine %.4f oracle monotone %.4f" % tuple(np.mean(res, 0)))
