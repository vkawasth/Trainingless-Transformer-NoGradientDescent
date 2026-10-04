"""Ceiling for 'the law carries over': hand the estimator the TRUE rank and TRUE sparsity (not the directions)."""
import os, sys, numpy as np
from law_prepass import R, basis, labels, directions, cos_true, truth, fit_with_prior
if __name__ == "__main__":
    res = []
    for seed in range(int(os.environ.get("NW", 10))):
        W = R.world(seed); c, q = W["ctx"], W["qry"]; X = W["X"]; Lo, Uo = R.manski(W["q"].mean(1))
        rm = lambda s: float(np.sqrt(np.mean((np.clip(s, 0, 1) - Uo) ** 2))); ys = (W["Y"] == W["T"]).astype(float)
        r_true, m_true = truth(seed); m = int(round(np.mean(m_true))); Qt = basis(seed)
        cs = cos_true(directions(X[c], labels(W, c), m), Qt, 3).mean()
        rng = np.random.default_rng(0); perm = rng.permutation(c); A, B = perm[:512], perm[512:]; preds = []
        for P1, P2 in ((A, B), (B, A)):
            Q = np.linalg.svd(directions(X[P1], labels(W, P1), m), full_matrices=False)[0][:, :min(r_true, 7)]
            preds.append(fit_with_prior(X[P2], ys[P2], X[q], Q, ranks=(min(r_true, 7),), ws=(0.3, 1, 3, 10, 30, 100))[1])
        base = rm(R._krr_linear_loo(X[c], ys[c], X[q])); o = rm(np.mean(preds, 0)); res.append((cs, base, o))
        print(seed, W["family"][:10], f"true law (r={r_true}, m={m}): cos top-3 {cs:.2f} | RMSE base {base:.3f} cross-fit with true law {o:.3f}", flush=True)
    print("mean cos %.2f | base %.4f | true-law %.4f" % tuple(np.mean(res, 0)))
