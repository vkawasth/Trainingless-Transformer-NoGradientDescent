"""Ceiling for the beacon: the TRUE propensity E_u[e(x, u, I)] in place of e_hat."""
import os, sys, numpy as np
from beacon import R, beacon_s
from scipy.special import expit
S = R.S
def true_g(W, seed):
    rng = S._rng(seed, "nuisance_parameters"); mu = S._unit_vector(rng, R.D, 14); vT = S._unit_vector(rng, R.D, 14)
    xc = S._rng(seed, "X_calibration").normal(size=(R.NCAL, R.D))
    bT = S._calibrate_intercept(np.concatenate([R.A_X * (xc @ vT) + R.A_U * u + R.A_I * j for u in (-1, 1) for j in (0, 1)]))
    e = lambda X, j: 0.5 * sum(expit(bT + R.A_X * (X @ vT) + R.A_U * u + R.A_I * j) for u in (-1, 1))
    c, q = W["ctx"], W["qry"]; X = W["X"]
    return 2 * e(X[c], W["I"][c]) - 1, [2 * e(X[q], j) - 1 for j in (0, 1)]
if __name__ == "__main__":
    res = []
    for seed in range(int(os.environ.get("NW", 10))):
        W = R.world(seed); Lo, Uo = R.manski(W["q"].mean(1)); rm = lambda s: float(np.sqrt(np.mean((s - Uo) ** 2)))
        p, wv = beacon_s(W, g_override=true_g(W, seed)); res.append(rm(p)); print(seed, W["family"][:10], f"beacon with TRUE propensity {rm(p):.3f} {wv}", flush=True)
    print("mean %.4f" % np.mean(res))
