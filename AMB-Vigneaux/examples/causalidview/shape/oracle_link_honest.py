"""Is the 'best monotone link of our index' ceiling real or in-sample optimism? Fit the oracle link on half the
queries, evaluate on the other half (and swap)."""
import os, sys, numpy as np
from isotonic_link import R, loo_index, IsotonicRegression
if __name__ == "__main__":
    res = []
    for seed in range(int(os.environ.get("NW", 40))):
        W = R.world(seed); c, q = W["ctx"], W["qry"]; X = W["X"]; y = (W["Y"][c] == W["T"][c]).astype(float)
        Lo, Uo = R.manski(W["q"].mean(1)); z, zq = loo_index(X[c], y, X[q]); lin = np.clip(zq, 0, 1)
        h = np.arange(len(q)) % 2 == 0; pred = np.empty(len(q))
        for A, B in ((h, ~h), (~h, h)):
            pred[B] = IsotonicRegression(out_of_bounds="clip").fit(zq[A], Uo[A]).predict(zq[B])
        res.append((np.sqrt(np.mean((lin - Uo) ** 2)), np.sqrt(np.mean((pred - Uo) ** 2))))
    print("mean linear %.4f | oracle monotone link, held-out half %.4f" % tuple(np.mean(res, 0)))
