"""Coherent posterior -> prior: the subspace posterior from block A is the prior on a DISJOINT block B (and swapped).
Prior weight and rank are chosen by LOO on B, which is now honest because the prior did not see B."""
import os, sys, numpy as np
_h = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_h, "..")); sys.path.insert(0, os.path.join(_h, "../../.."))
import run as R
from subspace_prior import pooled_Q, fit_with_prior

def crossfit(W, n_splits=1, seed=0, ranks=(0, 1, 2, 3, 4, 5), ws=(0.3, 1, 3, 10, 30, 100)):
    c, q = W["ctx"], W["qry"]; X = W["X"]; y_all = (W["Y"] == W["T"]).astype(float); rng = np.random.default_rng(seed); preds = []; picks = []
    for _ in range(n_splits):
        perm = rng.permutation(c); A, B = perm[: len(c) // 2], perm[len(c) // 2:]
        for P1, P2 in ((A, B), (B, A)):
            Wa = dict(W); Wa["ctx"] = P1; Q, _ = pooled_Q(Wa)
            e, p, r, w = fit_with_prior(X[P2], y_all[P2], X[q], Q, ranks, ws); preds.append(p); picks.append((r, w))
    return np.mean(preds, 0), picks

if __name__ == "__main__":
    res = []; NS = int(os.environ.get("NS", 1))
    for seed in range(int(os.environ.get("NW", 10))):
        W = R.world(seed); c, q = W["ctx"], W["qry"]; X = W["X"]; y = (W["Y"][c] == W["T"][c]).astype(float)
        Lo, Uo = R.manski(W["q"].mean(1)); rm = lambda s: float(np.sqrt(np.mean((np.clip(s, 0, 1) - Uo) ** 2)))
        base = rm(R._krr_linear_loo(X[c], y, X[q]))
        # half-data without prior, to separate "lost half the data" from "gained the prior"
        rng = np.random.default_rng(0); perm = rng.permutation(c); h = [perm[:512], perm[512:]]
        half = rm(np.mean([R._krr_linear_loo(X[b], (W["Y"][b] == W["T"][b]).astype(float), X[q]) for b in h], 0))
        p, picks = crossfit(W, NS); cf = rm(p)
        res.append((base, half, cf)); print(seed, W["family"][:10], f"base {base:.3f} | half-split avg, no prior {half:.3f} | cross-fit prior {cf:.3f} picks {picks}", flush=True)
    print("mean base, half-no-prior, crossfit-prior", np.round(np.mean(res, 0), 4))
