"""EB support prior with an inverse-Wishart hyperprior: Sigma = (nu * s_iso I + sum_k (mu mu^T + S)/c_k) / (nu + K), rank-r + iso.
Diagnostic sweep over nu (scored against the oracle only to see whether ANY strength helps)."""
import os, sys, numpy as np
_h = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_h, "..")); sys.path.insert(0, os.path.join(_h, "../../.."))
import run as R
from eb_prior import tasks

def eb_fit(X, Ys, r, nu, iters=300):
    n, p = X.shape; XtX = X.T @ X; names = list(Ys); K = len(names)
    Sig = np.eye(p); c = np.full(K, 1e-3); v = np.array([Ys[k].var() for k in names])
    for it in range(iters):
        Si = np.linalg.inv(Sig); M = nu * np.eye(p); post = {}
        for j, k in enumerate(names):
            S = np.linalg.inv(XtX / v[j] + Si / c[j]); mu = S @ X.T @ Ys[k] / v[j]; post[k] = (mu, S)
            res = Ys[k] - X @ mu; v[j] = (res @ res + np.trace(XtX @ S)) / n
            c[j] = (mu @ Si @ mu + np.trace(Si @ S)) / p; M += (np.outer(mu, mu) + S) / c[j]
        M /= (nu + K); ev, U = np.linalg.eigh(M); ev, U = ev[::-1], U[:, ::-1]
        s0 = ev[r:].mean(); new = s0 * np.eye(p) + (U[:, :r] * np.maximum(ev[:r] - s0, 0)) @ U[:, :r].T
        new *= p / np.trace(new)
        if np.abs(new - Sig).max() < 1e-7: Sig = new; break
        Sig = new
    return post

if __name__ == "__main__":
    NUS = (1, 5, 20, 50, 200, 1000); RS = (1, 3, 5); res = {}
    for seed in range(int(os.environ.get("NW", 6))):
        W = R.world(seed); c, q = W["ctx"], W["qry"]; Lo, Uo = R.manski(W["q"].mean(1)); rm = lambda s: float(np.sqrt(np.mean((s - Uo) ** 2)))
        X, Ys = tasks(W, c); Yc = {k: v[0] for k, v in Ys.items()}; mu0, mu1 = Ys["s"][1]; row = {}
        for r in RS:
            for nu in NUS:
                post = eb_fit(X, Yc, r, nu); row[(r, nu)] = rm(np.clip(0.5 * (mu0 + mu1) + W["X"][q] @ post["s"][0], 0, 1))
        row["base"] = rm(R._krr_linear_loo(W["X"][c], (W["Y"][c] == W["T"][c]).astype(float), W["X"][q]))
        for k, v in row.items(): res.setdefault(k, []).append(v)
        print(seed, "base %.3f" % row["base"], "best", min((v, k) for k, v in row.items() if k != "base"), flush=True)
    print({str(k): round(float(np.mean(v)), 4) for k, v in res.items()})
