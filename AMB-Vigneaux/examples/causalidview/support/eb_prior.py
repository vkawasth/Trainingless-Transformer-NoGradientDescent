"""Empirical-Bayes support prior. Every label type k (T, Y, TY, 1[Y=T], and their instrument interactions) is a linear
model y_k = X beta_k + noise, beta_k ~ N(0, c_k Sigma), with ONE shared prior covariance Sigma = s0 I + U diag(l) U^T (rank r).
E-step: each task's posterior (mu_k, S_k).  M-step (the posterior -> prior map): Sigma from avg_k (mu_k mu_k^T + S_k)/c_k,
projected to rank r + isotropic (closed form, as in probabilistic PCA); per-task scale c_k and noise v_k in closed form.
No gradients. The posterior covariance S_k is what keeps the data from being counted twice."""
import os, sys, numpy as np
_h = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_h, "..")); sys.path.insert(0, os.path.join(_h, "../../.."))
import run as R

def tasks(W, idx):
    X = W["X"][idx]; T = W["T"][idx].astype(float); Y = W["Y"][idx].astype(float); I = W["I"][idx].astype(float); S_ = (Y == T).astype(float)
    cols = {"s": S_, "T": T, "Y": Y, "TY": T * Y, "sI": S_ * I, "TI": T * I, "YI": Y * I}
    out = {}
    for k, col in cols.items():                       # remove instrument-arm means (intercepts are unpenalised)
        mu0, mu1 = col[I == 0].mean(), col[I == 1].mean(); out[k] = (col - np.where(I == 1, mu1, mu0), (mu0, mu1))
    return X, out

def eb_fit(X, Ys, r=4, iters=200, tol=1e-6):
    n, p = X.shape; XtX = X.T @ X; K = len(Ys); names = list(Ys)
    Sig = np.eye(p) * 0.01; c = np.ones(K); v = np.array([Ys[k].var() for k in names]); ll_old = None
    for it in range(iters):
        Si = np.linalg.inv(Sig); M = np.zeros((p, p)); post = {}
        for j, k in enumerate(names):
            A = XtX / v[j] + Si / c[j]; S = np.linalg.inv(A); mu = S @ X.T @ Ys[k] / v[j]; post[k] = (mu, S)
            res = Ys[k] - X @ mu; v[j] = (res @ res + np.trace(XtX @ S)) / n
            c[j] = (mu @ Si @ mu + np.trace(Si @ S)) / p
            M += (np.outer(mu, mu) + S) / c[j]
        M /= K; ev, U = np.linalg.eigh(M); ev, U = ev[::-1], U[:, ::-1]
        s0 = ev[r:].mean() if r < p else 1e-6; l = np.maximum(ev[:r] - s0, 0)
        Sig = s0 * np.eye(p) + (U[:, :r] * l) @ U[:, :r].T
        cs = np.trace(Sig) / p; Sig /= cs; c *= cs                       # identifiability: tr(Sig) = p
        if ll_old is not None and np.max(np.abs(M - ll_old)) < tol * np.abs(M).max(): break
        ll_old = M
    return post, Sig, dict(zip(names, c)), dict(zip(names, v)), it

def evidence(X, y, C, v):
    """log marginal likelihood of y ~ N(0, X C X^T + v I) (for choosing the rank r)"""
    n = len(y); K = X @ C @ X.T + v * np.eye(n); L = np.linalg.cholesky(K); a = np.linalg.solve(L, y)
    return -0.5 * a @ a - np.log(np.diag(L)).sum() - 0.5 * n * np.log(2 * np.pi)

def predict_s(W, ranks=(1, 2, 3, 4, 6, 8)):
    c, q = W["ctx"], W["qry"]; X, Ys = tasks(W, c); Yc = {k: v[0] for k, v in Ys.items()}; best = None
    for r in ranks:
        post, Sig, cc, vv, it = eb_fit(X, Yc, r)
        ev = sum(evidence(X, Yc[k], cc[k] * Sig, vv[k]) for k in Yc)       # joint evidence over all tasks
        if best is None or ev > best[0]: best = (ev, r, post, Sig)
    _, r, post, Sig = best; mu0, mu1 = Ys["s"][1]
    s_hat = 0.5 * (mu0 + mu1) + W["X"][q] @ post["s"][0]                    # s(x) marginalises over the instrument (P(I=1)=1/2)
    return np.clip(s_hat, 0, 1), r, Sig

if __name__ == "__main__":
    from oracle_subspace import basis
    res = []
    for seed in range(int(os.environ.get("NW", 10))):
        W = R.world(seed); c, q = W["ctx"], W["qry"]; X = W["X"]; y = (W["Y"][c] == W["T"][c]).astype(float)
        Lo, Uo = R.manski(W["q"].mean(1)); rm = lambda s: float(np.sqrt(np.mean((s - Uo) ** 2)))
        base = rm(R._krr_linear_loo(X[c], y, X[q]))
        s_hat, r, Sig = predict_s(W); Qt = basis(seed); ev, U = np.linalg.eigh(Sig); U = U[:, ::-1][:, :r]
        cos = np.linalg.svd(Qt.T @ U, compute_uv=False)
        res.append((base, rm(s_hat))); print(seed, W["family"][:10], f"base {base:.3f} | EB support prior {rm(s_hat):.3f} (rank {r}) cos(true) {np.round(cos, 2)}", flush=True)
    print("mean base, EB", np.round(np.mean(res, 0), 4))
