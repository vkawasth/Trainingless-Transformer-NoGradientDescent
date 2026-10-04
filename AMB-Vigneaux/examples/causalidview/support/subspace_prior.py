"""Support-guided prior: estimate the shared low-rank subspace from every label the world offers, use it as a prior.
pooled     : ridge (LOO) each label column on X -> coefficient matrix B -> SVD -> Q_r
sequential : T -> Y|T=1, Y|T=0 -> s, each stage's posterior mean direction added to the next stage's prior covariance
Final s fit: kernel ridge with prior covariance  eps*I + w * Q_r Q_r^T  (rank r, w, ridge by closed-form LOO)."""
import os, sys, numpy as np
_h = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_h, "..")); sys.path.insert(0, os.path.join(_h, "../../.."))
import run as R
from oracle_subspace import basis, krr_loo
LAMS = np.logspace(-2, 5, 29)

def ridge_dir(Xc, y, Sigma=None):
    """posterior mean of a linear model y ~ X beta with prior beta ~ N(0, Sigma/lam); lam by LOO. Returns beta."""
    Xs = Xc if Sigma is None else Xc @ np.linalg.cholesky(Sigma + 1e-9 * np.eye(len(Sigma)))
    K = Xs @ Xs.T; m = y.mean(); yc = y - m; ev, V = np.linalg.eigh(K); ev = np.maximum(ev, 0); yt = V.T @ yc; best = None
    for lam in LAMS:
        d = ev / (ev + lam); loo = (yc - V @ (d * yt)) / (1 - (V ** 2) @ d); e = np.mean(loo ** 2)
        if best is None or e < best[0]: best = (e, lam)
    alpha = V @ (yt / (ev + best[1])); L = np.eye(Xc.shape[1]) if Sigma is None else np.linalg.cholesky(Sigma + 1e-9 * np.eye(len(Sigma)))
    return L @ (Xs.T @ alpha)

def fit_with_prior(Xc, y, Xq, Q, ranks=(0, 1, 2, 3, 4, 5, 6), ws=(1, 3, 10, 30, 100)):
    best = None; G = Xc @ Xc.T; Gq = Xq @ Xc.T
    for r in ranks:
        for w in (ws if r else (0,)):
            P = Xc @ Q[:, :r]; Pq = Xq @ Q[:, :r]
            e, p = krr_loo(G + w * P @ P.T, y, Gq + w * Pq @ P.T, LAMS)
            if best is None or e < best[0]: best = (e, p, r, w)
    return best

def pooled_Q(W):
    c = W["ctx"]; X = W["X"][c]; T = W["T"][c].astype(float); Y = W["Y"][c].astype(float); I = W["I"][c].astype(float)
    cols = [T, Y, T * Y, (Y == T).astype(float), T * I, Y * I, (Y == T) * I]
    B = np.stack([ridge_dir(X, col - I * col[I == 1].mean() - (1 - I) * col[I == 0].mean()) for col in cols], 1)   # instrument means removed
    B /= np.linalg.norm(B, axis=0, keepdims=True) + 1e-12                                                             # each label one vote
    U, sv, _ = np.linalg.svd(B, full_matrices=False); return U, sv

def sequential_Q(W, w=10.0):
    c = W["ctx"]; X = W["X"][c]; T = W["T"][c].astype(float); Y = W["Y"][c].astype(float); I = W["I"][c].astype(float); p = X.shape[1]
    Sig = np.eye(p); dirs = []
    for X_, y_ in [(X, T - 0.5 * I), (X[T == 1], Y[T == 1]), (X[T == 0], Y[T == 0]), (X, (Y == T).astype(float))]:
        b = ridge_dir(X_, y_, Sig); u = b / (np.linalg.norm(b) + 1e-12); dirs.append(u)
        Sig = Sig + w * np.outer(u, u)          # posterior direction -> next prior (contravariant pull-back of the support)
    Qs, _ = np.linalg.qr(np.array(dirs).T); return Qs

if __name__ == "__main__":
    res = []
    for seed in range(int(os.environ.get("NW", 10))):
        W = R.world(seed); c, q = W["ctx"], W["qry"]; X = W["X"]; y = (W["Y"][c] == W["T"][c]).astype(float)
        Lo, Uo = R.manski(W["q"].mean(1)); rm = lambda s: float(np.sqrt(np.mean((np.clip(s, 0, 1) - Uo) ** 2)))
        base = rm(R._krr_linear_loo(X[c], y, X[q]))
        Qp, sv = pooled_Q(W); bp = fit_with_prior(X[c], y, X[q], Qp)
        Qs = sequential_Q(W); bs = fit_with_prior(X[c], y, X[q], Qs, ranks=(0, 1, 2, 3, 4))
        Qt = basis(seed); ang = np.linalg.svd(Qt.T @ Qp[:, :5], compute_uv=False)
        res.append((base, rm(bp[1]), rm(bs[1])))
        print(seed, W["family"][:10], f"base {base:.3f} | pooled {rm(bp[1]):.3f} (r={bp[2]},w={bp[3]}) | sequential {rm(bs[1]):.3f} (r={bs[2]},w={bs[3]}) | cos(true, pooled top5) {np.round(ang,2)}", flush=True)
    print("mean base, pooled, sequential", np.round(np.mean(res, 0), 4))
