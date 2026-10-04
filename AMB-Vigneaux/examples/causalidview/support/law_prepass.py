"""Can a cheap pre-pass estimate the LAW of supports inside one world, and does using it help?
Pre-pass (closed form, seconds):
  rank r_hat   : singular values of the stacked per-label ridge coefficients vs a permutation null (labels shuffled)
  sparsity m_hat: active coordinates of the STRONG task (T | X, I) by ARD; the law says all directions share it
  signal       : LOO R^2 per label
Use: sparse directions (each label's coefficients hard-thresholded to its top-m_hat coordinates, refit) -> SVD -> Q,
compared with dense directions by cosine to the true subspace, and by Manski RMSE with an honest cross-fitted prior."""
import os, sys, numpy as np
_h = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_h, "..")); sys.path.insert(0, os.path.join(_h, "../../.."))
sys.path.insert(0, os.path.join(_h, "../ablations"))
import run as R
from oracle_subspace import basis
from subspace_prior import ridge_dir, fit_with_prior
from ard_lib import ard
S = R.S

def labels(W, idx):
    T = W["T"][idx].astype(float); Y = W["Y"][idx].astype(float); I = W["I"][idx].astype(float); s = (Y == T).astype(float)
    cols = [T, Y, T * Y, s, T * I, Y * I, s * I]
    return [col - np.where(I == 1, col[I == 1].mean(), col[I == 0].mean()) for col in cols]

def directions(X, cols, m=None):
    B = []
    for y in cols:
        b = ridge_dir(X, y)
        if m is not None:
            keep = np.argsort(-np.abs(b))[:m]; bk = np.zeros_like(b); bk[keep] = ridge_dir(X[:, keep], y); b = bk
        B.append(b / (np.linalg.norm(b) + 1e-12))
    return np.array(B).T

def prepass(W, idx, n_perm=5, seed=0):
    X = W["X"][idx]; cols = labels(W, idx); rng = np.random.default_rng(seed)
    sv = np.linalg.svd(np.array([ridge_dir(X, y) for y in cols]).T, compute_uv=False)
    null = np.max([np.linalg.svd(np.array([ridge_dir(X, y[pp]) for y in cols]).T, compute_uv=False)[0]
                   for pp in (rng.permutation(len(idx)) for _ in range(n_perm))])
    r_hat = int((sv > null).sum())
    T = W["T"][idx].astype(float); I = W["I"][idx].astype(float)
    _, mu, a = ard(np.c_[X, I], T); m_hat = int((a[:-1] < 1e3).sum())
    return r_hat, m_hat

def truth(seed):
    fam = S.CATE_FAMILIES[seed % len(S.CATE_FAMILIES)]; rng = S._rng(seed, "nuisance_parameters")
    mu = S._unit_vector(rng, R.D, 14); vT = S._unit_vector(rng, R.D, 14); cp = S._cate_parameters(seed, R.D, fam)
    return basis(seed).shape[1], [int((v != 0).sum()) for v in (vT, mu, cp["a"], cp["b"])]

def cos_true(Q, Qt, r):
    U = np.linalg.svd(Q, full_matrices=False)[0][:, :r]; return np.linalg.svd(Qt.T @ U, compute_uv=False)

if __name__ == "__main__":
    res = []
    for seed in range(int(os.environ.get("NW", 10))):
        W = R.world(seed); c, q = W["ctx"], W["qry"]; X = W["X"]; Lo, Uo = R.manski(W["q"].mean(1))
        rm = lambda s: float(np.sqrt(np.mean((np.clip(s, 0, 1) - Uo) ** 2)))
        r_true, m_true = truth(seed); r_hat, m_hat = prepass(W, c); Qt = basis(seed)
        cd = cos_true(directions(X[c], labels(W, c)), Qt, 3); cs = cos_true(directions(X[c], labels(W, c), m_hat), Qt, 3)
        # honest cross-fitted prior, dense vs sparse directions, rank r_hat
        rng = np.random.default_rng(0); perm = rng.permutation(c); A, B = perm[:512], perm[512:]; ys = (W["Y"] == W["T"]).astype(float)
        out = {}
        for name, m in (("dense", None), ("sparse", m_hat)):
            preds = []
            for P1, P2 in ((A, B), (B, A)):
                Q = np.linalg.svd(directions(X[P1], labels(W, P1), m), full_matrices=False)[0]
                preds.append(fit_with_prior(X[P2], ys[P2], X[q], Q, ranks=tuple(range(0, max(r_hat, 1) + 1)), ws=(0.3, 1, 3, 10, 30, 100))[1])
            out[name] = rm(np.mean(preds, 0))
        base = rm(R._krr_linear_loo(X[c], ys[c], X[q]))
        res.append((r_hat, r_true, m_hat, np.mean(m_true), cd.mean(), cs.mean(), base, out["dense"], out["sparse"]))
        print(seed, W["family"][:10], f"rank {r_hat} (true {r_true}) | sparsity {m_hat} (true {m_true}) | mean cos top-3 dense {cd.mean():.2f} sparse {cs.mean():.2f}"
              f" | RMSE base {base:.3f} crossfit dense {out['dense']:.3f} sparse {out['sparse']:.3f}", flush=True)
    a = np.array(res, float); print("mean: rank_hat %.1f true %.1f | m_hat %.1f true %.1f | cos dense %.2f sparse %.2f | base %.4f dense %.4f sparse %.4f" % tuple(a.mean(0)))
