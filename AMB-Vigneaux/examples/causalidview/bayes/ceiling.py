"""Ceiling for any amortised prior (TabPFN included): the exact Bayes posterior under the GENERATOR's own prior family.
Parameters: vT, mu_w (14-sparse unit vectors), a, b (10-sparse unit vectors); amplitudes, loadings and the effect
family are the published / world constants (family known: a slightly stronger prior than the generator's uniform
choice of family). Likelihood: P(T, Y | x, I) = 1/2 sum_u P(T | x, u, I) P(Y | T, x, u) on the 1024 context units.
Sampler: Metropolis-within-Gibbs (no gradients). Each unit vector is w = z/|z| on its support, z ~ N(0, I) (uniform on
the sphere); moves: random-walk on z (adaptive scale during burn-in) and support swaps. Posterior mean of the Manski
upper bound s(x) = P(Y = T | x) at the query units, averaged over I ~ Bernoulli(1/2), as in the oracle."""
import os, sys, time, numpy as np
_h = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_h, "..")); sys.path.insert(0, os.path.join(_h, "../../.."))
import run as R
from scipy.special import expit, log_expit
S = R.S
KACT = {"vT": 14, "mu": 14, "a": 10, "b": 10}

def calib(seed, m=5000):
    return S._rng(seed, "X_calibration").normal(size=(R.NCAL, R.D))[:m]

class Model:
    def __init__(self, fam, xc):
        self.fam, self.xc = fam, xc
    def derived(self, th):
        """intercept bT and the effect standardisation, both functions of the parameters (computed as the generator does)"""
        xc = self.xc
        bT = S._calibrate_intercept(np.concatenate([R.A_X * (xc @ th["vT"]) + R.A_U * u + R.A_I * j for u in (-1, 1) for j in (0, 1)]))
        rc = S._raw_cate(xc, self.fam, {"a": th["a"], "b": th["b"]}); return bT, rc.mean(), rc.std()
    def parts(self, th, X, d):
        bT, m, sd = d; tau = R.A_TAU * np.tanh((S._raw_cate(X, self.fam, {"a": th["a"], "b": th["b"]}) - m) / sd)
        fb = 0.7 * (X @ th["mu"]) + 0.25 * np.sin(X @ np.roll(th["mu"], 1)); xv = X @ th["vT"]
        return bT, xv, fb, tau
    def loglik(self, th, X, T, Y, I, d):
        bT, xv, fb, tau = self.parts(th, X, d); L = 0.0; acc = np.zeros(len(T))
        for u in (-1, 1):
            e = expit(bT + R.A_X * xv + R.A_U * u + R.A_I * I); pT = np.where(T == 1, e, 1 - e)
            py = np.clip(0.5 + R.A_B * np.tanh(fb + R.C_U * u) + (T - 0.5) * tau, 1e-9, 1 - 1e-9)
            acc += 0.5 * pT * np.where(Y == 1, py, 1 - py)
        return float(np.log(acc).sum())
    def s_query(self, th, X, d):
        bT, xv, fb, tau = self.parts(th, X, d); s = np.zeros(len(X))
        for u in (-1, 1):
            for j in (0, 1):
                e = expit(bT + R.A_X * xv + R.A_U * u + R.A_I * j); b = 0.5 + R.A_B * np.tanh(fb + R.C_U * u)
                s += 0.25 * (e * (b + tau / 2) + (1 - e) * (1 - (b - tau / 2)))
        return s

def unit(z, supp, D=R.D):
    w = np.zeros(D); w[supp] = z / np.linalg.norm(z); return w

def run_chain(W, fam, xc, n_iter, burn, seed, init=None):
    rng = np.random.default_rng(seed); c, q = W["ctx"], W["qry"]; X = W["X"][c]; T = W["T"][c]; Y = W["Y"][c]; I = W["I"][c].astype(float); Xq = W["X"][q]
    M = Model(fam, xc); st = {}
    for k, kk in KACT.items():
        supp = rng.choice(R.D, kk, replace=False); st[k] = [supp, rng.normal(size=kk)]
    if init is not None:                                   # data-driven start for vT (IRLS treatment fit); valid for any MCMC
        idx = np.argsort(-np.abs(init))[:KACT["vT"]]; st["vT"] = [idx, init[idx] * 10]
    th = {k: unit(z, s) for k, (s, z) in st.items()}; d = M.derived(th); ll = M.loglik(th, X, T, Y, I, d)
    lp = lambda z: -0.5 * float(z @ z)
    scale = {k: 0.3 for k in KACT}; acc = {k: [0, 0] for k in KACT}; draws = []; trace = []
    for it in range(n_iter):
        for k in KACT:
            supp, z = st[k]
            for move in ("rw", "swap"):
                if move == "rw":
                    z2, s2 = z + scale[k] * rng.normal(size=len(z)), supp
                else:
                    i = rng.integers(len(supp)); out = rng.choice(np.setdiff1d(np.arange(R.D), supp)); s2 = supp.copy(); s2[i] = out; z2 = z
                th2 = dict(th); th2[k] = unit(z2, s2); d2 = M.derived(th2) if k in ("vT", "a", "b") else d
                ll2 = M.loglik(th2, X, T, Y, I, d2)
                if np.log(rng.random()) < ll2 + lp(z2) - ll - lp(z):
                    st[k] = [s2, z2]; supp, z = s2, z2; th, d, ll = th2, d2, ll2
                    if move == "rw": acc[k][0] += 1
                if move == "rw": acc[k][1] += 1
            if it < burn and it % 50 == 49:                  # adapt the random-walk scale toward ~30% acceptance
                r = acc[k][0] / max(acc[k][1], 1); scale[k] *= np.exp(r - 0.3); acc[k] = [0, 0]
        trace.append(ll)
        if os.environ.get("VERBOSE") and it % 250 == 0: print("   it", it, "loglik %.1f" % ll, flush=True)
        if it >= burn and it % 5 == 0:
            draws.append(M.s_query(th, Xq, d))
    return np.mean(draws, 0), np.array(trace)

if __name__ == "__main__":
    worlds = [int(s) for s in os.environ.get("SEEDS", "0 1 2 3 5 6 7 8").split()]
    NIT, BURN = int(os.environ.get("NIT", 3000)), int(os.environ.get("BURN", 1500)); CH = int(os.environ.get("CHAINS", 2)); res = []
    for seed in worlds:
        W = R.world(seed); fam = W["family"]; xc = calib(seed); Lo, Uo = R.manski(W["q"].mean(1)); rm = lambda s: float(np.sqrt(np.mean((s - Uo) ** 2)))
        c = W["ctx"]; Z = np.hstack([np.ones((len(c), 1)), W["X"][c], W["I"][c][:, None].astype(float)])
        bT0 = R._cv_logistic(Z, W["T"][c].astype(float), free=(0, Z.shape[1] - 1))[1:-1]
        base = rm(np.clip(R._krr_linear_loo(W["X"][c], (W["Y"][c] == W["T"][c]).astype(float), W["X"][W["qry"]]), 0, 1))
        t0 = time.time(); means = []; tails = []
        for ch in range(CH):
            m, tr = run_chain(W, fam, xc, NIT, BURN, 1000 * seed + ch + 100 * int(os.environ.get('CHSEED', 0)), init=bT0); means.append(m); tails.append(tr[-200:].mean())
        os.makedirs(os.path.join(_h, 'chains'), exist_ok=True); np.save(os.path.join(_h, 'chains', '%s_w%d_%s.npy' % ('ceiling', seed, os.environ.get('TAG', 'r0'))), np.array(means)); post = np.mean(means, 0); r = (base, rm(post), *[rm(m) for m in means], float(np.sqrt(np.mean((means[0] - means[-1]) ** 2))))
        res.append(r[:2])
        print(seed, fam[:10], "base %.3f | Bayes ceiling %.3f (chains %s; between-chain rms %.3f) | loglik tails %s | %.0fs" % (r[0], r[1], " ".join("%.3f" % x for x in r[2:-1]), r[-1], np.round(tails, 1), time.time() - t0), flush=True)
    print("mean base %.4f | Bayes ceiling %.4f" % tuple(np.mean(res, 0)))
