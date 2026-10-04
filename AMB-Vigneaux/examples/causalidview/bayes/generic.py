"""Fair generic prior (not the generator's form): a sparse multi-index model with spline links, shared directions.
  P(T=1 | x, I)      = sigmoid(a0 + aI I + g * wT.x)
  P(Y=1 | T=t, x, I) = sigmoid(c_t + cI_t I + sum_k B(w_k.x) beta_{t,k}),  k = 1..K (K = 3 shared directions)
B = cubic B-spline basis on the standardised index (7 functions). Priors: each direction a 12-sparse random unit vector
(z ~ N(0, I) on a uniformly random support); spline and intercept coefficients N(0, 1). No latent confounder, no tanh,
no amplitudes. Sampler: Metropolis-within-Gibbs (random walk + support swaps), no gradients. Output: posterior mean of
s(x) = sum_I 1/2 [e m1 + (1 - e)(1 - m0)] at the query units."""
import os, sys, time, numpy as np
_h = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_h, "..")); sys.path.insert(0, os.path.join(_h, "../../.."))
import run as R
from scipy.special import expit
from scipy.interpolate import BSpline
K, SP, NB = 3, 12, 7
KNOTS = np.r_[[-3] * 3, np.linspace(-3, 3, NB - 2), [3] * 3]

def basis(z):
    z = np.clip(z, -2.999, 2.999); return np.stack([BSpline(KNOTS, np.eye(NB)[i], 3, extrapolate=False)(z) for i in range(NB)], 1)

def unit(z, supp, D=R.D):
    w = np.zeros(D); w[supp] = z / np.linalg.norm(z); return w

class Gen:
    def __init__(self, X, T, Y, I):
        self.X, self.T, self.Y, self.I = X, T, Y, I
    def probs(self, p, X, I):
        e = expit(p["a"][0] + p["a"][1] * I + p["g"] * (X @ p["wT"]) * 3)
        F = sum(basis(X @ p["w"][k]) @ p["beta"][:, k, :].T for k in range(K))       # (n, 2): arm t
        m = expit(p["c"][None, :, 0] + p["c"][None, :, 1] * I[:, None] + F); return e, m
    def loglik(self, p):
        e, m = self.probs(p, self.X, self.I); T, Y = self.T, self.Y
        pt = np.where(T == 1, e, 1 - e); my = m[np.arange(len(T)), T]; py = np.where(Y == 1, my, 1 - my)
        return float(np.log(np.clip(pt * py, 1e-12, None)).sum())
    def cells(self, p, Xq):
        """q[:, j, t, y] = P(T=t, Y=y | x, I=j)"""
        q = np.zeros((len(Xq), 2, 2, 2))
        for j in (0, 1):
            e, m = self.probs(p, Xq, np.full(len(Xq), float(j)))
            q[:, j, 1, 1] = e * m[:, 1]; q[:, j, 1, 0] = e * (1 - m[:, 1]); q[:, j, 0, 1] = (1 - e) * m[:, 0]; q[:, j, 0, 0] = (1 - e) * (1 - m[:, 0])
        return q
    def s(self, p, Xq):
        out = 0
        for j in (0, 1):
            e, m = self.probs(p, Xq, np.full(len(Xq), float(j))); out = out + 0.5 * (e * m[:, 1] + (1 - e) * (1 - m[:, 0]))
        return out

def run_chain(W, n_iter, burn, seed, state=None, return_state=False, thin=5):
    """state: (dirs, p, scales) from a previous call (warm start on new data); return_state: also return it"""
    rng = np.random.default_rng(seed); c, q = W["ctx"], W["qry"]
    G = Gen(W["X"][c], W["T"][c], W["Y"][c], W["I"][c].astype(float))
    dirs = {k: [rng.choice(R.D, SP, replace=False), rng.normal(size=SP)] for k in ["T"] + list(range(K))}
    p = dict(a=np.zeros(2), g=0.3, c=np.zeros((2, 2)), beta=np.zeros((2, K, NB)))
    if state is not None:
        dirs = {k: [v[0].copy(), v[1].copy()] for k, v in state[0].items()}; p = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in state[1].items()}
    def assemble(p, dirs):
        p = dict(p); p["wT"] = unit(*dirs["T"][::-1]); p["w"] = [unit(*dirs[k][::-1]) for k in range(K)]; return p
    P = assemble(p, dirs); ll = G.loglik(P); lpz = lambda z: -0.5 * float(z @ z)
    sc = {k: 0.3 for k in dirs}; sc.update(a=0.1, g=0.05, c=0.1, beta=0.1)
    if state is not None: sc = dict(state[2])
    acc = {k: [0, 0] for k in sc}; draws = []
    for it in range(n_iter):
        for k in list(dirs):                                       # directions: random walk on z, support swap
            for move in ("rw", "swap"):
                s0, z0 = dirs[k]
                if move == "rw": s1, z1 = s0, z0 + sc[k] * rng.normal(size=SP)
                else:
                    s1 = s0.copy(); s1[rng.integers(SP)] = rng.choice(np.setdiff1d(np.arange(R.D), s0)); z1 = z0
                d2 = dict(dirs); d2[k] = [s1, z1]; P2 = assemble(p, d2); l2 = G.loglik(P2)
                if np.log(rng.random()) < l2 + lpz(z1) - ll - lpz(z0):
                    dirs, P, ll = d2, P2, l2; acc[k][0] += move == "rw"
                acc[k][1] += move == "rw"
        for k in ("a", "g", "c", "beta"):                          # coefficients: random walk, N(0, 1) prior
            v0 = np.asarray(p[k], float); v1 = v0 + sc[k] * rng.normal(size=v0.shape)
            p2 = dict(p); p2[k] = v1 if v0.ndim else float(v1); P2 = assemble(p2, dirs); l2 = G.loglik(P2)
            if np.log(rng.random()) < l2 - 0.5 * float((v1 ** 2).sum()) - ll + 0.5 * float((v0 ** 2).sum()):
                p, P, ll = p2, P2, l2; acc[k][0] += 1
            acc[k][1] += 1
        if it < burn and it % 50 == 49:
            for k in sc:
                r = acc[k][0] / max(acc[k][1], 1); sc[k] *= np.exp(r - 0.3); acc[k] = [0, 0]
        if it >= burn and (it - burn) % thin == 0:
            draws.append(G.cells(P, W["X"][q]))
    if return_state: return np.mean(draws, 0), ll, (dirs, p, sc)
    return np.mean(draws, 0), ll

if __name__ == "__main__":
    worlds = [int(s) for s in os.environ.get("SEEDS", "0 1 2 3 5 6 7 8").split()]
    NIT, BURN, CH = int(os.environ.get("NIT", 3000)), int(os.environ.get("BURN", 1500)), int(os.environ.get("CHAINS", 2)); res = []
    for seed in worlds:
        W = R.world(seed); Lo, Uo = R.manski(W["q"].mean(1)); c = W["ctx"]; nq = len(W["qry"])
        ivo = np.array([R.iv_bounds(W["q"][i]) for i in range(nq)])
        rm = lambda s: float(np.sqrt(np.mean((s - Uo) ** 2)))
        base = rm(np.clip(R._krr_linear_loo(W["X"][c], (W["Y"][c] == W["T"][c]).astype(float), W["X"][W["qry"]]), 0, 1))
        t0 = time.time(); fn = os.path.join(_h, 'chains', 'genericcells_w%d_%s.npy' % (seed, os.environ.get('TAG', 'r0')))
        if os.path.exists(fn):                                   # resume: chains already sampled for this world
            cells = np.load(fn)
        else:
            out = [run_chain(W, NIT, BURN, 7000 * seed + ch + 100 * int(os.environ.get('CHSEED', 0))) for ch in range(CH)]
            cells = np.array([o[0] for o in out]); os.makedirs(os.path.join(_h, 'chains'), exist_ok=True); np.save(fn, cells)
        post = cells.mean(0); Lm, Um = R.manski(post.mean(1)); iv = np.array([R.iv_bounds(post[i]) for i in range(nq)])
        r = dict(seed=seed, family=W["family"], base=base, manski=R.ep_rmse(Lm, Um, Lo, Uo), iv=R.ep_rmse(iv[:, 0], iv[:, 1], ivo[:, 0], ivo[:, 1]),
                 infeasible=float(np.mean(iv[:, 2] > 1e-7)), chains=[rm(R.manski(cc.mean(1))[1]) for cc in cells], seconds=time.time() - t0)
        res.append(r)
        print(seed, W["family"][:10], "base %.3f | generic-prior Bayes: Manski %.3f IV %.3f (IV infeasible %.2f) | chains %s | %.0fs" % (
            base, r["manski"], r["iv"], r["infeasible"], " ".join("%.3f" % x for x in r["chains"]), r["seconds"]), flush=True)
    import json; json.dump(res, open(os.path.join(_h, 'generic_%s.json' % os.environ.get('TAG', 'r0')), 'w'), indent=1)
    print("mean base %.4f | generic-prior Bayes Manski %.4f IV %.4f" % (np.mean([r["base"] for r in res]), np.mean([r["manski"] for r in res]), np.mean([r["iv"] for r in res])))
