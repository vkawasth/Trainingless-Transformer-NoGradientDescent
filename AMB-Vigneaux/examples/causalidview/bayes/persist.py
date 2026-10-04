"""Can Bayesian inference be made persistent? Context arrives in batches (512, then 8 x 64, up to 1024 units).
  persistent-k : chains are kept alive; after each batch every chain continues k iterations on the enlarged data
                 (warm start from its last state; draws from the last k/2 iterations)
  rerun        : 4 fresh chains x 3000 iterations (burn 1500) on the data so far, at n = 768 and n = 1024
  stale        : the n = 512 posterior, never updated
  ridge exact  : the structured backbone's ridge with a fixed penalty kept as additive sufficient statistics
                 (sum x x^T, sum x y, sum y, n): each update costs O(batch p^2) and is checked exact against a batch refit
Cost = likelihood work = sum over iterations of n (the per-iteration likelihood touches every unit), and wall seconds.
Manski and IV endpoint RMSE against the oracle at every step. Resumable: one json per world."""
import os, sys, time, json, numpy as np
_h = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_h, "..")); sys.path.insert(0, os.path.join(_h, "../../.."))
import run as R
import generic as G

def score(cells, W, ivo):
    Lo, Uo = R.manski(W["q"].mean(1)); Lm, Um = R.manski(cells.mean(1)); iv = np.array([R.iv_bounds(c) for c in cells])
    return R.ep_rmse(Lm, Um, Lo, Uo), R.ep_rmse(iv[:, 0], iv[:, 1], ivo[:, 0], ivo[:, 1])

def ridge_stream(W, order, sizes, lam=30.0):
    X = W["X"]; y = (W["Y"] == W["T"]).astype(float); p = X.shape[1]; A = np.zeros((p, p)); bx = np.zeros(p); sy = 0.0; n = 0; prev = 0
    out = []; Xq = X[W["qry"]]
    for m in sizes:
        idx = order[prev:m]; t0 = time.perf_counter()
        A += X[idx].T @ X[idx]; bx += X[idx].T @ y[idx]; sy += y[idx].sum(); n += len(idx)
        mx = bx * 0  # centring via sums: fit y - ybar on X - xbar
        xs = X[order[:m]].sum(0); Ac = A - np.outer(xs, xs) / n; bc = bx - xs * sy / n
        beta = np.linalg.solve(Ac + lam * np.eye(p), bc); t_inc = time.perf_counter() - t0
        t0 = time.perf_counter(); Xa = X[order[:m]]; Xc = Xa - Xa.mean(0); beta_b = np.linalg.solve(Xc.T @ Xc + lam * np.eye(p), Xc.T @ (y[order[:m]] - y[order[:m]].mean())); t_bat = time.perf_counter() - t0
        out.append(dict(n=m, max_abs_diff=float(np.abs(beta - beta_b).max()), t_incremental=t_inc, t_batch=t_bat))
        prev = m
    return out

if __name__ == "__main__":
    seeds = [int(s) for s in os.environ.get("SEEDS", "0 1 2 3 4").split()]; CH = 4; KS = (100, 300)
    os.makedirs(os.path.join(_h, "persist"), exist_ok=True)
    for seed in seeds:
        fn = os.path.join(_h, "persist", "w%d.json" % seed)
        if os.path.exists(fn): print(open(fn).read()[:300]); continue
        W = R.world(seed); nq = len(W["qry"]); ivo = np.array([R.iv_bounds(W["q"][i]) for i in range(nq)])
        order = np.random.default_rng(seed).permutation(W["ctx"]); sizes = [512 + 64 * t for t in range(9)]
        Wn = lambda m: dict(W, ctx=order[:m]); res = dict(seed=seed, family=W["family"], sizes=sizes)
        # initial posterior at n = 512 (shared by every persistent arm and by "stale")
        t0 = time.time(); init = [G.run_chain(Wn(512), 3000, 1500, 5000 * seed + ch, return_state=True) for ch in range(CH)]
        t_init = time.time() - t0; cells0 = np.mean([c[0] for c in init], 0); s0 = score(cells0, W, ivo)
        res["init"] = dict(score=s0, work=CH * 3000 * 512, seconds=t_init)
        for k in KS:
            states = [c[2] for c in init]; path = [s0]; work = 0; secs = 0.0
            for t in range(1, 9):
                m = sizes[t]; t0 = time.time(); outs = [G.run_chain(Wn(m), k, k // 2, 5000 * seed + 100 * t + ch + 7 * k, state=states[ch], return_state=True, thin=2) for ch in range(CH)]
                secs += time.time() - t0; work += CH * k * m; states = [o[2] for o in outs]
                path.append(score(np.mean([o[0] for o in outs], 0), W, ivo))
            res["persistent_%d" % k] = dict(path=path, update_work=work, update_seconds=secs)
        res["stale_at_1024"] = s0
        for m in (768, 1024):
            t0 = time.time(); outs = [G.run_chain(Wn(m), 3000, 1500, 5000 * seed + 900 + ch + m) for ch in range(CH)]
            res["rerun_%d" % m] = dict(score=score(np.mean([o[0] for o in outs], 0), W, ivo), work=CH * 3000 * m, seconds=time.time() - t0)
        res["ridge"] = ridge_stream(W, order, sizes)
        json.dump(res, open(fn, "w"), indent=1)
        p3 = res["persistent_300"]["path"]; p1 = res["persistent_100"]["path"]
        print(seed, W["family"][:10], "n=1024 Manski: rerun %.3f | persist-300 %.3f | persist-100 %.3f | stale %.3f || work: rerun %.2e, persist-300 updates %.2e (all 8 batches)" % (
            res["rerun_1024"]["score"][0], p3[-1][0], p1[-1][0], s0[0], res["rerun_1024"]["work"], res["persistent_300"]["update_work"]), flush=True)
