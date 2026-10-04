"""Provenance-aware pooling of chains (each chain = one branch of a per-branch structured posterior, its record = its
held-out evidence) against equal pooling (Giry flattening), on the shallow-MLP worlds and on control worlds.
Each chain is fitted on 80% of the context; the other 20% is held out. For every chain we keep the posterior-mean cells
at the query units and its predictive probability of each held-out unit's observed (T, Y) given (x, I). Pooling rules:
  equal        w_c = 1/C
  stacking     w maximises sum_i log sum_c w_c p_c(i) on the simplex (Yao, Vehtari, Gelman 2022); solved by the EM
               fixed point w_c <- w_c mean_i p_c(i) / sum_c' w_c' p_c'(i)  (multiplicative, no gradients)
  pseudo-BMA   w_c ∝ exp(sum_i log p_c(i))
  best chain   the single chain with the highest held-out log score
Manski and IV bounds from the pooled cells, scored against the oracle. Resumable (one .npz per world)."""
import os, sys, time, json, numpy as np
_h = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_h, "..")); sys.path.insert(0, os.path.join(_h, "../../.."))
import run as R
import generic as G

def stack_weights(P, iters=2000):
    C = P.shape[1]; w = np.full(C, 1.0 / C)
    for _ in range(iters):
        mix = P @ w; w_new = w * (P / mix[:, None]).mean(0)
        if np.abs(w_new - w).max() < 1e-10: w = w_new; break
        w = w_new
    return w / w.sum()

def score(cells, W, ivo):
    Lo, Uo = R.manski(W["q"].mean(1)); Lm, Um = R.manski(cells.mean(1)); iv = np.array([R.iv_bounds(cells[i]) for i in range(len(cells))])
    return R.ep_rmse(Lm, Um, Lo, Uo), R.ep_rmse(iv[:, 0], iv[:, 1], ivo[:, 0], ivo[:, 1])

if __name__ == "__main__":
    seeds = [int(s) for s in os.environ.get("SEEDS", " ".join(str(s) for s in range(4, 40, 5))).split()]
    CH, NIT, BURN = int(os.environ.get("CHAINS", 8)), int(os.environ.get("NIT", 3000)), int(os.environ.get("BURN", 1500))
    os.makedirs(os.path.join(_h, "stacking"), exist_ok=True); rows = []
    for seed in seeds:
        fn = os.path.join(_h, "stacking", "w%d.npz" % seed); W = R.world(seed); nq = len(W["qry"])
        if not os.path.exists(fn):
            perm = np.random.default_rng(seed).permutation(W["ctx"]); fit, val = perm[: int(0.8 * len(perm))], perm[int(0.8 * len(perm)):]
            Wf = dict(W); Wf["ctx"] = fit; Wf["qry"] = np.concatenate([W["qry"], val])
            Q, Pv = [], []
            for ch in range(CH):
                cells, _ = G.run_chain(Wf, NIT, BURN, 9000 * seed + ch)
                Q.append(cells[:nq]); cv = cells[nq:]
                Pv.append(cv[np.arange(len(val)), W["I"][val], W["T"][val], W["Y"][val]])
            np.savez(fn, Q=np.array(Q), P=np.array(Pv).T)
        d = np.load(fn); Q, P = d["Q"], d["P"]
        ivo = np.array([R.iv_bounds(W["q"][i]) for i in range(nq)])
        ll = np.log(np.clip(P, 1e-300, None)).sum(0)
        rules = {"equal": np.full(Q.shape[0], 1 / Q.shape[0]), "stacking": stack_weights(P),
                 "pseudo-BMA": np.exp(ll - ll.max()) / np.exp(ll - ll.max()).sum(), "best chain": np.eye(Q.shape[0])[np.argmax(ll)]}
        r = dict(seed=seed, family=W["family"], **{k: score(np.tensordot(w, Q, 1), W, ivo) for k, w in rules.items()},
                 weights={k: [round(float(x), 3) for x in w] for k, w in rules.items()},
                 heldout_logscore_spread=float(ll.max() - ll.min()))
        rows.append(r)
        print(seed, W["family"][:10], " | ".join("%s M %.3f IV %.3f" % (k, *r[k]) for k in rules), "| log-score spread %.1f" % r["heldout_logscore_spread"], flush=True)
    tag = os.environ.get("TAG", "mlp")
    json.dump(rows, open(os.path.join(_h, "stacking_%s.json" % tag), "w"), indent=1)
    for k in ("equal", "stacking", "pseudo-BMA", "best chain"):
        print("%-11s Manski %.4f IV %.4f" % (k, np.mean([r[k][0] for r in rows]), np.mean([r[k][1] for r in rows])))
