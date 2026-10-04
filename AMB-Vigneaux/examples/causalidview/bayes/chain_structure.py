"""Is the structure behind the chain plateau short-lived (a few modes, found quickly) or long-tailed?
From the 16 saved chains per world (posterior-mean cells per chain):
  (1) tail test: pooled error err^2(k) over random k-subsets, fitted by  floor + b / k^alpha.  alpha ~ 1 means chain-to-chain
      variation has finite variance (no heavy tail); alpha << 1 would mean rare far-off chains keep moving the mean.
  (2) mode count: chains clustered by the RMS distance between their Manski curves s_c(x) (single linkage at a threshold
      set by the within-cluster spread); the number of distinct modes found as chains are added -- saturating = short-lived.
  (3) outlier chains: share of chains farther than 3x the median distance from the pooled curve; excess kurtosis of deviations."""
import os, sys, glob, json, numpy as np
_h = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_h, "..")); sys.path.insert(0, os.path.join(_h, "../../.."))
import run as R
from scipy.optimize import curve_fit
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import kurtosis

def curves(seed):
    cells = np.concatenate([np.load(f) for f in sorted(glob.glob(os.path.join(_h, "chains", "genericcells_w%d_*.npy" % seed)))])
    return (cells[..., 1, 1] + cells[..., 0, 0]).mean(-1)          # (chains, queries): s_c(x) averaged over I

if __name__ == "__main__":
    rng = np.random.default_rng(0); K = range(1, 17); E = np.zeros((40, 16)); modes = np.zeros((40, 16)); out = []; kur = []; nmodes = []
    for seed in range(40):
        W = R.world(seed); _, Uo = R.manski(W["q"].mean(1)); s = curves(seed); n = len(s)
        for k in K:
            E[seed, k - 1] = np.mean([np.mean((s[rng.choice(n, k, replace=False)].mean(0) - Uo) ** 2) for _ in range(200)])
        D = np.sqrt(((s[:, None, :] - s[None, :, :]) ** 2).mean(-1)); med = np.median(D[np.triu_indices(n, 1)])
        lab = fcluster(linkage(s, "single", metric=lambda a, b: np.sqrt(np.mean((a - b) ** 2))), t=0.5 * med, criterion="distance")
        nmodes.append(len(set(lab)))
        for k in K:                                                    # modes discovered among the first k chains (random orders)
            modes[seed, k - 1] = np.mean([len(set(lab[rng.permutation(n)[:k]])) for _ in range(100)])
        dev = np.sqrt(((s - s.mean(0)) ** 2).mean(-1)); out.append(float(np.mean(dev > 3 * np.median(dev))))
        kur.append(float(kurtosis((s - s.mean(0)).ravel())))
    e = E.mean(0); k = np.arange(1, 17, dtype=float)
    (a1, b1), _ = curve_fit(lambda k, a, b: a + b / k, k, e); (a, b, al), _ = curve_fit(lambda k, a, b, al: a + b / k ** al, k, e, p0=(e[-1], e[0] - e[-1], 1.0))
    pred = a1 + b1 / k; r2 = 1 - np.sum((e - pred) ** 2) / np.sum((e - e.mean()) ** 2)
    m = modes.mean(0)
    summ = dict(err2_by_k=e.tolist(), fit_1_over_k=dict(floor=a1, b=b1, R2=r2), fit_power=dict(floor=a, b=b, alpha=al),
                floor_rmse=float(np.sqrt(max(a1, 0))), modes_found_by_k=m.tolist(), modes_total_mean=float(np.mean(nmodes)),
                modes_total_range=(int(min(nmodes)), int(max(nmodes))), outlier_chain_share=float(np.mean(out)), excess_kurtosis_mean=float(np.mean(kur)))
    json.dump(summ, open(os.path.join(_h, "chain_structure.json"), "w"), indent=1)
    print("err^2(k) = floor + b/k: floor %.5f (RMSE %.4f), b %.5f, R2 %.4f | power fit alpha %.2f" % (a1, np.sqrt(max(a1, 0)), b1, r2, al))
    print("modes found among first k chains (mean over worlds):", np.round(m[[0, 1, 3, 7, 15]], 2), "| total modes per world: mean %.1f, range %s" % (np.mean(nmodes), summ["modes_total_range"]))
    print("outlier chains (>3x median deviation): %.3f | excess kurtosis of chain deviations: %.2f" % (np.mean(out), np.mean(kur)))
