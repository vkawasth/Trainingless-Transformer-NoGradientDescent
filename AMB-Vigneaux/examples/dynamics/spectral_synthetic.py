"""Which detector sees which kind of phase change in a weekly series (73 weeks before, 73 after)?
Plants after tau: none; LEVEL shift (+0.6 sd); PERSISTENCE change (AR(1) rho 0.2 -> 0.8, same marginal var);
CYCLE (new 8-week oscillation, amplitude 0.6 sd); TREND only (smooth drift, no break; should NOT count as a step).
Detectors: Welch mean test; Slepian-detrended step test (NW=1, plus linear term); multitaper band ratios (NW=3) in
low (<0.05 cyc/wk), mid (0.05-0.15), high (>0.15) bands, each at 0.05/3;
Thomson harmonic F-test for a line component present after but not before (Bonferroni over frequencies).
"""
import sys, json, collections
import numpy as np
sys.path.insert(0, "../..")
from amb_vigneaux.spectral import slepian_step_test, spectral_change, harmonic_ftest
from scipy import stats

def ar1(rng, n, rho):
    e = rng.normal(size=n) * np.sqrt(1 - rho ** 2); x = np.empty(n); x[0] = rng.normal()
    for i in range(1, n): x[i] = rho * x[i - 1] + e[i]
    return x

def gen(rng, plant, n=73):
    a = ar1(rng, n, 0.2)
    b = ar1(rng, n, 0.8 if plant == "persistence" else 0.2)
    if plant == "level": b += 0.6
    if plant == "cycle": b += 0.6 * np.sqrt(2) * np.sin(2 * np.pi * np.arange(n) / 8 + rng.uniform(0, 6.28))
    y = np.concatenate([a, b])
    if plant == "trend": y += np.linspace(-0.6, 0.6, 2 * n)
    return y

if __name__ == "__main__":
    rng = np.random.default_rng(0); reps = 400; n = 73; R = {}
    names = ["Welch mean", "Slepian step", "MT low", "MT mid", "MT high", "line F (after only)"]
    for plant in ("none", "level", "persistence", "cycle", "trend"):
        h = collections.Counter()
        for _ in range(reps):
            y = gen(rng, plant, n)
            h["Welch mean"] += stats.ttest_ind(y[:n], y[n:], equal_var=False).pvalue < 0.05
            h["Slepian step"] += slepian_step_test(y, n, NW=1)["p"] < 0.05
            sc = spectral_change(y[:n], y[n:], NW=3)
            for nm, b in zip(names[2:5], sc): h[nm] += b["p"] < 0.05 / 3
            h["line F (after only)"] += harmonic_ftest(y[n:], NW=3)["p_adj"] < 0.05 and harmonic_ftest(y[:n], NW=3)["p_adj"] >= 0.05
        R[plant] = {k: h[k] / reps for k in names}
        print(f"{plant:>12s} " + "  ".join(f"{k} {R[plant][k]:.2f}" for k in names))
    json.dump(R, open("spectral_synthetic_results.json", "w"), indent=1)
