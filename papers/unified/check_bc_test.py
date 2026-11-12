"""check_bc_test.py -- synthetic validation of the finite-sample Beck-Chevalley test used in Section 17 (Studies 1 and 2).

The test: binary Y_a, Y_b, centre M (a tuple of discrete variables). Windows sigma(Y_a, M), sigma(M, Y_b), meet sigma(M).
By Thm 14.4 / check H1, Beck-Chevalley holds iff Y_a _||_ Y_b | M. Statistic: cbar = mass-weighted |phi(Y_a, Y_b | M = m)| over strata
with non-degenerate margins. Null: permute Y_b within strata of M. Within a stratum this keeps both margins, so the permuted table's
n11 is hypergeometric; we sample it directly (exact and fast). The defect norm (Friedrichs cosine, L4b) is c = max_m |phi_m|.

  S1  the hypergeometric null equals the explicit within-stratum permutation null (same distribution, KS test on 4000 draws)
  S2  population side: cbar_pop = 0 and c_pop = 0 exactly for every Markov law (Y_a - M - Y_b), > 0 for latent mixtures
  S3  size: when Beck-Chevalley HOLDS (Markov chains; latent included in the centre) the test rejects at rate ~ 0.05 (never above the
      binomial 99% bound), and its p-values are uniform (KS)
  S4  power: when Beck-Chevalley FAILS (item-level latent difficulty, Y_k iid given the item) rejection -> 1 as n grows,
      and the sample cbar converges to the population cbar
  S5  centre enlargement: putting the latent into the centre restores Beck-Chevalley (size ~ 0.05); adding the partial
      proxy D = Y4 + Y5 only shrinks the defect (population cbar strictly between 0 and the one-sample value)
  S6  calibrated replica of Study 2: a latent-class binomial model fitted to the observed wrong-count distribution alone reproduces
      the observed cbar with centre Y2 (0.32) and with centre (Y2, D) (0.13) within simulation intervals
Run: python check_bc_test.py   (about a minute)
"""
import itertools, json, numpy as np
from scipy.stats import kstest, binom, ks_2samp
rng = np.random.default_rng(20261013)
results = []
def check(name, cond, info=""):
    results.append((name, bool(cond))); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""))

# ---------------------------------------------------------------- the test
def strata(Y, a, m, b):
    keys, idx = np.unique(Y[:, m], axis=0, return_inverse=True); idx = idx.ravel(); k = len(keys)
    n = np.bincount(idx, minlength=k).astype(float)
    r1 = np.bincount(idx, weights=Y[:, a], minlength=k); c1 = np.bincount(idx, weights=Y[:, b], minlength=k)
    n11 = np.bincount(idx, weights=Y[:, a] * Y[:, b], minlength=k)
    return n, r1, c1, n11
def cbar_tab(n, r1, c1, n11):
    """n, r1, c1 shape (k,), n11 shape (..., k): weighted |phi| over non-degenerate strata."""
    ok = (r1 > 0) & (r1 < n) & (c1 > 0) & (c1 < n)
    if not ok.any(): return np.full(np.shape(n11)[:-1], np.nan)
    den = np.sqrt(r1 * (n - r1) * c1 * (n - c1))
    phi = np.where(ok, (n * n11 - r1 * c1) / np.where(ok, den, 1), 0.0)   # n11*n00 - n10*n01 = n*n11 - r1*c1
    return (np.abs(phi) * np.where(ok, n, 0)).sum(-1) / n[ok].sum()
def bc_test(Y, a, m, b, B=1000, rng=rng, return_null=False):
    n, r1, c1, n11 = strata(Y, a, m, b); obs = float(cbar_tab(n, r1, c1, n11))
    null_n11 = np.stack([rng.hypergeometric(c1[j], n[j] - c1[j], r1[j], size=B) if r1[j] > 0 else np.zeros(B)
                         for j in range(len(n))], -1).astype(float)
    null = cbar_tab(n, r1, c1, null_n11)
    p = float((1 + (null >= obs - 1e-12).sum()) / (B + 1))
    return (obs, p, null) if return_null else (obs, p)
def perm_explicit(Y, a, m, b, B, rng):
    keys, idx = np.unique(Y[:, m], axis=0, return_inverse=True); idx = idx.ravel(); out = []
    for _ in range(B):
        Z = Y.copy()
        for k in range(len(keys)):
            w = np.where(idx == k)[0]; Z[w, b] = Z[rng.permutation(w), b]
        out.append(float(cbar_tab(*strata(Z, a, m, b))))
    return np.array(out)

# ---------------------------------------------------------------- generators
def markov(n, rng):
    """Y2 ~ Bern(q); Y1 | Y2 and Y3 | Y2 independent, random kernels: Beck-Chevalley holds."""
    q = rng.uniform(0.15, 0.6); k1 = rng.uniform(0.05, 0.95, 2); k3 = rng.uniform(0.05, 0.95, 2)
    y2 = (rng.random(n) < q).astype(int)
    return np.column_stack([(rng.random(n) < k1[y2]).astype(int), y2, (rng.random(n) < k3[y2]).astype(int)])
def latent(n, w, p, rng, K=5):
    """item class z ~ w; Y_1..Y_K iid Bern(p[z]) given the item (the HaluEval situation). Returns Y (n x K) and z."""
    z = rng.choice(len(w), n, p=w); Y = (rng.random((n, K)) < np.asarray(p)[z][:, None]).astype(int); return Y, z
def pop_cbar_latent(w, p, centre):
    """exact population cbar and c for the latent model; centre in {'Y2', 'Y2D', 'Y2Z'}. Columns: Y1, Y2, Y3, Y4, Y5."""
    tab = {}
    for z, (wz, pz) in enumerate(zip(w, p)):
        for y in itertools.product((0, 1), repeat=5):
            pr = wz * np.prod([pz if v else 1 - pz for v in y])
            key = {"Y2": (y[1],), "Y2D": (y[1], y[3] + y[4]), "Y2Z": (y[1], z)}[centre]
            t = tab.setdefault(key, np.zeros((2, 2))); t[y[0], y[2]] += pr
    num = den = 0.0; cmax = 0.0
    for t in tab.values():
        r0, r1, c0, c1 = t[0].sum(), t[1].sum(), t[:, 0].sum(), t[:, 1].sum()
        if min(r0, r1, c0, c1) < 1e-15: continue
        f = abs(t[0, 0] * t[1, 1] - t[0, 1] * t[1, 0]) / np.sqrt(r0 * r1 * c0 * c1)
        num += t.sum() * f; den += t.sum(); cmax = max(cmax, f)
    return num / den, cmax
def pop_markov(q, k1, k3):
    num = den = 0.0
    for y2 in (0, 1):
        pm = q if y2 else 1 - q; a, c = k1[y2], k3[y2]
        t = pm * np.array([[(1 - a) * (1 - c), (1 - a) * c], [a * (1 - c), a * c]])
        f = abs(t[0, 0] * t[1, 1] - t[0, 1] * t[1, 0]) / np.sqrt(t[0].sum() * t[1].sum() * t[:, 0].sum() * t[:, 1].sum())
        num += pm * f; den += pm
    return num / den

# ---------------------------------------------------------------- S1
Y, _ = latent(400, [0.8, 0.2], [0.05, 0.8], rng)
_, _, nh = bc_test(Y, 0, [1], 2, B=4000, return_null=True); ne = perm_explicit(Y, 0, [1], 2, 4000, np.random.default_rng(1))
ks = ks_2samp(nh, ne)
check("S1 hypergeometric null = explicit within-stratum permutation null", ks.pvalue > 0.01,
      f"KS p = {ks.pvalue:.3f}; medians {np.median(nh):.4f} vs {np.median(ne):.4f}")

# ---------------------------------------------------------------- S2
mk = [pop_markov(rng.uniform(.1, .9), rng.uniform(0, 1, 2), rng.uniform(0, 1, 2)) for _ in range(200)]
lt = [pop_cbar_latent([0.8, 0.2], [0.05, 0.8], c) for c in ("Y2", "Y2D", "Y2Z")]
check("S2 population cbar = 0 for all 200 random Markov laws; > 0 for the latent mixture; = 0 with the latent in the centre",
      max(mk) < 1e-12 and lt[0][0] > 0.05 and lt[2][0] < 1e-12,
      f"max Markov {max(mk):.1e}; latent: Y2 {lt[0][0]:.3f}, (Y2,D) {lt[1][0]:.3f}, (Y2,Z) {lt[2][0]:.1e}")

# ---------------------------------------------------------------- S3 size
def size(gen, R=1000, B=500):
    ps = np.array([gen() for _ in range(R)]); return float((ps <= 0.05).mean()), ps
hi = binom.ppf(0.995, 1000, 0.05) / 1000
out = {}
for n in (300, 1800):
    rate, ps = size(lambda: bc_test(markov(n, rng), 0, [1], 2, B=500)[1])
    out[f"markov_n{n}"] = rate
    check(f"S3 size, Markov chains n = {n}: rejection rate <= binomial 99% bound {hi:.3f}; p-values ~ uniform",
          rate <= hi and kstest(ps, "uniform").pvalue > 1e-3,
          f"rate {rate:.3f}, KS p {kstest(ps, 'uniform').pvalue:.3f}")

# ---------------------------------------------------------------- S4 power and consistency
w, p = [0.8, 0.2], [0.05, 0.8]; cpop = pop_cbar_latent(w, p, "Y2")[0]; power = {}; est = {}
for n in (100, 300, 1000, 3000):
    obs, ps = [], []
    for _ in range(200):
        Y, _ = latent(n, w, p, rng); o, pv = bc_test(Y, 0, [1], 2, B=500); obs.append(o); ps.append(pv)
    power[n] = float((np.array(ps) <= 0.05).mean()); est[n] = float(np.mean(obs))
check("S4 power rises to 1 under latent difficulty (Beck-Chevalley fails); sample cbar -> population cbar",
      power[3000] > 0.99 and power[100] < power[1000] and abs(est[3000] - cpop) < 0.03,
      f"population cbar {cpop:.3f}; power {power}; mean cbar {{n: v}} = { {k: round(v, 3) for k, v in est.items()} }")

# ---------------------------------------------------------------- S5 centre enlargement
rateZ, _ = size(lambda: (lambda Y, z: bc_test(np.column_stack([Y, z]), 0, [1, 5], 2, B=500)[1])(*latent(1800, w, p, rng)), R=1000)
pd = pop_cbar_latent(w, p, "Y2D")[0]
check("S5 latent in the centre restores Beck-Chevalley (size ~ 0.05); proxy D = Y4+Y5 only shrinks the defect",
      rateZ <= hi and 0 < pd < cpop, f"size with (Y2,Z): {rateZ:.3f}; population cbar Y2 {cpop:.3f} > (Y2,D) {pd:.3f} > 0")

# ---------------------------------------------------------------- S6 calibrated replica of Study 2
counts = np.array([1441, 60, 41, 34, 41, 183]); N = counts.sum(); ks_ = np.arange(6)
def fit(J, iters=3000):
    best = None
    for start in range(20):
        r = np.random.default_rng(start); wj = r.dirichlet(np.ones(J)); pj = np.sort(r.uniform(0.01, 0.99, J))
        for _ in range(iters):
            L = wj[None, :] * binom.pmf(ks_[:, None], 5, pj[None, :]); R_ = L / L.sum(1, keepdims=True)
            wj = (counts[:, None] * R_).sum(0) / N; pj = np.clip((counts[:, None] * R_ * ks_[:, None]).sum(0) / (5 * (counts[:, None] * R_).sum(0)), 1e-6, 1 - 1e-6)
        ll = (counts * np.log((wj[None, :] * binom.pmf(ks_[:, None], 5, pj[None, :])).sum(1))).sum()
        if best is None or ll > best[0]: best = (ll, wj, pj)
    return best
ll3, w3, p3 = fit(3)
expct = N * (w3[None, :] * binom.pmf(ks_[:, None], 5, p3[None, :])).sum(1)
sims = {"Y2": [], "Y2D": []}
for _ in range(300):
    Y, _ = latent(N, w3, p3, rng)
    sims["Y2"].append(float(cbar_tab(*strata(Y, 0, [1], 2))))
    Yd = np.column_stack([Y[:, :3], Y[:, 3] + Y[:, 4]]); sims["Y2D"].append(float(cbar_tab(*strata(Yd, 0, [1, 3], 2))))
iv = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in sims.items()}
obs_c = {"Y2": 0.3205, "Y2D": 0.1271}                     # Study 2, confirmatory set (study2_amendment1_results.json)
check("S6 a latent-class model fitted to the wrong-count distribution alone reproduces Study 2's cbar (centre Y2 and (Y2,D))",
      all(iv[k][0] <= obs_c[k] <= iv[k][1] for k in obs_c),
      f"classes w={np.round(w3, 3).tolist()} p={np.round(p3, 3).tolist()}; fitted counts {np.round(expct).astype(int).tolist()} vs "
      f"{counts.tolist()}; 95% sim intervals Y2 [{iv['Y2'][0]:.3f}, {iv['Y2'][1]:.3f}] (obs 0.321), "
      f"(Y2,D) [{iv['Y2D'][0]:.3f}, {iv['Y2D'][1]:.3f}] (obs 0.127)")

json.dump(dict(size=out, size_latent_in_centre=rateZ, power=power, mean_cbar=est, pop_cbar=dict(Y2=cpop, Y2D=pd),
               replica=dict(w=w3.tolist(), p=p3.tolist(), fitted_counts=expct.tolist(), intervals=iv)),
          open("check_bc_test.json", "w"), indent=1)
print(f"\n{sum(ok for _, ok in results)}/{len(results)} checks passed")
