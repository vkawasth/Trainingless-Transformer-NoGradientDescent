"""Estimate edge moduli |s_i| and CF of a binary 4-cycle from N samples per edge:
Tsallis-2 (unbiased I_2), Shannon plug-in, Shannon Miller–Madow, and the labelled direct
estimator (sample correlation; uses outcome labels, so it is NOT a Layer-2 estimator)."""
import json, numpy as np
from amb_vigneaux.magphase import *

rng = np.random.default_rng(0)
configs = {"strong (PR-like)": [0.95, 0.9, 0.9, -0.95], "near threshold": [0.8, 0.7, 0.75, -0.7],
           "weak edges": [0.3, 0.5, 0.2, -0.4]}
Ns, R = (30, 100, 300, 1000), 400
res = {}
for name, s in configs.items():
    s = np.array(s); odd = np.prod(s) < 0
    cf_true = cf_cycle_from_moduli(np.abs(s), odd)
    for N in Ns:
        est = {k: {"s2": [], "cf": []} for k in ("tsallis2", "shannon", "shannon_mm", "labelled")}
        for _ in range(R):
            T = [sample_binary_edge(si, N, rng) for si in s]
            m = {"tsallis2": [modulus_from_i2(i2_unbiased(t)) for t in T],
                 "shannon": [modulus_from_mi(mi_plugin(t)) for t in T],
                 "shannon_mm": [modulus_from_mi(mi_plugin(t, True)) for t in T],
                 "labelled": [abs((t[0, 0] + t[1, 1] - t[0, 1] - t[1, 0]) / N) for t in T]}
            for k in est:
                est[k]["s2"].append(np.array(m[k]) ** 2 - s ** 2)
                est[k]["cf"].append(cf_cycle_from_moduli(m[k], odd) - cf_true)
        for k, v in est.items():
            e2, ec = np.array(v["s2"]), np.array(v["cf"])
            res[(name, N, k)] = dict(bias_s2=float(e2.mean()), rmse_s2=float(np.sqrt((e2 ** 2).mean())),
                                    bias_cf=float(ec.mean()), rmse_cf=float(np.sqrt((ec ** 2).mean())))
    print(f"\n{name}: s = {s.tolist()}, CF = {cf_true:.3f}")
    print(f"{'N':>6}{'estimator':>13}{'bias s²':>10}{'RMSE s²':>10}{'bias CF':>10}{'RMSE CF':>10}")
    for N in Ns:
        for k in ("tsallis2", "shannon", "shannon_mm", "labelled"):
            r = res[(name, N, k)]
            print(f"{N:>6}{k:>13}{r['bias_s2']:>10.4f}{r['rmse_s2']:>10.4f}{r['bias_cf']:>10.4f}{r['rmse_cf']:>10.4f}")
json.dump({"|".join(map(str, k)): v for k, v in res.items()}, open("examples/s2_vs_shannon.json", "w"), indent=1)
