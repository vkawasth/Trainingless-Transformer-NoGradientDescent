"""When does gluing the RIGHT regions help? Synthetic: 8 sources x 12 topics, logit p = a_s + b_t, plus source-specific
loops (interactions, sd 1.5) planted in 3 contaminated topics; 30 units per cell; targets = held-out cells of CLEAN
topics. Rules compared (mean |p_hat - p_true|, 40 replications x 16 targets):
  full cover; consistency-pruned region (drop the topic whose removal most improves glue p until the region glues);
  the raise-path end and the lower-path end (regions chosen for the desired outcome); median over glueable regions.
"""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "../.."))
import numpy as np
from amb_vigneaux.lattice_path import evaluate, prune_path

rng = np.random.default_rng(0)
S, T, BAD, N = 8, 12, 3, 30
res = {k: [] for k in ("full cover", "consistency-pruned", "raise-path end", "lower-path end", "full cover glues")}
for rep in range(40):
    a = rng.normal(0, 0.8, S); b = rng.normal(0, 0.8, T); bad = rng.choice(T, BAD, replace=False)
    eta = a[:, None] + b[None, :]
    for t in bad:
        eta[:, t] += rng.normal(0, 1.5, S)
    P = 1 / (1 + np.exp(-eta)); cells = {(s, t): (int(rng.binomial(N, P[s, t])), N) for s in range(S) for t in range(T)}
    clean = [t for t in range(T) if t not in bad]
    for _ in range(16):
        tg = (int(rng.integers(S)), int(rng.choice(clean))); truth = P[tg]
        full = evaluate(cells, tg, list(range(T)))
        res["full cover"].append(abs(full["p"] - truth)); res["full cover glues"].append(full["glues"])
        up = prune_path(cells, tg, list(range(T)), +1, min_size=5); dn = prune_path(cells, tg, list(range(T)), -1, min_size=5)
        if up:
            res["consistency-pruned"].append(abs(up[0]["p"] - truth))          # the first glueable region reached by pruning
            res["raise-path end"].append(abs(up[-1]["p"] - truth))
        if dn:
            res["lower-path end"].append(abs(dn[-1]["p"] - truth))
out = {k: float(np.mean(v)) for k, v in res.items()}
print(f"full cover glues in {out['full cover glues']:.2f} of cases (3 contaminated topics of 12)")
for k in ("full cover", "consistency-pruned", "raise-path end", "lower-path end"):
    print(f"   {k:22s} mean |p_hat - p_true| = {out[k]:.3f}   (n {len(res[k])})")
json.dump(out, open(os.path.join(here, "synthetic_paths_results.json"), "w"), indent=1)
