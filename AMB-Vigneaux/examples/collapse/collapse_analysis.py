"""Which lens sees collapse first? z-scores of each generation's metric against the
sampling noise of that metric (5 independent NEVAL samples from the generation-0 model)."""
import json, sys, numpy as np
exec(open("collapse.py").read().split("# ------------------------------------------------------------------ run")[0].replace("N = int(sys.argv[1])", "N = 1 #").replace("GENS = int(sys.argv[2])", "GENS = 1 #"))
tag = sys.argv[1]
R = json.load(open(f"cH/collapse_{tag}.json"))
keys = [("S_alpha_8", "0.0", "S_0"), ("S_alpha_8", "0.5", "S_0.5"), ("S_alpha_8", "1.0", "Shannon H"),
        ("S_alpha_8", "2.0", "S_2"), ("tail_recall_8", None, "tail recall*"), ("recall_8", None, "recall*")]
reps = [data_metrics(sample(GEN0, NEVAL)) for _ in range(5)]
def get(m, k, a):
    v = m[k]
    return v[a] if a is not None and isinstance(v, dict) and a in v else (v[float(a)] if a is not None else v)
sd = {name: np.std([get(r, k, a) for r in reps], ddof=1) for k, a, name in keys}
# held-out ll noise: per-sequence sd / sqrt(n_val)
b = inside(VA, GEN0)[L][:, 0]; per = np.log(b.mean(1)); sd["ll_real"] = per.std(ddof=1) / np.sqrt(len(per))
out = {}
for arm in ("replace", "accumulate"):
    rows = R[arm]; base = rows[0]
    print(f"\n{tag} {arm}: first generation with z <= -3 (z at the last generation)")
    for k, a, name in keys + [("heldout_ll_real", None, "ll_real")]:
        if name == "ll_real":
            series = [r["heldout_ll_real"] for r in rows]
        else:
            series = [get(r, k, a) for r in rows]
        z = [(x - series[0]) / sd[name] for x in series]
        first = next((g for g, zz in enumerate(z) if zz <= -3), None)
        out.setdefault(arm, {})[name] = dict(first=first, z_last=z[-1], rel_change_last=(series[-1] - series[0]) / abs(series[0]))
        print(f"  {name:>13}: first {str(first):>4}   z_last {z[-1]:8.1f}   relative change {100*(series[-1]-series[0])/abs(series[0]):7.2f}%")
json.dump(dict(sd={k: float(v) for k, v in sd.items()}, **out), open(f"cH/collapse_{tag}_analysis.json", "w"), indent=1)
print("\n* recall metrics need the true support; not available on real corpora")
