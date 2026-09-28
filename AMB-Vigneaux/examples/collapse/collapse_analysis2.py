"""Detection against a realistic null: the 'fresh' arm (same N, same warm-started EM, but each
generation trains on fresh REAL data, so the model wanders by estimation noise only).
For each metric: null sd = sd of (metric_g - metric_0) over fresh generations 1..19;
flag generation g of an arm when (metric_g - metric_0) <= -3 null sd."""
import json, numpy as np
A_ = json.load(open("cH/collapse_N1000_G20.json")); F = json.load(open("cH/collapse_N1000_G19_fresh.json"))
metrics = {"S_0": lambda r: r["S_alpha_8"]["0.0"], "S_0.5": lambda r: r["S_alpha_8"]["0.5"],
           "Shannon H": lambda r: r["S_alpha_8"]["1.0"], "S_2": lambda r: r["S_alpha_8"]["2.0"],
           "ll_real (needs real data)": lambda r: r["heldout_ll_real"],
           "tail recall (needs truth)": lambda r: r["tail_recall_8"]}
out = {}
print(f"{'metric':>27}{'null sd':>11}{'replace: first flag':>21}{'z@10':>8}{'z@20':>8}{'accumulate: flags':>19}")
for name, f in metrics.items():
    fr = [f(r) for r in F["fresh"]]; d0 = np.array(fr[1:]) - fr[0]
    sd = float(np.sqrt(np.mean(d0 ** 2)))               # RMS drift under the null
    rep = [f(r) for r in A_["replace"]]; acc = [f(r) for r in A_["accumulate"]]
    zr = [(x - rep[0]) / sd for x in rep]; za = [(x - acc[0]) / sd for x in acc]
    first = next((g for g, z in enumerate(zr) if z <= -3), None)
    nacc = sum(z <= -3 for z in za[1:])
    out[name] = dict(null_sd=sd, first_replace=first, z10=zr[10], z20=zr[20], accumulate_flags=int(nacc),
                     fresh_series=fr, replace_series=rep, accumulate_series=acc)
    print(f"{name:>27}{sd:>11.4g}{str(first):>21}{zr[10]:>8.1f}{zr[20]:>8.1f}{nacc:>12d} / 20")
json.dump(out, open("cH/collapse_N1000_analysis2.json", "w"), indent=1)
