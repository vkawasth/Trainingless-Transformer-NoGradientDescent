"""Paths over the MBIC gluing lattice that raise or lower the predicted probability that a held-out outlet x topic
cell is majority-biased; envelopes over all glueable regions; and a check against the held-out truth.
Data via MBIC_XLSX.
"""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "../.."), os.path.join(here, "../bias3")]
import numpy as np
from amb_vigneaux.lattice_path import envelope, greedy_path, evaluate, prune_path
import mbic

d = mbic.load()
s = d.groupby("sentence_id").agg(y=("b", "mean"), o=("outlet", "first"), t=("topic", "first"))
s["lab"] = (s.y >= 0.5).astype(int)
cells = {}
for (o, t), g in s.groupby(["o", "t"]):
    cells[(o, t)] = (int(g.lab.sum()), int(len(g)))
topics = sorted(s.t.unique()); outlets = sorted(s.o.unique())
targets = [c for c, (k, n) in cells.items() if n >= 10]
R = dict(targets=[]); rules = {}
MIN = int(os.environ.get("MIN_SIZE", 5))
print(f"regions: glueable topic sets of {MIN}..7 topics containing the target's topic (the target cell itself held out)")
print(f"MBIC: {len(cells)} outlet x topic cells, {len(targets)} targets with >= 10 sentences")
cover = inside = 0; spreads = []; err_full = []; err_mid = []; flips = 0
for tg in targets:
    k, n = cells[tg]; truth = k / n
    env = envelope(cells, tg, topics, max_size=8, min_size=MIN)
    full = evaluate(cells, tg, topics)
    up = prune_path(cells, tg, topics, +1, min_size=MIN); dn = prune_path(cells, tg, topics, -1, min_size=MIN)
    if env["hi"] is None:
        continue
    lo, hi = env["lo"]["p"], env["hi"]["p"]; cover += 1; inside += lo <= truth <= hi; spreads.append(hi - lo)
    if full is not None:
        err_full.append(abs(full["p"] - truth))
    err_mid.append(abs(np.median(env["p_values"]) - truth))
    regs = env["regions"]
    best_fit = max(regs, key=lambda r: (r[1], r[2]))                # the most consistent region (highest glue p)
    rules.setdefault("most consistent region", []).append(abs(best_fit[0] - truth))
    rules.setdefault("raise-path end (max p)", []).append(abs(env["hi"]["p"] - truth))
    rules.setdefault("lower-path end (min p)", []).append(abs(env["lo"]["p"] - truth))
    wts = np.array([r[1] for r in regs]); pv = np.array([r[0] for r in regs])
    rules.setdefault("glue-p weighted average", []).append(abs(float((wts * pv).sum() / wts.sum()) - truth))
    flips += (lo < 0.5 < hi)
    R["targets"].append(dict(target=list(tg), truth=truth, n=n, lo=lo, hi=hi, lo_region=env["lo"]["region"], hi_region=env["hi"]["region"],
                             n_glueable=env["n_glueable"], full=full["p"] if full else None, full_glues=full["glues"] if full else None,
                             up_path=[(x["dropped"], round(x["p"], 3)) for x in up], down_path=[(x["dropped"], round(x["p"], 3)) for x in dn]))
R.update(n=cover, frac_truth_inside=inside / cover, median_spread=float(np.median(spreads)),
         frac_outcome_flippable=flips / cover, mae_full=float(np.mean(err_full)), mae_median_region=float(np.mean(err_mid)))
print(f"targets evaluated {cover}: the held-out truth lies inside the envelope for {inside}/{cover}; median envelope width "
      f"{np.median(spreads):.3f};  outcome [p > 1/2] can be flipped by region choice for {flips}/{cover} targets")
print(f"prediction error (mean |p_hat - truth|): full cover {np.mean(err_full):.3f};  median over glueable regions {np.mean(err_mid):.3f};  "
      + ";  ".join(f"{k} {np.mean(v):.3f}" for k, v in rules.items()))
R["rule_mae"] = {k: float(np.mean(v)) for k, v in rules.items()}
show = sorted(R["targets"], key=lambda r: -(r["hi"] - r["lo"]))[:4]
for r in show:
    print(f"\n  {r['target'][0]} x {r['target'][1]}: truth {r['truth']:.2f} (n {r['n']}); envelope [{r['lo']:.2f}, {r['hi']:.2f}] over {r['n_glueable']} glueable regions;"
          f" full cover {r['full']:.2f} (glues {r['full_glues']})")
    print(f"     raise path (drop topics from the full cover): {r['up_path']}")
    print(f"     lower path: {r['down_path']}")
json.dump(R, open(os.path.join(here, f"mbic_paths_min{MIN}_results.json"), "w"), indent=1, default=str)
