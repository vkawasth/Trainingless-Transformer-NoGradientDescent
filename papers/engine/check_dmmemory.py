"""Checks for 'Decision-maker memory' (companion note; module pce/dmmemory.py). E = 1000 episodes per history; boundaries calibrated
on honest histories with seeds 10001-10400; evaluation on seeds 1-200 (disjoint).
  DM1 IDENTITY  honest increments have mean 0 in every family, and the walks grow like sqrt(n) (variance of the walk over n near constant)
  DM2 IDENTITY  the increments do not depend on the common shocks at all (median-centred signs are invariant to a shift shared by all sources)
  DM3 FINDING   out-of-sample false-alarm rate at most 7% per family on honest histories (nominal 5%)
  DM4 FINDING   a single source tilted towards X (by 0.012 against noise 0.05) is flagged by the tilt walk in at least 90% of histories, and
                named correctly in at least 90% of those; the co-revision and order walks stay at the null rate
  DM5 FINDING   a tilting coalition with a fixed order of revision is flagged by the order walk in every history, early, naming the pair
  DM6 FINDING   the same coalition with an exchangeable order ('braidable') is invisible to the order walk (null rate), while its tilt is caught
  DM7 FINDING   a follower that copies a partner's deviation without any tilt: tilt walk at the null rate; co-revision and order walks flag
                the right pair
  DM8 FINDING   a primed memory (watching the declared relations of B and C) flags earlier at the same false-alarm rate
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import dmmemory as DM
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
E = 1000; CAL = range(10001, 10401); EV = range(1, 201)
P = DM.PAIRS; pBC = P.index((1, 2)); WATCH = {"tilt": [1, 2], "corev": [pBC], "order": [pBC]}

# DM1
means = {f: [] for f in ("tilt", "corev", "order")}; var_ratio = {f: [] for f in means}
ends = {f: {n: [] for n in (100, 1000)} for f in means}
for sd in CAL:
    inc = DM.simulate("honest", E, sd)
    for f in means:
        means[f].append(inc[f].mean()); S = np.cumsum(inc[f], axis=0)
        for n in (100, 1000): ends[f][n].append(S[n - 1].ravel())
for f in means:
    v100 = np.var(np.concatenate(ends[f][100])) / 100; v1000 = np.var(np.concatenate(ends[f][1000])) / 1000; var_ratio[f] = v1000 / v100
check("DM1 IDENTITY honest walks: mean-zero increments, variance growing linearly in n",
      all(abs(np.mean(means[f])) < 0.01 for f in means) and all(0.85 < var_ratio[f] < 1.15 for f in means),
      f"mean increment { {f: round(float(np.mean(v)), 4) for f, v in means.items()} }; var(S_1000)/1000 over var(S_100)/100 { {f: round(float(r), 3) for f, r in var_ratio.items()} }")

# DM2
same = all(np.array_equal(DM.simulate("honest", 200, sd)[f], DM.simulate("shocks", 200, sd)[f]) for sd in range(1, 21) for f in means)
check("DM2 IDENTITY increments are exactly invariant to common shocks", same, "20 seeds, shocks of size 0.05 vs 0.2")

B = DM.calibrate(E, CAL); BW = DM.calibrate(E, CAL, WATCH)
R = {}
for sc in DM.SCENARIOS:
    R[sc] = {"plain": [DM.run(sc, E, sd, B)[0] for sd in EV], "primed": [DM.run(sc, E, sd, BW, WATCH)[0] for sd in EV]}
def rate(sc, f, mode="plain"): return float(np.mean([f in a for a in R[sc][mode]]))
def med(sc, f, mode="plain"):
    v = [a[f][0] for a in R[sc][mode] if f in a]; return float(np.median(v)) if v else None
def right(sc, f, ok):
    v = [ok(a[f][1]) for a in R[sc]["plain"] if f in a]; return float(np.mean(v)) if v else 0.0
summ = {sc: {f: dict(rate=rate(sc, f), median=med(sc, f), rate_primed=rate(sc, f, "primed"), median_primed=med(sc, f, "primed")) for f in means} for sc in DM.SCENARIOS}
json.dump(dict(bounds=B, bounds_primed=BW, results=summ), open(os.path.join(HERE, "dmmemory_summary.json"), "w"), indent=1)

# DM3
fa = {f: max(rate("honest", f), rate("shocks", f)) for f in means}; fap = {f: rate("honest", f, "primed") for f in means}
check("DM3 FINDING false alarms on honest evaluation histories at most 7% per family (plain and primed)", max(fa.values()) <= 0.07 and max(fap.values()) <= 0.07,
      f"boundaries { {f: round(b, 2) for f, b in B.items()} }; false-alarm rates {fa}; primed {fap}")
# DM4
check("DM4 FINDING a single tilted source is caught by the tilt walk and named; the other walks stay null",
      rate("tilt", "tilt") >= 0.9 and right("tilt", "tilt", lambda j: j == 1) >= 0.9 and rate("tilt", "corev") <= 0.07 and rate("tilt", "order") <= 0.07,
      f"tilt walk {rate('tilt', 'tilt'):.2f} (median episode {med('tilt', 'tilt')}), names B {right('tilt', 'tilt', lambda j: j == 1):.2f}; co-revision {rate('tilt', 'corev'):.2f}; order {rate('tilt', 'order'):.2f}")
# DM5
check("DM5 FINDING a fixed-order coalition is caught by the order walk in every history, early, with the right pair",
      rate("ordered", "order") == 1.0 and med("ordered", "order") <= 40 and right("ordered", "order", lambda j: j == pBC) == 1.0,
      f"order walk {rate('ordered', 'order'):.2f}, median episode {med('ordered', 'order')}; tilt walk {rate('ordered', 'tilt'):.2f}")
# DM6
check("DM6 FINDING an exchangeable-order ('braidable') coalition is invisible to the order walk; its tilt is still caught",
      rate("braidable", "order") <= 0.07 and rate("braidable", "tilt") >= 0.9,
      f"order walk {rate('braidable', 'order'):.2f}; tilt walk {rate('braidable', 'tilt'):.2f} (median episode {med('braidable', 'tilt')})")
# DM7
check("DM7 FINDING a follower without tilt: tilt walk null; co-revision and order walks catch the right pair",
      rate("follower", "tilt") <= 0.07 and rate("follower", "corev") >= 0.9 and right("follower", "corev", lambda j: j == pBC) >= 0.9 and rate("follower", "order") == 1.0,
      f"tilt {rate('follower', 'tilt'):.2f}; co-revision {rate('follower', 'corev'):.2f} (right pair {right('follower', 'corev', lambda j: j == pBC):.2f}, median {med('follower', 'corev')}); order {rate('follower', 'order'):.2f}")
# DM8
pairs8 = [("tilt", "tilt"), ("ordered", "tilt"), ("follower", "corev"), ("ordered", "order")]
check("DM8 FINDING a primed memory flags earlier at the same false-alarm rate",
      all(med(sc, f, "primed") < med(sc, f) for sc, f in pairs8) and max(fap.values()) <= 0.07,
      f"median episode plain -> primed: { {sc + '/' + f: (med(sc, f), med(sc, f, 'primed')) for sc, f in pairs8} }")
print(f"{sum(results)}/{len(results)} passed")
