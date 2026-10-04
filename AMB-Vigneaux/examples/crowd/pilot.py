"""Pilot: natural sphere covers in crowdsourced entity resolution (CrowdER 'product' data).

Data: CROWD_DIR = the 'datasets/d_jn-product' folder of github.com/zhydhkcws/crowd_truth_infer (Zheng et al. 2017;
original data Wang et al. 2012): 8,315 product pairs, 176 workers, three workers per item, binary answer 'same product',
gold truth for every item. Because each item is labelled by exactly three workers, four workers whose four triples
each labelled some items form a TETRAHEDRAL cover: every triple observed jointly, all four never (nerve = S^2).
For each such quadruple (all triples >= 15 items):
  CF of the four triple tables; bounds on P(at least 3 of 4 say 'match'); descent defect (overlap disagreement)
  a sampling floor: CF of tables resampled (same counts) from the nearest consistent global law (200 draws)
  the gold rate of each triple's items (a population difference between contexts)
"""
import os, sys, json, itertools, collections
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "../.."))
import numpy as np, pandas as pd
from amb_vigneaux.scenario import Scenario, EmpiricalModel
from amb_vigneaux.outcome import contextual_fraction
from amb_vigneaux.bounds import functional, outcome_bounds
from amb_vigneaux.deficits import nearest_consistent
from amb_vigneaux.higher import betti

rng = np.random.default_rng(0)
D = os.environ.get("CROWD_DIR", "data/crowd_truth_infer/datasets/d_jn-product")
M = ("a", "b", "c", "d"); SC = Scenario({x: (0, 1) for x in M}, tuple(itertools.combinations(M, 3)))

a = pd.read_csv(os.path.join(D, "answer.csv")); gold = pd.read_csv(os.path.join(D, "truth.csv")).set_index("question")["truth"]
per = {q: dict(zip(g.worker, g.answer)) for q, g in a.groupby("question")}
tri = collections.defaultdict(list)
for q, v in per.items():
    for c in itertools.combinations(sorted(v), 3):
        tri[c].append(q)
top = [w for w, _ in collections.Counter(a.worker).most_common(40)]
quads = []
for q4 in itertools.combinations(sorted(top), 4):
    cs = [len(tri.get(c, [])) for c in itertools.combinations(q4, 3)]
    if min(cs) >= 15:
        quads.append(q4)
out = []
f = functional(SC, lambda g: float(sum(g.values()) >= 3))
print(f"{len(per)} items, gold positive rate {gold.mean():.3f}; quadruples forming a tetrahedral cover (all triples >= 15 items): {len(quads)}")
for q4 in quads:
    counts, tabs, gold_rate = {}, {}, []
    for C, trip in zip(SC.contexts, itertools.combinations(q4, 3)):
        t = np.zeros(8)
        for q in tri[trip]:
            v = per[q]; t[4 * v[trip[0]] + 2 * v[trip[1]] + v[trip[2]]] += 1
        counts[C] = t; tabs[C] = (t + .5) / (t + .5).sum(); gold_rate.append(float(gold[tri[trip]].mean()))
    e = EmpiricalModel(SC, tabs); cf = contextual_fraction(e).value; b = outcome_bounds(e, f)
    q = nearest_consistent(e, weights={C: counts[C].sum() for C in SC.contexts})
    null = []
    for _ in range(200):
        tb = {C: (rng.multinomial(int(counts[C].sum()), SC.restriction_matrix(SC.measurements, C) @ q) + .5) for C in SC.contexts}
        null.append(contextual_fraction(EmpiricalModel(SC, {C: v / v.sum() for C, v in tb.items()})).value)
    null = np.array(null)
    r = dict(workers=list(q4), n_items=[int(counts[C].sum()) for C in SC.contexts], betti=betti(SC.contexts), CF=cf,
             CF_null_median=float(np.median(null)), CF_null_q95=float(np.quantile(null, .95)), p_CF=float((null >= cf).mean()),
             bounds=[b["lo"], b["hi"]], defect=b["defect"], gold_rate=gold_rate)
    out.append(r)
    print(f"  n {r['n_items']}  CF {cf:.3f} (sampling floor median {r['CF_null_median']:.3f}, 95% {r['CF_null_q95']:.3f}, p {r['p_CF']:.2f})"
          f"  P(>=3 of 4 'match') in [{b['lo']:.3f}, {b['hi']:.3f}]  gold rate by triple {np.round(gold_rate, 2)}")
ps = np.array([r["p_CF"] for r in out])
print(f"quadruples with CF above the sampling floor at 0.05: {(ps < 0.05).sum()}/{len(out)}")
json.dump(out, open(os.path.join(here, "pilot_results.json"), "w"), indent=1)
