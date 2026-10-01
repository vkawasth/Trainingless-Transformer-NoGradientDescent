"""Is the outcome a PATH phenomenon or a SURFACE phenomenon on the gluing lattice?  And does feedback (keep going
where a step helped, turn away where it hurt) find the extremes?

For each MBIC target cell, evaluate the prediction eta(U) = logit p_hat over ALL topic regions U containing t* with
|U| >= 5 (about 7,800 regions), and keep the glueable ones. Then:
  first-order structure   fit eta(U) ~ c + sum_x w_x [x in U] (least squares over the glueable regions); R^2 is the
                          share of the landscape explained by independent per-topic influences w_x (degree-1 Walsh
                          weight); adding all pairwise terms gives the degree-2 share.
  surface test            call a move 'aligned' if it adds a topic with w_x > 0 or drops one with w_x < 0. The monotone
                          surface exists if (almost) every aligned move between glueable regions raises eta: then ANY walk
                          of aligned moves raises the probability, not just one path.
  feedback search         hill-climbing with feedback (from a random glueable region, take the best single add/drop that
                          raises eta, stop when none does) vs the exhaustive maximum, 20 random starts per target.
Data via MBIC_XLSX.
"""
import os, sys, json, itertools
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "../.."), os.path.join(here, "../bias3")]
import numpy as np
from amb_vigneaux.lattice_path import evaluate
import mbic

rng = np.random.default_rng(0)
d = mbic.load()
s = d.groupby("sentence_id").agg(y=("b", "mean"), o=("outlet", "first"), t=("topic", "first"))
s["lab"] = (s.y >= 0.5).astype(int)
cells = {(o, t): (int(g.lab.sum()), int(len(g))) for (o, t), g in s.groupby(["o", "t"])}
topics = sorted(s.t.unique())
targets = sorted([c for c, (k, n) in cells.items() if n >= 15], key=lambda c: -cells[c][1])[:12]


def landscape(tg):
    others = [x for x in topics if x != tg[1]]; L = {}
    for k in range(4, len(others) + 1):
        for comb in itertools.combinations(others, k):
            e = evaluate(cells, tg, (tg[1],) + comb)
            if e is not None and e["glues"]:
                L[frozenset(comb)] = np.log(e["p"] / (1 - e["p"]))
    return others, L


def analyse(tg):
    others, L = landscape(tg)
    keys = list(L); y = np.array([L[k] for k in keys])
    X1 = np.array([[1.0] + [1.0 if x in k else 0.0 for x in others] for k in keys])
    c1, *_ = np.linalg.lstsq(X1, y, rcond=None); r2_1 = 1 - ((y - X1 @ c1) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    pairs = list(itertools.combinations(range(len(others)), 2))
    X2 = np.hstack([X1, np.array([[X1[i, a + 1] * X1[i, b + 1] for a, b in pairs] for i in range(len(keys))])])
    c2, *_ = np.linalg.lstsq(X2, y, rcond=None); r2_2 = 1 - ((y - X2 @ c2) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    w = dict(zip(others, c1[1:]))
    aligned = up = 0
    for k in keys:                                     # aligned single moves between glueable regions
        for x in others:
            k2 = (k - {x}) if x in k else (k | {x})
            if k2 not in L:
                continue
            is_aligned = (x not in k and w[x] > 0) or (x in k and w[x] < 0)
            if is_aligned:
                aligned += 1; up += L[k2] > L[k]
    gmax = max(y); hits = []; steps = []
    for _ in range(20):
        k = keys[int(rng.integers(len(keys)))]; n = 0
        while True:
            nb = [(k - {x}) if x in k else (k | {x}) for x in others]
            nb = [z for z in nb if z in L and L[z] > L[k]]
            if not nb:
                break
            k = max(nb, key=lambda z: L[z]); n += 1
        hits.append(abs(L[k] - gmax) < 1e-9); steps.append(n)
    p = lambda e: 1 / (1 + np.exp(-e))
    return dict(target=list(tg), n_glueable=len(keys), r2_first_order=float(r2_1), r2_second_order=float(r2_2),
                aligned_moves=aligned, frac_aligned_raise=up / max(aligned, 1), feedback_hit_max=float(np.mean(hits)),
                feedback_steps=float(np.mean(steps)), p_min=float(p(min(y))), p_max=float(p(gmax)),
                raisers=[x for x in others if w[x] < 0], adders=[x for x in others if w[x] > 0])


if __name__ == "__main__":
    out = []
    for tg in targets:
        r = analyse(tg); out.append(r)
        print(f"{tg[0]:>10s} x {tg[1]:<38s} glueable {r['n_glueable']:5d}  R2 first-order {r['r2_first_order']:.3f}"
              f" (+pairs {r['r2_second_order']:.3f})  aligned moves that raise p {r['frac_aligned_raise']:.3f}"
              f"  feedback finds max {r['feedback_hit_max']:.2f} ({r['feedback_steps']:.1f} steps)  p in [{r['p_min']:.2f}, {r['p_max']:.2f}]")
    print(f"\nmedian: R2 first-order {np.median([r['r2_first_order'] for r in out]):.3f}, second-order {np.median([r['r2_second_order'] for r in out]):.3f};"
          f"  aligned moves raising p {np.median([r['frac_aligned_raise'] for r in out]):.3f};  feedback reaches the global max in"
          f" {np.mean([r['feedback_hit_max'] for r in out]):.2f} of starts")
    json.dump(out, open(os.path.join(here, "landscape_results.json"), "w"), indent=1)
