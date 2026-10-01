"""Figure: the loop-content surface on the gluing lattice, and how gluing regions lifts (or lowers) the probability.
One MBIC target (outlet x topic held out). Panels:
  (a) loop content: each topic's influence w_x on logit p_hat (first-order landscape), against the target outlet's
      weighted loop residual on that topic;
  (b) the surface: every glueable region (>= 5 topics) placed by its first-order score sum_{x in U} w_x against its
      actual p_hat; the raise path, the lower path and the full cover marked; held-out truth and the 1/2 line;
  (c) the paths step by step (dropping topics from the full cover).
Data via MBIC_XLSX.
"""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "../.."), os.path.join(here, "../bias3"), here]
import numpy as np
import landscape as LS
from amb_vigneaux.lattice_path import fit_region, _logit, _without, prune_path, evaluate

TG = (os.environ.get("OUTLET", "breitbart"), os.environ.get("TOPIC", "gender"))
cells = LS.cells; topics = LS.topics
others, L = LS.landscape(TG)
keys = list(L); y = np.array([L[k] for k in keys])
X = np.array([[1.0] + [1.0 if x in k else 0.0 for x in others] for k in keys])
c, *_ = np.linalg.lstsq(X, y, rcond=None); w = dict(zip(others, c[1:]))
cw = _without(cells, TG); full = fit_region(cw, sorted({a for a, _ in cells}), set(topics)); S, T = full["S"], full["T"]
res = {}
for x in others:
    yx, wx = _logit(*cw[(TG[0], x)]) if (TG[0], x) in cw else (0.0, 0.0)
    res[x] = wx * (yx - full["coef"][S.index(TG[0])] - full["coef"][len(S) + T.index(x)]) if (TG[0], x) in cw else 0.0
k, n = cells[TG]; truth = k / n
up = prune_path(cells, TG, topics, +1, min_size=5); dn = prune_path(cells, TG, topics, -1, min_size=5)
fe = evaluate(cells, TG, topics)
expit = lambda e: 1 / (1 + np.exp(-e))
score = np.array([sum(w[x] for x in kk) for kk in keys])
def reg_key(e):
    return frozenset(set(e["region"]) - {TG[1]})
def pt(e):
    kk = reg_key(e); return sum(w[x] for x in kk), e["p"]
print(f"target {TG[0]} x {TG[1]}: truth {truth:.2f} (n {n}); full cover {fe['p']:.2f}; glueable regions {len(keys)};"
      f" p_hat range [{expit(y.min()):.2f}, {expit(y.max()):.2f}]; first-order R2 "
      f"{1 - ((y - X @ c) ** 2).sum() / ((y - y.mean()) ** 2).sum():.3f}; corr(w, residual) {np.corrcoef([w[x] for x in others], [res[x] for x in others])[0, 1]:.3f}")
print("raise path:", [(e["dropped"], round(e["p"], 3)) for e in up]); print("lower path:", [(e["dropped"], round(e["p"], 3)) for e in dn])

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ink, muted, blue, orange, grey = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#b8b6ae"
fig = plt.figure(figsize=(7.4, 8.4)); gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.35, 1.0], hspace=0.55)
a0, a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
for a in (a0, a1, a2):
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.tick_params(colors=muted, labelsize=7.5)
order = sorted(others, key=lambda x: w[x])
short = lambda x: x.replace("international-politics-and-world-news", "intl-politics").replace("trump-presidency", "trump-pres.")
a0.barh(range(len(order)), [w[x] for x in order], color=[blue if w[x] > 0 else orange for x in order], height=0.62)
rs = np.array([res[x] for x in order]); scale = np.abs([w[x] for x in order]).max() / max(np.abs(rs).max(), 1e-12)
a0.scatter(rs * scale, range(len(order)), color=ink, s=10, zorder=3, label="loop residual of the outlet on the topic (rescaled)")
a0.axvline(0, color=ink, lw=0.7); a0.set_yticks(range(len(order))); a0.set_yticklabels([short(x) for x in order], fontsize=6.8)
a0.set_xlabel("influence on logit p̂ when the topic is glued in", fontsize=7.5, color=muted)
a0.set_title(f"(a) Loop content: each topic's pull on {TG[0]} × {TG[1]}  (blue: gluing it in raises p̂)", fontsize=8.5, loc="left", color=ink)
a0.legend(fontsize=6.8, frameon=False, loc="upper left")
a1.scatter(score, expit(y), s=3, color=grey, alpha=0.35, lw=0, label=f"all {len(keys)} glueable regions")
for path, col, lab in ((up, blue, "raise path (drop lowering topics)"), (dn, orange, "lower path (drop raising topics)")):
    P = [pt(e) for e in path]
    a1.plot([p[0] for p in P], [p[1] for p in P], "-o", color=col, lw=1.8, ms=3.5, label=lab)
fx, fy = pt(fe); a1.scatter([fx], [fy], s=60, color=ink, zorder=5, marker="D", label="full cover (all topics)")
a1.axhline(truth, color=ink, lw=1, ls="--"); a1.text(a1.get_xlim()[1], truth, "held-out truth ", fontsize=7, color=ink, va="bottom", ha="right")
a1.axhline(0.5, color=muted, lw=0.8, ls=":"); a1.text(a1.get_xlim()[1], 0.5, "outcome threshold 1/2 ", fontsize=7, color=muted, va="top", ha="right")
a1.set_xlabel("first-order score  Σ w_x over the glued topics", fontsize=7.5, color=muted); a1.set_ylabel("predicted p̂", fontsize=7.5, color=muted)
a1.set_title("(b) The surface: p̂ over the lattice is almost a function of the loop-content score", fontsize=8.5, loc="left", color=ink)
a1.legend(fontsize=6.8, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.08))
for path, col, lab in ((up, blue, "raise"), (dn, orange, "lower")):
    a2.plot(range(len(path)), [e["p"] for e in path], "-o", color=col, lw=1.8, ms=3.5, label=lab + " path")
    for i, e in enumerate(path[1:], 1):
        a2.text(i, e["p"] + (0.012 if col == blue else -0.03), short(e["dropped"]), fontsize=6, color=col, rotation=35, ha="left")
a2.axhline(truth, color=ink, lw=1, ls="--"); a2.axhline(0.5, color=muted, lw=0.8, ls=":")
a2.set_xlabel("topics dropped from the full cover", fontsize=7.5, color=muted); a2.set_ylabel("predicted p̂", fontsize=7.5, color=muted)
a2.set_title("(c) Paths: which topics to un-glue to lift or lower p̂ (every step stays glueable)", fontsize=8.5, loc="left", color=ink)
a2.legend(fontsize=6.8, frameon=False, loc="center right")
fig.savefig(os.path.join(here, "surface_chart.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(here, "surface_chart.png"), dpi=150, bbox_inches="tight")
json.dump(dict(target=list(TG), truth=truth, full=fe["p"], w=w, residual=res, up=[(e["dropped"], e["p"]) for e in up],
               down=[(e["dropped"], e["p"]) for e in dn]), open(os.path.join(here, "surface_chart_results.json"), "w"), indent=1)
