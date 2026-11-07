"""Figure for Study 18 (reads study18_final.json; plotting only, no analysis): fig_study18.pdf"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import kit_study18 as K
BLUE, ORANGE, AQUA, VIOLET, ROSE, INK, INK2, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#c2417a", "#22211f", "#52514e", "#9a9a9a"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
r = json.load(open(os.path.join(HERE, "study18_final.json"))); conds, W = K.conditions(); R = r["runs"]
grp = lambda f: [R[str(c["idx"])] for c in conds if f(c)]
FAM = ("F0", "F1", "F2"); VAR = (("plain", ORANGE, "o", "plain"), ("consistent", BLUE, "s", "consistent ($\\lambda_c$=1)"))
fig, ax = plt.subplots(2, 2, figsize=(12, 8.2))

a = ax[0, 0]
for k, (v, col, mk, lab) in enumerate(VAR):
    off = -0.12 if k == 0 else 0.12; means = []
    for i, f in enumerate(FAM):
        y = [x["tv_true_Q"] for x in grp(lambda c, f=f, v=v: c["kind"] == "S" and c["fam"] == f and c["variant"] == v)]
        a.plot([i + off] * len(y), y, mk, color=col, ms=4, alpha=0.55, mfc="none"); means.append(np.mean(y))
    a.plot(np.arange(3) + off, means, mk + "-", color=col, lw=2, ms=7, label=lab)
a.set_xticks(range(3)); a.set_xticklabels(["F0\nfibre dim 0\n(CH facet)", "F1\nfibre dim 1", "F2\nfibre dim 2"])
a.set_ylabel("TV(true joint, model joint)"); a.legend(frameon=False, fontsize=8, title="training (9 models each)", title_fontsize=8)
a.set_title("(a) Same sparsity (8 nonzero cells), different fibre dimension: with consistency\ntraining error rises with fibre dimension (H7: CIs include 0)", fontsize=9)

b = ax[0, 1]; FMK = {"F0": "o", "F1": "^", "F2": "s"}
for v, col, _, lab in VAR:
    for f in FAM:
        g = grp(lambda c, f=f, v=v: c["kind"] == "S" and c["fam"] == f and c["variant"] == v)
        b.plot([q["fit"] for q in g], [q["combination_gap"] for q in g], FMK[f], color=col, ms=6, alpha=0.75, mfc=col if f == "F0" else "none", label=f"{lab}, {f}")
b.set_xscale("symlog", linthresh=1e-3); b.set_xlim(-1e-4, 0.6)
b.set_xlabel("poor fitting: KL(training pair tables, direct predictions), summed over 4 contexts")
b.set_ylabel("failed combination: TV(joint's pair tables,\ndirect predictions), summed over 4 contexts")
b.legend(frameon=False, fontsize=7.2, ncol=2, loc="upper right")
b.set_title("(b) Consistency training removes failed combination (order drift\n0.31 to 0.01 on F0) but fits the training marginals less well (E7, E8)", fontsize=9)

c = ax[1, 0]; df, fl = r["H10"]["defects"], r["H10"]["floors"]; x = np.arange(1, 7)
c.bar(x - 0.18, fl, 0.34, color=GREY, label="noise floor: product model, seed to seed")
c.bar(x + 0.18, df, 0.34, color=VIOLET, label="tensor-square defect, consistent models")
c.set_xticks(x); c.set_xticklabels([f"pair {i}" for i in x]); c.set_ylabel("TV"); c.set_ylim(0, 0.075); c.legend(frameon=False, fontsize=7.8, loc="upper left")
lo, hi = r["H10"]["ci_excess"]
c.set_title(f"(c) Independent systems against the noise floor (H10): excess defect\n95% CI [{lo:.3f}, {hi:.3f}] lies above 0, so the alternative is reported", fontsize=9)

d = ax[1, 1]
g2 = grp(lambda c: c["kind"] == "G2"); s = grp(lambda c: c["kind"] == "S")
d.plot([x["cf_data"] for x in g2], [x["cf_model"] for x in g2], "o", color=ORANGE, ms=7, label="G2 cohort worlds (9 models)")
d.plot([x["cf_data"] for x in s], [x["cf_model"] for x in s], "s", color=BLUE, ms=4, alpha=0.5, mfc="none", label="S faces (81 models)")
d.plot([0, 0.6], [0, 0.6], "--", color=INK2, lw=0.8); d.text(0.42, 0.45, "model = data", fontsize=7.6, color=INK2, rotation=38)
d.set_xlim(-0.02, 0.6); d.set_ylim(-0.02, 0.6); d.set_xlabel("contextual fraction of the training pair tables"); d.set_ylabel("contextual fraction of the model's direct pair tables")
d.legend(frameon=False, fontsize=7.8, loc="upper left")
d.set_title("(d) Models report less contextuality than their data (E6):\nG2 mean 0.43 in the data, 0.21 in the model", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_study18.pdf")); fig.savefig(os.path.join(HERE, "fig_study18.png"), dpi=150); print("ok")
