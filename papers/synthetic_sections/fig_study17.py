"""Figure for Study 17 (reads study17_final.json; plotting only, no analysis): fig_study17.pdf"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import kit_learned_sections as K
BLUE, ORANGE, AQUA, VIOLET, ROSE, INK, INK2, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#c2417a", "#22211f", "#52514e", "#9a9a9a"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
r = json.load(open(os.path.join(HERE, "study17_final.json"))); conds, W = K.conditions(); R = r["runs"]
grp = lambda f: [x for x in R if f(conds[x["idx"]])]; mean = lambda g, k: float(np.mean([x[k] for x in g]))
fig, ax = plt.subplots(2, 2, figsize=(12, 8.2))
a = ax[0, 0]; eps = ["1/5", "1/20", "1/100", "0"]; xs = np.arange(4)
for path, col, lab in (("CH", BLUE, "toward a CH facet"), ("zero", ORANGE, "toward a structural zero")):
    g = [grp(lambda c, e=e: c["kind"] == "C" and c.get("path") == path and c.get("eps") == e) for e in eps]
    a.plot(xs, [mean(x, "infibre") for x in g], "o-", color=col, lw=2, label=f"{lab}: in-fibre (coherent, unsupported)")
    a.plot(xs, [mean(x, "offfibre") for x in g], "s--", color=col, lw=1.4, label=f"{lab}: off-fibre (inconsistent with data)")
a.set_xticks(xs); a.set_xticklabels([f"$\\varepsilon$={e}" for e in eps]); a.set_ylabel("KL (nats), mean of 2 seeds"); a.legend(frameon=False, fontsize=7.2)
a.set_title("(a) At the CH facet the in-fibre part vanishes, as the theorem says,\nbut the model's inconsistency with its own data grows", fontsize=9)
b = ax[0, 1]; io = [x["iota"] for x in grp(lambda c: c["kind"] == "G1")]
b.hist(io, bins=np.linspace(0, 1, 21), color=BLUE, alpha=0.85); b.axvline(0.5, color=ROSE, ls="--", lw=1); top = b.get_ylim()[1]; b.set_ylim(0, top * 1.25)
b.text(0.52, top * 1.12, "leximin-like $\\rightarrow$", fontsize=8, color=ROSE); b.text(0.48, top * 1.12, "$\\leftarrow$ maximum-entropy-like", fontsize=8, color=BLUE, ha="right")
b.set_xlabel("section index $\\iota$ (0 = maximum entropy, 1 = leximin)"); b.set_ylabel("G1 models (of 48)")
b.set_title(f"(b) Which section learning picks: {r['H1']['share_below_half']:.0%} of models\nare maximum-entropy-like (H1)", fontsize=9)
c = ax[1, 0]
labels = ["G1\nN=300", "G1\nN=3000", "G2\nN=300", "G2\nN=3000", "CHSH\n$\\lambda$=3/5", "CHSH\n$\\lambda$=4/5", "PR box\n$\\lambda$=1"]
gs = [grp(lambda c: c["kind"] == "G1" and c["N"] == 300), grp(lambda c: c["kind"] == "G1" and c["N"] == 3000), grp(lambda c: c["kind"] == "G2" and c["N"] == 300),
      grp(lambda c: c["kind"] == "G2" and c["N"] == 3000)] + [grp(lambda c, l=l: c["kind"] == "C" and c.get("path") == "PR" and c.get("eps") == l) for l in ("3/5", "4/5", "1")]
cols = [BLUE, BLUE, ORANGE, ORANGE, ROSE, ROSE, ROSE]
for k, (g, col) in enumerate(zip(gs, cols)):
    v = [x["drift"] for x in g]; c.bar(k, np.mean(v), color=col, alpha=0.85, width=0.6); c.plot([k] * len(v), v, "o", color=INK, ms=2.5, alpha=0.6)
c.set_xticks(range(7)); c.set_xticklabels(labels, fontsize=7.6); c.set_ylabel("order drift (mean TV between generation orders)")
c.set_title("(c) Shadows from separate cohorts, and contextual data, make\nthe model's answers depend on the order it generates them (H4, H6)", fontsize=9)
d = ax[1, 1]; df = r["H5"]["defects"]
d.plot(range(1, len(df) + 1), sorted(df), "o", color=VIOLET, ms=6, label="learned composition (12 cases)")
d.axhline(0.0752, color=ROSE, ls="--", lw=1, label="leximin selection on the note's chains (0.0752)"); d.axhline(0.0188, color=AQUA, ls=":", lw=1.4, label="pre-registered threshold (0.0188)")
d.axhline(0, color=INK2, lw=0.6); d.text(0.6, 0.002, "maximum entropy: 0", fontsize=7.4, color=INK2)
d.set_xlabel("case (sorted)"); d.set_ylabel("tensor-square defect (TV)"); d.legend(frameon=False, fontsize=7.4, loc="upper left", bbox_to_anchor=(0.0, 0.86)); d.set_ylim(-0.005, 0.085)
d.set_title("(d) Learned composition does not respect independent systems (H5):\nmaximum-entropy-like joints, but not system independent", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_study17.pdf")); fig.savefig(os.path.join(HERE, "fig_study17.png"), dpi=150); print("ok")
