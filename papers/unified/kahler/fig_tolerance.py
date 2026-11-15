"""Study 29 (sec:tolerance; engine/study29_tolerance/tolerance.json from check_tolerance.py). A: linear programmes for all four band
ends and the defined test, per family, against the 10000 tables. B: share of defined tables whose after-evidence band the certified
top-20 search had to evaluate (pre-registered ceiling 50%). C: coverage of the 95% intervals over 500 replicate samples of 400 tables,
stratified against simple random sampling, for the mean drift and the share of tables with drift above 0.05."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); D = json.load(open(os.path.join(HERE, "..", "engine", "study29_tolerance", "tolerance.json")))
INK, MUTED = "#222222", "#666666"; BLUE, TEAL, RED, GREY = "#1f5f9e", "#2b8a7e", "#b3261e", "#c9c9c9"
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
lab = {"tol3-0": "3x3\ne=0", "tol3-1/200": "3x3\ne=1/200", "tol3-1/100": "3x3\ne=1/100", "tol3-1/50": "3x3\ne=1/50", "tol4-1/100": "4x4\ne=1/100", "pair332": "3x3x2\npairwise"}
fams = list(D); xs = np.arange(len(fams))
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.4), gridspec_kw=dict(wspace=0.3))
ax = axs[0]; sv = [sum(D[f]["solves"].values()) for f in fams]
ax.bar(xs, sv, color=TEAL, width=0.6); ax.axhline(2000, color=RED, lw=0.8, ls="--"); ax.text(len(fams) - 0.6, 2000 * 1.15, "pre-registered ceiling 0.2 T", color=RED, fontsize=6.3, ha="right")
for x, v in zip(xs, sv): ax.text(x, v * 1.12, str(v), ha="center", fontsize=6.3, color=INK)
ax.set_yscale("log"); ax.set_ylim(10, 20000); ax.set_xticks(xs); ax.set_xticklabels([lab[f] for f in fams], fontsize=6.2); ax.set_ylabel("linear programmes for 10000 tables")
ax.set_title("A  The library still compresses", loc="left", fontsize=8.2, fontweight="bold", color=INK)
ax = axs[1]; sh = [D[f]["topk"]["share"] for f in fams]
ax.bar(xs, sh, color=[RED if s > 0.5 else BLUE for s in sh], width=0.6); ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
for x, v in zip(xs, sh): ax.text(x, v + 0.02, f"{v:.0%}", ha="center", fontsize=6.3, color=INK)
ax.set_ylim(0, 1); ax.set_xticks(xs); ax.set_xticklabels([lab[f] for f in fams], fontsize=6.2); ax.set_ylabel("share of tables evaluated")
ax.text(5, 0.92, "bound saturates:\nbands narrow and interior", fontsize=6.2, color=RED, ha="center")
ax.set_title("B  Certified top-20: cost of the search", loc="left", fontsize=8.2, fontweight="bold", color=INK)
ax = axs[2]; tf = [f for f in fams if D[f]["sample"]]; x2 = np.arange(len(tf))
for off, key, col, mk, name in ((-0.15, "mean_str", BLUE, "o", "mean drift, stratified"), (-0.05, "mean_srs", BLUE, "x", "mean drift, simple random"),
                                 (0.05, "share_str", TEAL, "o", "share D > 0.05, stratified"), (0.15, "share_srs", TEAL, "x", "share D > 0.05, simple random")):
    ax.scatter(x2 + off, [D[f]["sample"]["coverage"][key] for f in tf], color=col, marker=mk, s=18, label=name)
ax.axhline(0.95, color=MUTED, lw=0.7, ls=":"); ax.axhline(0.93, color=RED, lw=0.8, ls="--"); ax.text(-0.3, 0.9315, "pre-registered floor 0.93", color=RED, fontsize=6.2)
ax.set_ylim(0.9, 0.97); ax.set_xticks(x2); ax.set_xticklabels([lab[f] for f in tf], fontsize=6.2); ax.set_ylabel("coverage of 95% intervals (500 samples)")
ax.legend(frameon=False, fontsize=5.8, loc="upper left", ncol=2)
ax.set_title("C  A sample of 400 gives honest intervals", loc="left", fontsize=8.2, fontweight="bold", color=INK)
fig.savefig(os.path.join(HERE, "fig_tolerance.png"), dpi=220, bbox_inches="tight"); print("ok")
