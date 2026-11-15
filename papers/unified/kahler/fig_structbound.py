"""Study 30 (sec:structbound; engine/study30_structure/structbound.json from check_structbound.py). A: share of tables evaluated by the
band-only search of Study 29 and by the structure-aware screen, with the linear programmes each needed. B: P332, share of tables a bound
excludes from the top-20 (20th largest drift 0.1456): band-only U0, block relaxation U1, their combination with library vertices, and
the final bound after dual certificates (post-hoc diagnosis, recorded in the script docstring). C: Spearman correlation of each bound
with the exact drift (pre-registered floor 0.7 for min(U0, U1) in P332)."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); D = json.load(open(os.path.join(HERE, "..", "engine", "study30_structure", "structbound.json")))
INK, MUTED = "#222222", "#666666"; BLUE, TEAL, RED, GREY = "#1f5f9e", "#2b8a7e", "#b3261e", "#bdbdbd"
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fams = ["P332", "P332tol", "P442"]; lab = {"P332": "3x3x2\nexact", "P332tol": "3x3x2\n+-1/100", "P442": "4x4x2\nexact"}; xs = np.arange(3)
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.4), gridspec_kw=dict(wspace=0.32))
ax = axs[0]
b1 = [D[f]["band_only"]["share"] for f in fams]; b2 = [D[f]["screen"]["share"] for f in fams]
ax.bar(xs - 0.2, b1, 0.4, color=GREY, label="band-only search (Study 29)"); ax.bar(xs + 0.2, b2, 0.4, color=TEAL, label="structure-aware screen")
for x, f in zip(xs, fams):
    ax.text(x - 0.2, D[f]["band_only"]["share"] * 1.15, f"{D[f]['band_only']['share']:.0%}\n{D[f]['band_only']['solves']} LPs", ha="center", fontsize=6, color=INK)
    ax.text(x + 0.2, D[f]["screen"]["share"] * 1.15, f"{D[f]['screen']['share']:.1%}\n{D[f]['screen']['solves']} LPs", ha="center", fontsize=6, color=INK)
ax.set_yscale("log"); ax.set_ylim(1e-3, 5); ax.axhline(0.5, color=RED, lw=0.8, ls="--"); ax.text(2.5, 0.3, "ceiling 50%", color=RED, fontsize=6.2, ha="right")
ax.set_xticks(xs); ax.set_xticklabels([lab[f] for f in fams], fontsize=6.4); ax.set_ylabel("share of 10000 tables evaluated"); ax.legend(frameon=False, fontsize=6.2, loc="upper left")
ax.set_title("A  Certified top-20: 0.3-0.6% evaluated", loc="left", fontsize=8.2, fontweight="bold", color=INK)
ax = axs[1]; names = ["band only\nU0", "block\nrelaxation U1", "U0, U1 and\nvertices", "after dual\ncertificates"]; vals = [0.186, 0.709, 0.864, 0.998]
ax.bar(np.arange(4), vals, 0.6, color=[GREY, BLUE, BLUE, TEAL])
for x, v in enumerate(vals): ax.text(x, v + 0.02, f"{v:.1%}", ha="center", fontsize=6.4, color=INK)
ax.set_ylim(0, 1.1); ax.set_xticks(np.arange(4)); ax.set_xticklabels(names, fontsize=6.3); ax.set_ylabel("share of tables excluded from the top-20")
ax.set_title("B  3x3x2: what each layer rules out", loc="left", fontsize=8.2, fontweight="bold", color=INK)
ax = axs[2]
for off, key, col, name in ((-0.24, "U0", GREY, "band only U0"), (-0.08, "U1", BLUE, "block relaxation U1"), (0.08, "minU0U1", RED, "min(U0, U1)"), (0.24, "combined", TEAL, "with vertices")):
    ax.bar(xs + off, [D[f]["spearman"][key] for f in fams], 0.15, color=col, label=name)
ax.axhline(0.7, color=RED, lw=0.8, ls="--"); ax.text(-0.4, 0.715, "pre-registered floor (3x3x2 exact)", color=RED, fontsize=6.2, va="bottom")
ax.set_ylim(0, 1.15); ax.set_xticks(xs); ax.set_xticklabels([lab[f] for f in fams], fontsize=6.4); ax.set_ylabel("Spearman correlation with exact drift")
ax.legend(frameon=False, fontsize=6, loc="upper left", ncol=2)
ax.set_title("C  Ranking stays only moderate", loc="left", fontsize=8.2, fontweight="bold", color=INK)
fig.savefig(os.path.join(HERE, "fig_structbound.png"), dpi=220, bbox_inches="tight"); print("ok")
