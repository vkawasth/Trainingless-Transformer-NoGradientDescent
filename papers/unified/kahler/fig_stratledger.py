"""The stratified ledger (sec:ledger; engine/stratledger.json from check_stratledger.py). A: linear programmes solved as the library
grows over the tables, for the four band ends in 3 x 3 and 4 x 4 (linear objectives plateau at C(r+c-2, r-1) = 6 and 20). B: the 3 x 3
population in the plane of the block's row and column mass, coloured by the optimal-basis region of the upper end; the support wall
f0+f1+g0+g1 = 1 (below it the band is undefined) and the top-20 drift tables. C: the ledger bound against the exact drift: valid
for every table, but a weak ranking."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__)); R = json.load(open(os.path.join(HERE, "..", "engine", "stratledger.json")))
INK, MUTED, GRID = "#222222", "#666666", "#d9d9d9"; RED, BLUE, AMBER, TEAL = "#b3261e", "#1f5f9e", "#c27c0e", "#2b8a7e"
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.6), gridspec_kw=dict(wspace=0.32))
ax = axs[0]
for tag, col in (("3x3", BLUE), ("4x4", RED)):
    for key, ls in (("hi0", "-"), ("lo0", "--"), ("hi1", ":"), ("lo1", "-.")):
        g = R[tag]["growth"][key]; ax.plot(np.arange(1, len(g) + 1), g, color=col, ls=ls, lw=1.1, label=f"{tag} {key}" if key in ("hi0", "lo0") else None)
ax.axhline(6, color=BLUE, lw=0.6, alpha=0.5); ax.axhline(20, color=RED, lw=0.6, alpha=0.5)
ax.text(9000, 2.5, "linear objectives: 6", color=BLUE, fontsize=6.3, ha="right"); ax.text(9000, 21.5, "linear objectives: 20", color=RED, fontsize=6.3, ha="right")
ax.set_xscale("log"); ax.set_xlabel("tables processed"); ax.set_ylabel("linear programmes solved"); ax.legend(fontsize=6.0, frameon=False, loc="upper left")
ax.set_title("A  A few solves certify every table", loc="left", fontsize=8.2, fontweight="bold", color=INK)
ax = axs[1]; tb = R["3x3"]["tables"]; strata = sorted({t["stratum"] for t in tb}); cmap = plt.get_cmap("tab10")
x = np.array([float(Fr(t["f"][0]) + Fr(t["f"][1])) for t in tb]); y = np.array([float(Fr(t["g"][0]) + Fr(t["g"][1])) for t in tb]); s = np.array([t["stratum"] for t in tb])
for k, st in enumerate(strata):
    m = s == st; ax.scatter(x[m], y[m], s=1.2, color=cmap(k % 10), lw=0, alpha=0.6)
und = R["3x3"]["undefined"]; ax.plot([0, 1], [1, 0], color=RED, lw=1.0, ls="--"); ax.text(0.04, 0.12, "support wall: below it the\nblock can vanish and the\nband is undefined", color=RED, fontsize=6.3)
D = np.array([t["drift"] for t in tb]); top = np.argsort(-D)[:20]; ax.scatter(x[top], y[top], s=14, facecolors="none", edgecolors=INK, lw=0.8, label="top-20 drift")
ax.set_xlabel("f0 + f1 (row mass of the block)"); ax.set_ylabel("g0 + g1 (column mass of the block)"); ax.legend(fontsize=6.2, frameon=False, loc="lower right")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("B  3x3 ledger: basis regions and the support wall", loc="left", fontsize=8.2, fontweight="bold", color=INK)
ax = axs[2]; U = np.array([t["bound"] for t in tb]); ax.scatter(U, D, s=1.2, color=BLUE, alpha=0.4, lw=0); ax.plot([0, U.max()], [0, U.max()], color=MUTED, lw=0.8, ls=":")
ax.scatter(U[top], D[top], s=14, facecolors="none", edgecolors=INK, lw=0.8)
ax.text(0.01, 0.16, f"Spearman {R['3x3']['spearman']:.2f}; certified top-20\nafter {R['3x3']['topk']['evaluations']} of {len(tb)} evaluations", fontsize=6.3)
ax.set_xlabel("ledger bound U (from osc of log L)"); ax.set_ylabel("exact drift D")
ax.set_title("C  The bound is valid but ranks weakly", loc="left", fontsize=8.2, fontweight="bold", color=INK)
fig.savefig(os.path.join(HERE, "fig_stratledger.png"), dpi=220, bbox_inches="tight"); print("ok")
