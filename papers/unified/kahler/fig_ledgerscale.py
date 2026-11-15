"""Study 28 (sec:ledgerscale; engine/study28_scale/ledgerscale.json from check_ledgerscale.py). A: linear programmes (four band ends
summed) against the number of defined tables, for the largest shape of each family and the two square shapes; a slope-1 guide shows
what solving every table would cost. B: solves at T = 10000 against the tree bound r^(c-1) c^(r-1) for all 15 shapes. C: build time of
the ledger against the time to solve every band end directly."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); D = json.load(open(os.path.join(HERE, "..", "engine", "study28_scale", "ledgerscale.json")))
INK, MUTED = "#222222", "#666666"; COL = {"2": "#1f5f9e", "3": "#2b8a7e", "4": "#b3261e"}
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.5), gridspec_kw=dict(wspace=0.33))
TS = ("1000", "3000", "10000")
tot = lambda R, T: sum(e["solves"] for e in R["per_T"][T]["ends"].values())
ax = axs[0]
for tag, ls in (("2x8", "-"), ("3x3", ":"), ("3x7", "-"), ("4x4", ":"), ("4x6", "-")):
    R = D[tag]; x = [R["per_T"][T]["defined"] for T in TS]; y = [tot(R, T) for T in TS]
    ax.plot(x, y, color=COL[tag[0]], ls=ls, marker="o", ms=3.5, lw=1.2); ax.text(x[-1] * 1.12, y[-1], tag, color=INK, fontsize=6.5, va="center")
xx = np.array([200, 12000]); ax.plot(xx, 4 * xx, color=MUTED, lw=0.7, ls="--"); ax.text(260, 4 * 260 * 1.6, "solving every table", color=MUTED, fontsize=6.3, rotation=28)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(180, 30000); ax.set_xlabel("defined tables"); ax.set_ylabel("linear programmes (four ends)")
ax.set_title("A  Solves grow far slower than tables", loc="left", fontsize=8.2, fontweight="bold", color=INK)
ax = axs[1]
for tag, R in D.items():
    ax.scatter(R["tree_bound"], tot(R, "10000") / 4, s=16, color=COL[tag[0]], lw=0)
    if tag in ("2x2", "2x8", "3x3", "3x7", "4x4", "4x6"): ax.annotate(tag, (R["tree_bound"], tot(R, "10000") / 4), xytext=(4, -3), textcoords="offset points", fontsize=6.3, color=INK)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("tree bound r^(c-1) c^(r-1)"); ax.set_ylabel("solves per band end, T = 10000")
for k, lab in COL.items(): ax.scatter([], [], color=lab, s=16, label=f"{k} x n")
ax.legend(frameon=False, fontsize=6.3, loc="upper left")
ax.set_title("B  Regions met are a tiny share of the bound", loc="left", fontsize=8.2, fontweight="bold", color=INK)
ax = axs[2]; tags = list(D); xs = np.arange(len(tags))
ax.bar(xs - 0.2, [D[t]["direct_estimate"] for t in tags], width=0.4, color="#c9c9c9", label="solve every band end")
ax.bar(xs + 0.2, [D[t]["per_T"]["10000"]["build_time"] for t in tags], width=0.4, color=COL["3"], label="build the ledger")
ax.set_yscale("log"); ax.set_xticks(xs); ax.set_xticklabels(tags, rotation=60, fontsize=6.2); ax.set_ylabel("seconds (T = 10000)")
ax.legend(frameon=False, fontsize=6.3, loc="upper right")
ax.set_title("C  Building the ledger is 6 to 2900 times faster", loc="left", fontsize=8.2, fontweight="bold", color=INK)
fig.savefig(os.path.join(HERE, "fig_ledgerscale.png"), dpi=220, bbox_inches="tight"); print("ok")
