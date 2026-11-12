"""Figure for 'Beliefs under contamination' (module pce/bridgetoy.py).
  fig_bridge.pdf  (a) the toy bridge: five variables on a cycle, the five sources on its edges, the inspection chord (X2, X4);
                  (b) for each matched history, the decision maker's 1-robust interval for B = P(X3 = X4) after each step, with the
                  threshold 1/2 and the naive value (dots)"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from pce import bridgetoy as BT
BLUE, ORANGE, VIOLET, ROSE, GREEN, INK, INK2, GREY = "#2a78d6", "#eb6834", "#6250d6", "#c2417a", "#2f8f5b", "#22211f", "#52514e", "#9a9a9a"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
P, gA, gB, h, Q, S3_in, S3_out, c2, H = BT.histories()
fig = plt.figure(figsize=(11, 4.8)); gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 1.6], wspace=0.12)
a = fig.add_subplot(gs[0]); a.set_aspect("equal"); a.set_axis_off()
ang = np.pi / 2 + 2 * np.pi * np.arange(5) / 5; xy = np.c_[np.cos(ang), np.sin(ang)]
for i in range(5):
    j = (i + 1) % 5; col = ROSE if i == 2 else BLUE
    a.plot(*zip(xy[i], xy[j]), color=col, lw=3 if i == 2 else 2)
    m = (xy[i] + xy[j]) / 2 * 1.22; a.text(*m, f"S{i+1}", ha="center", va="center", fontsize=9.5, color=col)
a.plot(*zip(xy[1], xy[3]), color=GREEN, lw=1.8, ls="--"); a.text(*((xy[1] + xy[3]) / 2 + [0.08, 0.1]), "inspection", color=GREEN, fontsize=8.5)
for i in range(5):
    a.scatter(*xy[i], s=260, color="white", edgecolor=INK, zorder=3); a.text(*xy[i], f"X{i+1}", ha="center", va="center", fontsize=8.5, zorder=4)
a.text(0, -1.45, "decision A: P(X1 = X2), seen by S1\ndecision B: P(X3 = X4), seen only by S3", ha="center", fontsize=8.5, color=INK2)
a.set_xlim(-1.5, 1.5); a.set_ylim(-1.75, 1.4); a.set_title("(a) The toy bridge", fontsize=9.5, loc="left")
b = fig.add_subplot(gs[1]); names = ["H1", "H2", "H3", "H4", "H5"]; cols = [INK, GREY, ROSE, ORANGE, VIOLET]
desc = {"H1": "honest", "H2": "S3 withheld,\ninspection refused", "H3": "S3 poisoned\nbeyond budget", "H4": "S3 within budget,\nS2 recruited", "H5": "S3 poisoned,\nthen corrected"}
for k, (name, col) in enumerate(zip(names, cols)):
    _, rows = BT.run(H[name], gB); x0 = k * 8.5
    for t, (lab, v, d) in enumerate(rows):
        lo, hi = d["rob"]; b.plot([x0 + t, x0 + t], [lo, hi], color=col, lw=4, solid_capstyle="butt", alpha=0.85)
        b.scatter(x0 + t, d["mid"], s=14, color="white", edgecolor=col, zorder=3)
        b.text(x0 + t, -0.03, lab, ha="center", fontsize=6.5, color=INK2, rotation=90, va="top")
    b.text(x0 + 3, 1.04, f"{name}\n{desc[name]}", ha="center", va="bottom", fontsize=7.6, color=col)
b.axhline(0.5, color=INK2, lw=0.8, ls=":"); b.axhline(gB @ P, color=GREEN, lw=0.8, ls="--")
b.text(42.5, gB @ P - 0.06, "true B", color=GREEN, fontsize=8, ha="right"); b.text(42.5, 0.45, "threshold", color=INK2, fontsize=8, ha="right")
b.set_xlim(-1, 43); b.set_ylim(-0.16, 1.27); b.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); b.set_xticks([]); b.set_ylabel("B = P(X3 = X4)")
b.set_title("(b) The decision maker's 1-robust interval for B after each step (bars) and the naive value (dots)", fontsize=9.5, loc="left")
for s in ("bottom",): b.spines[s].set_visible(False)
fig.savefig(os.path.join(HERE, "fig_bridge.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_bridge.png"), dpi=150, bbox_inches="tight")
