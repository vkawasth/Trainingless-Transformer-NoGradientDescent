"""Figure for prop:chainglue of the companion note (module pce/chainglue.py).
  fig_chainglue.pdf  (a) the figure-eight pseudo-diagram as a chain of three pieces, crossings {1, 2}, {3}, {4}, with the cut states and the
                     quotient update, and the closing condition that a loop would add; (b) the worked example: a two-point law with
                     P(knotted) = 1, glued through the quotient (1) and through 'region 1 agrees' alone (1/2)"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
BLUE, ORANGE, VIOLET, ROSE, INK, INK2, GREY = "#2a78d6", "#eb6834", "#6250d6", "#c2417a", "#22211f", "#52514e", "#9a9a9a"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
fig = plt.figure(figsize=(10.5, 4.6)); gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 0.75], wspace=0.15)
a = fig.add_subplot(gs[0]); a.set_axis_off(); a.set_xlim(0, 13.4); a.set_ylim(-1.0, 6.6)
pieces = [(0.3, "piece D1\ne1, e2", BLUE), (5.0, "piece D2\ne3", ORANGE), (9.7, "piece D3\ne4", VIOLET)]
for x0, lab, col in pieces:
    a.add_patch(FancyBboxPatch((x0, 2.6), 2.6, 1.5, boxstyle="round,pad=0.08", fc="white", ec=col, lw=1.8)); a.text(x0 + 1.3, 3.35, lab, ha="center", va="center", fontsize=9, color=INK)
for x1, x2, lab in ((2.95, 4.95, "c1: does region 1\nagree? (2 classes)"), (7.65, 9.65, "c2: 3 classes\ndisagree;\nagree, e3 = +;\nagree, e3 = −")):
    a.add_patch(FancyArrowPatch((x1, 3.55), (x2, 3.55), arrowstyle="-|>", mutation_scale=14, color=INK2, lw=1.4)); a.text((x1 + x2) / 2, 2.45, lab, ha="center", va="top", fontsize=8.5, color=INK2)
a.text(11.6, 2.45, "knotted iff c2 is an\n'agree' class\nand e4 = e3", ha="center", va="top", fontsize=8.5, color=INK2)
a.text(6.3, 0.25, "quotient update c2 = h(c1, e3):\n(disagree, ·) → disagree;  (agree, ±) → agree with e3 = ±", ha="center", fontsize=8.5, color=INK)
a.add_patch(FancyArrowPatch((11.0, 4.25), (1.6, 4.25), connectionstyle="arc3,rad=0.28", arrowstyle="-|>", mutation_scale=12, color=ROSE, lw=1.2, ls="--"))
a.text(6.3, 5.85, "a loop would add a closing condition: the last state\nmust match the first — this can fail (Prop. 78)", ha="center", fontsize=8.5, color=ROSE)
a.text(6.3, -0.75, "gluing keeps every table (state, next piece);\nthe order of the cuts does not matter", ha="center", fontsize=8.5, color=INK2)
a.set_title("(a) The figure-eight pseudo-diagram as a chain of three pieces", fontsize=9.5, loc="left")
b = fig.add_subplot(gs[1])
vals = [1.0, 1.0, 0.5]; labs = ["true\nlaw", "glued via\nquotient c2", "glued via\n'region 1\nagrees'"]; cols = [INK, VIOLET, ROSE]
b.bar(range(3), vals, color=cols, width=0.6)
for i, v in enumerate(vals): b.text(i, v + 0.03, f"{v:g}", ha="center", fontsize=9, color=INK)
b.set_xticks(range(3)); b.set_xticklabels(labs, fontsize=8.5); b.set_ylim(0, 1.2); b.set_ylabel("P(knotted)")
b.set_title("(b) Worked example: a two-point law", fontsize=9.5, loc="left")
fig.savefig(os.path.join(HERE, "fig_chainglue.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_chainglue.png"), dpi=150, bbox_inches="tight")
