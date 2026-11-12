"""Figure for 'Transport along revisions, holonomy and laundering'.
  (a) the revision loop: data D, then D plus a published joint table, then the table retracted (data D again); what a recomputer and an
      updater hold at the end
  (b) the 'pathway alone' field plane: the image of the fibre (what a recomputer must allow), the curve reachable by retracted {R,S}
      tables, the honest and the laundered updater, and the true law (world A of check_fieldholonomy.py)"""
import os, sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
import warnings; warnings.filterwarnings("ignore")
from pce import fieldholonomy as FH
COL = ["#1f5f9e", "#c27c0e", "#8e3b8e", "#2e7d32"]; INK, MUTED, GRID = "#222222", "#666666", "#d9d9d9"
U = np.ones(8) / 8
PA = np.array([0.08, 0.042, 0.087, 0.06, 0.225, 0.253, 0.124, 0.128]); PA = PA / PA.sum()
data = [((0, 2), FH.marg(PA, (0, 2))), ((1, 2), FH.marg(PA, (1, 2)))]
H = FH.refit(U, data); V = np.array(FH.fibre_vertices(data))
diff = lambda Q: FH.joint_field(Q)[0] - FH.joint_field(Q)[1]
vmin = min(V, key=lambda Q: diff(Q) if np.isfinite(diff(Q)) else 9); T = 0.85 * vmin + 0.15 * H
L = FH.refit(FH.jeffrey(U, (0, 1, 2), dict(zip(FH.OUT, T))), data)
fig = plt.figure(figsize=(13, 5.2)); gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.15)

a = fig.add_subplot(gs[0]); a.axis("off"); a.set_xlim(0, 10.4); a.set_ylim(0.2, 5.8)
def box(x, y, w, h, txt, fc):
    a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=fc, ec=INK, lw=1)); a.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8)
box(0.1, 4.1, 2.9, 1.3, "version 1\ndata: tables\n{R,Y} and {S,Y}", "#f3f3f3")
box(3.75, 4.1, 2.9, 1.3, "version 2\nplus a published\njoint table (R,S,Y)", "#fbe7c4")
box(7.4, 4.1, 2.9, 1.3, "version 3\ntable retracted:\nsame data as version 1", "#f3f3f3")
for x0, x1, lab in ((3.05, 3.7, "publish"), (6.7, 7.35, "retract")):
    a.annotate("", xy=(x1, 4.75), xytext=(x0, 4.75), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3)); a.text((x0 + x1) / 2, 5.55, lab, ha="center", fontsize=8, color=MUTED)
box(0.1, 1.5, 4.95, 1.9, "recomputer (keeps no memory)\nholds the fibre of the current data;\nversion 1 = version 3, so no holonomy;\nthe order stays open (a pre-crossing)", "#e6eef7")
box(5.35, 1.5, 4.95, 1.9, "updater (carries a law)\nJeffrey update per version, then a refit;\nconsistent with every current table,\nbut moved: a confident, reversed order", "#f7ead6")
a.text(5.2, 0.55, "Edits on the data's own contexts are erased by the refit;\nonly a context the data never had leaves a trace.", ha="center", fontsize=8.5, color=INK)
a.set_title("(a) a closed loop of revisions", fontsize=10, loc="left")

b = fig.add_subplot(gs[1]); rng = np.random.default_rng(0); pts = []
with np.errstate(all="ignore"):
    for w in rng.dirichlet(np.ones(len(V)) * 0.4, 20000):
        q = FH.joint_field(w @ V)
        if np.all(np.isfinite(q)): pts.append(q)
pts = np.array(pts); b.scatter(pts[:, 0], pts[:, 1], s=1.5, color=GRID, rasterized=True, label="fibre: all laws a recomputer must allow")
cur = []
for g in np.linspace(-6, 6, 121):
    tab = {(0, 0): np.exp(g), (0, 1): 1.0, (1, 0): 1.0, (1, 1): np.exp(g)}; z = sum(tab.values()); tab = {k: v / z for k, v in tab.items()}
    cur.append(FH.joint_field(FH.refit(FH.jeffrey(U, (0, 1), tab), data, 600)))
cur = np.array(cur); b.plot(cur[:, 0], cur[:, 1], color=COL[2], lw=2, label="reachable by retracted {R,S} tables")
lim = [0, 1]; b.plot(lim, lim, "--", color=INK, lw=1, label="equal: the crossing")
for P_, c, lab, mk in ((PA, INK, "true law", "*"), (H, COL[0], "honest updater", "o"), (L, COL[1], "laundered updater", "s")):
    q = FH.joint_field(P_); b.plot(q[0], q[1], mk, color=c, ms=11 if mk == "*" else 9, label=lab, mec="white", mew=1)
b.set_xlim(0, 1); b.set_ylim(0, 1); b.set_aspect("equal"); b.tick_params(labelsize=8)
b.set_xlabel("P(outcome | R alone)", fontsize=9); b.set_ylabel("P(outcome | S alone)", fontsize=9)
b.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5)); b.spines[["top", "right"]].set_visible(False)
b.set_title("(b) where each decision maker ends", fontsize=10, loc="left")
fig.savefig(os.path.join(HERE, "fig_holonomy.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_holonomy.png"), dpi=150, bbox_inches="tight")
print(np.round(FH.joint_field(H), 3), np.round(FH.joint_field(L), 3), np.round(FH.joint_field(PA), 3))
