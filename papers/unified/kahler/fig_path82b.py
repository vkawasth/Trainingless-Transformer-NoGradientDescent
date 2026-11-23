"""Figure for Study 82 (fig_path82b.pdf): the wall-crossing comparison along the path, as one picture.
Top: the six objects along the margin parameter t (ladder points of PW1/PW2 marked), the Stokes event, the mutation of the sixth object,
and the two carried comparisons. Bottom: the quivers (irreducible morphisms, with multiplicities) of C' before and C'' after the wall."""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
INK, MUTED, BLUE, ORANGE, GREEN, GRID = "#1F2937", "#6B7280", "#1D4ED8", "#C2410C", "#047857", "#E5E7EB"
HERE = os.path.dirname(os.path.abspath(__file__))
fig = plt.figure(figsize=(10.5, 7.0)); ax = fig.add_axes([0.06, 0.40, 0.90, 0.56]); qa = fig.add_axes([0.04, 0.02, 0.44, 0.33]); qb = fig.add_axes([0.52, 0.02, 0.44, 0.33])
TW = -0.000825; L1 = [-0.1, -0.07, -0.04, -0.01, 0.005, 0.02, 0.025]; L2 = [0.01, 0.015, 0.02, 0.025]
X = lambda t: (t + 0.1) / 0.125                                   # map t in [-0.1, 0.025] to [0, 1]
names = ["$V_1 \\to O_{E_1}(-1)$", "$V_2 \\to O_{E_2}(-1)$", "$V_3 \\to O(-H)$", "$V_4 \\to O(E_3-H)$", "$V_5 \\to O$", "$V_6 \\to O(H)$"]
ys = [5, 4, 3, 2, 1, 0]
for k in range(5):
    ax.plot([X(-0.1), X(0.025)], [ys[k], ys[k]], color=BLUE, lw=2.2, solid_capstyle="round"); ax.text(X(-0.1) - 0.01, ys[k], names[k], ha="right", va="center", fontsize=9, color=INK)
ax.plot([X(-0.1), X(TW)], [0, 0], color=BLUE, lw=2.2); ax.text(X(-0.1) - 0.01, 0, names[5], ha="right", va="center", fontsize=9, color=INK)
ax.plot([X(TW), X(0.025)], [-0.9, -0.9], color=ORANGE, lw=2.6); ax.text(X(0.025) + 0.01, -0.9, "$V_{6'} \\to T = L_{O}O(H)[-1]$", ha="left", va="center", fontsize=9, color=ORANGE)
ax.annotate("", xy=(X(TW) + 0.004, -0.86), xytext=(X(TW) - 0.004, -0.04), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4))
ax.text(X(TW) - 0.012, -0.55, "mutation (not a continuation):\n$[V_{6'}] = [V_6] - 3[V_5]$", fontsize=8, color=ORANGE, va="center", ha="right")
ax.axvline(X(TW), color=INK, lw=1, ls="--"); ax.text(X(TW), 6.3, "Stokes event $t=-0.000825$\n($v_5$ crosses the ray of $v_6$)", ha="center", fontsize=8.5, color=INK)
for t in L1:
    for k in range(5): ax.plot(X(t), ys[k], "o", ms=4.5, color="white", mec=BLUE, mew=1.2, zorder=4)
for t in (-0.1, -0.07, -0.04, -0.01): ax.plot(X(t), 0, "o", ms=4.5, color="white", mec=BLUE, mew=1.2, zorder=4)
for t in L2: ax.plot(X(t), -0.9, "o", ms=4.5, color="white", mec=ORANGE, mew=1.2, zorder=4)
# carried comparisons
ax.add_patch(FancyArrowPatch((X(-0.1), -2.25), (X(0.025), -2.25), arrowstyle="-|>", mutation_scale=14, color=GREEN, lw=1.6))
ax.text(X(-0.04), -2.5, "PW1: the Study 76 comparison $\\Phi$, followed along the path on the five unchanged objects,\nequals the comparison found at the end, up to one sign per object (a natural isomorphism)", fontsize=8.3, color=GREEN, va="top", ha="center")
ax.add_patch(FancyArrowPatch((X(0.010), -3.5), (X(0.025), -3.5), arrowstyle="-|>", mutation_scale=14, color=GREEN, lw=1.6))
ax.text(X(0.0175), -3.75, "PW2-strict: the comparison frozen at $t=0.010$,\ncarried without solving again or adjusting signs,\nis still exact at $t=0.025$", fontsize=8.3, color=GREEN, va="top", ha="center")
ax.add_patch(FancyBboxPatch((X(0.025) + 0.055, 0.8), 0.2, 3.9, boxstyle="round,pad=0.02", fc="#ECFDF5", ec=GREEN, lw=1))
ax.text(X(0.025) + 0.155, 2.75, "held-out end point\n$t = 0.025$\n\nall 47 products\nexact, both directions\n(inverse included)\n\nwrong carried maps\nfail (PW3)", ha="center", va="center", fontsize=8, color=INK)
ax.set_xlim(-0.22, 1.36); ax.set_ylim(-4.9, 6.9); ax.axis("off")
for t, lab in ((-0.1, "$-0.1$"), (-0.04, "$-0.04$"), (0.010, "$0.010$"), (0.025, "$0.025$")): ax.text(X(t), -1.45, lab, ha="center", fontsize=8, color=MUTED)
ax.text(X(-0.0375), -1.75, "margin parameter $t$ (ladder points marked)", ha="center", fontsize=8, color=MUTED)
# quivers
def quiver(a, labels, arrows, title):
    xs = np.linspace(0.07, 0.93, 6); yv = [0.62, 0.25, 0.45, 0.45, 0.45, 0.45]; P = list(zip(xs, yv))
    for (i, j, m, col) in arrows:
        (x1, y1), (x2, y2) = P[i], P[j]; rad = 0.0 if abs(j - i) == 1 or (i < 2 and j == 2) else -0.35
        a.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, color=col, lw=1.2, shrinkA=10, shrinkB=10, connectionstyle=f"arc3,rad={rad}"))
        if m > 1: a.text((x1 + x2) / 2, (y1 + y2) / 2 + (0.22 if rad else -0.09), f"{m}", ha="center", fontsize=8.5, color=col, fontweight="bold")
    for k, ((x, y), lab) in enumerate(zip(P, labels)):
        a.plot(x, y, "o", ms=11, mfc="white", mec=INK, mew=1, zorder=5); a.text(x, y + 0.07, lab, ha="center", va="bottom", fontsize=7.5, color=INK)
    a.set_title(title, fontsize=9, color=INK); a.set_xlim(0, 1); a.set_ylim(0, 0.95); a.axis("off")
quiver(qa, ["$F_1$", "$F_2$", "$F_3$", "$F_4$", "$F_5=O$", "$F_6=O(H)$"],
       [(0, 2, 1, INK), (1, 2, 1, INK), (2, 3, 1, INK), (2, 4, 1, INK), (3, 4, 2, INK), (4, 5, 3, BLUE)], "before the wall: $C'$ (Study 76)")
quiver(qb, ["$F_1$", "$F_2$", "$F_3$", "$F_4$", "$T$", "$F_5=O$"],
       [(0, 2, 1, INK), (1, 2, 1, INK), (2, 3, 1, INK), (2, 4, 2, ORANGE), (3, 4, 1, ORANGE), (4, 5, 3, ORANGE)], "after the wall: $C''$ (Studies 81-82)")
fig.savefig(os.path.join(HERE, "fig_path82b.pdf")); fig.savefig(os.path.join(HERE, "fig_path82b.png"), dpi=120); print("written")
