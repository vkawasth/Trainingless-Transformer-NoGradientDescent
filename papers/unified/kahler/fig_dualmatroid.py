"""The central dual-matroid diagram (sec:credalshape, lem:cographic). One nondegenerate 3 x 3 vertex, computed exactly: margins
f = (1/2, 1/3, 1/6), g = (2/5, 7/20, 1/4), maximising a fixed linear objective. Left: the table, positive cells (the basis) shaded.
Middle: the same cells as edges of K_{3,3}; the positive cells form a spanning tree (graphic matroid basis), the zero cells its
complement (cographic basis). Right: what each side is used for."""
import os
from fractions import Fraction as Fr
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog
HERE = os.path.dirname(os.path.abspath(__file__))
f = [Fr(1, 2), Fr(1, 3), Fr(1, 6)]; g = [Fr(2, 5), Fr(7, 20), Fr(1, 4)]; cells = [(i, j) for i in range(3) for j in range(3)]
A = np.array([[1 if c[0] == i else 0 for c in cells] for i in range(3)] + [[1 if c[1] == j else 0 for c in cells] for j in range(3)], float)
w = np.array([3, 1, 0, 0, 2, 1, 1, 0, 2], float); x = linprog(-w, A_eq=A, b_eq=[float(v) for v in f + g], bounds=[(0, None)] * 9, method="highs").x
pos = [k for k in range(9) if x[k] > 1e-10]; assert len(pos) == 5
INK, MUTED, TEAL, RED = "#222222", "#666666", "#2b8a7e", "#b3261e"
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.3), gridspec_kw=dict(width_ratios=[1, 1.2, 1.5], wspace=0.15))
ax = axs[0]
for k, (i, j) in enumerate(cells):
    ax.add_patch(plt.Rectangle((j, 2 - i), 1, 1, fc="#cfe8e4" if k in pos else "white", ec=MUTED))
    ax.text(j + 0.5, 2.5 - i, f"{Fr(x[k]).limit_denominator(1000)}" if k in pos else "0", ha="center", va="center", fontsize=8, color=INK if k in pos else RED)
for i in range(3): ax.text(-0.15, 2.5 - i, f"f{i+1} = {f[i]}", ha="right", va="center", fontsize=7, color=MUTED)
for j in range(3): ax.text(j + 0.5, 3.15, f"g{j+1} = {g[j]}", ha="center", fontsize=7, color=MUTED)
ax.set_xlim(-1.3, 3.1); ax.set_ylim(-0.2, 3.5); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("A vertex of the coupling fibre", loc="left", fontsize=8.4, fontweight="bold", color=INK)
ax = axs[1]; R = {i: (0, 2 - i) for i in range(3)}; C = {j: (2, 2 - j) for j in range(3)}
for k, (i, j) in enumerate(cells):
    (x0, y0), (x1, y1) = R[i], C[j]
    ax.plot([x0, x1], [y0, y1], color=TEAL if k in pos else RED, lw=2.2 if k in pos else 1.0, ls="-" if k in pos else (0, (3, 2)))
for i in range(3): ax.scatter(*R[i], s=120, color="white", ec=INK, zorder=3); ax.text(R[i][0] - 0.18, R[i][1], f"row {i+1}", ha="right", va="center", fontsize=7)
for j in range(3): ax.scatter(*C[j], s=120, color="white", ec=INK, zorder=3); ax.text(C[j][0] + 0.18, C[j][1], f"col {j+1}", ha="left", va="center", fontsize=7)
ax.plot([], [], color=TEAL, lw=2.2, label="positive cells: a spanning tree (graphic basis)"); ax.plot([], [], color=RED, ls=(0, (3, 2)), label="zero cells: the complement (cographic basis)")
ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.02)); ax.set_xlim(-0.9, 2.9); ax.set_ylim(-0.4, 2.3); ax.axis("off")
ax.set_title("The same cells as edges of K(3,3)", loc="left", fontsize=8.4, fontweight="bold", color=INK)
ax = axs[2]; ax.axis("off")
box = dict(boxstyle="round,pad=0.5", fc="#eef7f7", ec=TEAL, lw=0.8); box2 = dict(boxstyle="round,pad=0.5", fc="#fbeaea", ec=RED, lw=0.8)
ax.text(0.02, 0.86, "graphic side (positive cells)\nthe LP basis: x_B = A_B^{-1} b moves linearly\ninside a chamber; reused across tables by the\nstratified ledger (basis count; chambers refine regions)", fontsize=6.9, va="top", bbox=box, transform=ax.transAxes)
ax.text(0.02, 0.43, "cographic side (zero cells)\nthe tight facets x_ij = 0: their normals span the\nvertex cone; the same cell normals whose rank\nincrements give the scale multiplicities (one scale per dimension)", fontsize=6.9, va="top", bbox=box2, transform=ax.transAxes)
ax.text(0.02, 0.02, "rc = (r + c - 1) + (r - 1)(c - 1): one graph, two dual bases.\nShared substrate; neither theorem implies the other.", fontsize=6.6, color=MUTED, transform=ax.transAxes)
ax.set_title("What each side is used for", loc="left", fontsize=8.4, fontweight="bold", color=INK)
fig.savefig(os.path.join(HERE, "fig_dualmatroid.png"), dpi=220, bbox_inches="tight"); print("ok", pos)
