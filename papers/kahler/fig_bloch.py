"""Figure for prop:bloch of the companion note (module pce/blochtoric.py).
  fig_bloch.pdf  (a) example A as three Bloch spheres (columns 1, 3, 4 of the 2 x 4 table): latitude = P(row 2 | column) = 1/2, longitude =
                 holonomy, so the eight objects are the sigma_x product basis; (b) B's six holonomies as the cyclic group Z/6; (c) C's eight
                 holonomies over the leximin fibre as Z/4 x Z/2, and the Z/2-coset over the second fibre. Dials as in fig:tfobjects."""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, FancyArrowPatch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from pce import blochtoric as B
from pce import cpl24 as C
BLUE, ORANGE, ROSE, INK, INK2, GREY = "#2a78d6", "#eb6834", "#c2417a", "#22211f", "#52514e", "#b8b8b4"
DC = [BLUE, ORANGE, ROSE]
plt.rcParams.update({"font.size": 9.5})
fig = plt.figure(figsize=(14, 5.0)); gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.9, 1.15], wspace=0.08)

# (a) three Bloch spheres
a = fig.add_subplot(gs[0]); a.set_axis_off(); a.set_aspect("equal"); a.set_xlim(-1.4, 8.1); a.set_ylim(-3.0, 1.6)
for k, (lab, hol) in enumerate((("column 1\nlongitude arg y1", "y1"), ("column 3\nlongitude arg y3", "y3"), ("column 4\nlongitude arg z", "z"))):
    ox = 2.75 * k
    a.add_patch(Circle((ox, 0), 1, fill=False, ec=INK2, lw=1.1)); a.add_patch(Ellipse((ox, 0), 2, 0.55, fill=False, ec=GREY, lw=0.9, ls="--"))
    a.plot([ox, ox], [-1.15, 1.15], color=GREY, lw=0.8); a.text(ox + 0.05, 1.18, "|0⟩: row 1", fontsize=7.4, color=INK2, va="bottom")
    a.text(ox + 0.05, -1.2, "|1⟩: row 2", fontsize=7.4, color=INK2, va="top")
    for sgn, nm in ((1, "+"), (-1, "−")):
        a.plot(ox + 0.62 * sgn, -0.22 * sgn, "o", ms=9, color=DC[k], mec="white", mew=1.2, zorder=3)
        a.text(ox + 0.62 * sgn, -0.22 * sgn + (0.24 if sgn > 0 else -0.26), f"|{nm}⟩", ha="center", fontsize=8.5, color=INK, va="center")
    a.text(ox, -1.65, lab, ha="center", va="top", fontsize=8, color=INK2)
a.set_title("(a) Example A: three Bloch spheres", fontsize=9.5, loc="left")

def dials(ax, x, y, th, r=0.11, gap=0.27):
    for j in range(3):
        cx = x + (j - 1) * gap; ax.add_patch(Circle((cx, y), r, fill=True, fc="white", ec=INK2, lw=0.8, zorder=3))
        ax.add_patch(FancyArrowPatch((cx, y), (cx + 0.85 * r * np.cos(th[j] * np.pi), y + 0.85 * r * np.sin(th[j] * np.pi)), arrowstyle="-|>", mutation_scale=6,
                                     color=DC[j], lw=1.2, zorder=4))

# (b) B: Z/6
b = fig.add_subplot(gs[1]); b.set_axis_off(); b.set_aspect("equal"); b.set_xlim(-1.6, 1.6); b.set_ylim(-1.75, 1.65)
gen = np.array([1, 1 / 3, 1 / 3]); pos = [(np.cos(np.pi / 2 - k * np.pi / 3), np.sin(np.pi / 2 - k * np.pi / 3)) for k in range(6)]
for k in range(6):
    (x1, y1), (x2, y2) = pos[k], pos[(k + 1) % 6]; b.plot([x1, x2], [y1, y2], color=GREY, lw=1.2, zorder=1)
for k in range(6):
    th = (k * gen) % 2; dials(b, *pos[k], th)
    b.text(pos[k][0] * 1.33, pos[k][1] * 1.33 - 0.02, "0" if k == 0 else f"{k}s", ha="center", va="center", fontsize=8, color=INK2)
b.text(0, 0.06, "Z/6", ha="center", fontsize=12, color=INK); b.text(0, -0.22, "s = (ζ³, ζ, ζ)\nζ = exp(iπ/3)", ha="center", va="top", fontsize=7.6, color=INK2)
b.text(0, -1.5, "fibre (1/40, 1/24, 1/24); 0 is the positive object", ha="center", va="top", fontsize=7.8, color=INK2)
b.set_title("(b) Example B: a cyclic group of order 6", fontsize=9.5, loc="left")

# (c) C: Z/4 x Z/2 and the coset
c = fig.add_subplot(gs[2]); c.set_axis_off(); c.set_aspect("equal"); c.set_xlim(-0.6, 4.2); c.set_ylim(-2.6, 2.0)
g4 = np.array([0, 1, 0.5]); h2 = np.array([1, 1, 0])
for j in range(2):
    yy = 1.0 - 1.15 * j
    for k in range(4):
        c.plot([k * 1.0, (k + 1) * 1.0 if k < 3 else 0], [yy, yy] if k < 3 else [yy, yy], color=GREY, lw=1.2, zorder=1) if k < 3 else None
        if j == 0: c.plot([k * 1.0, k * 1.0], [1.0, -0.15], color=GREY, lw=1.0, ls="--", zorder=1)
        dials(c, k * 1.0, yy, (k * g4 + j * h2) % 2, r=0.1, gap=0.24)
    c.annotate("", xy=(0, yy + 0.22), xytext=(3, yy + 0.22), arrowprops=dict(arrowstyle="->", color=GREY, connectionstyle="arc3,rad=0.25", lw=1))
c.text(3.55, 1.0, "+g", fontsize=8, color=INK2, va="center"); c.text(3.55, -0.15, "+g", fontsize=8, color=INK2, va="center"); c.text(-0.5, 0.42, "+h", fontsize=8, color=INK2, va="center")
c.text(-0.5, 1.75, "fibre (1/20, 3/40, 7/80):  g = (1, −1, i),  h = (−1, −1, 1)", fontsize=7.6, color=INK2)
c.text(-0.5, -0.75, "fibre (1/40, 1/40, 9/40)", fontsize=7.6, color=INK2)
for k, th in enumerate(((0.5, 0.5, 0.5), (1.5, 1.5, 1.5))):
    dials(c, 0.9 + 1.4 * k, -1.25, np.array(th), r=0.1, gap=0.24)
c.text(0.9, -1.55, "(i, i, i)", ha="center", fontsize=7.6, color=INK2); c.text(2.3, -1.55, "(−i, −i, −i)", ha="center", fontsize=7.6, color=INK2)
c.set_title("(c) Example C: Z/4 × Z/2, and a coset", fontsize=9.5, loc="left")
fig.savefig(os.path.join(HERE, "fig_bloch.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_bloch.png"), dpi=150, bbox_inches="tight")
