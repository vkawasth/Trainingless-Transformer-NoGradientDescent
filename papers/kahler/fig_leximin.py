"""Figure for section 11.6 of the companion note: the selection theorem on a column-binding example, f = (2/5, 3/5), g = (1/2, 3/10, 1/5)
(the coupling-polytopes paper, Example 3.4(b)). (a) the fibre and the two levels; (b) the functions along the candidate segment; (c) the
finite-T correction ln 2 / (2L); (d) the thimble along the segment."""
import os, sys, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from pce import couplingfibre as CF, datawalls as DW, kahlerbridge as KB
BLUE, ORANGE, AQUA, VIOLET, ROSE, INK, INK2, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#c2417a", "#22211f", "#52514e", "#9a9a9a"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
from fractions import Fraction as Fr
f, g = (Fr(2, 5), Fr(3, 5)), (Fr(1, 2), Fr(3, 10), Fr(1, 5)); P, A = CF.polygon(f, g); PF = np.array([[float(x), float(y)] for x, y in P])
F = CF.facet_cells([0.4, 0.6], [0.5, 0.3, 0.2]); V, L = CF.cell_functions(0.4, 0.5, 0.3); vf = [V[c] for c in F]; lf = [L[c] for c in F]
fig, ax = plt.subplots(2, 2, figsize=(11.2, 9.4))
a = ax[0, 0]; a.add_patch(Polygon(PF, closed=True, fc="#f3f3f3", ec=INK2, lw=1.2))
T = CF.tables(100, f, g); S = np.array([s for s, _ in T]) / 100; a.plot(S[:, 0], S[:, 1], ".", color=GREY, ms=2.2)
a.plot([0.1, 0.2], [0.2, 0.1], color=ORANGE, lw=5, alpha=0.75, label="level 1: cells 13 = 23 = 1/10 (column 3 spread evenly);\nthe candidates, an open segment")
a.plot([0.1, 0.2], [0.2, 0.1], "o", ms=8, mfc="white", mec=ORANGE, mew=2, label="endpoints: maximisers, not candidates")
a.plot(0.15, 0.15, "o", mfc="none", mec=BLUE, mew=2.2, ms=13, label="level 2: cells 11 = 12 = 22 = 3/20; the selected coupling")
a.plot(0.5 * 0.4, 0.4 * 0.3, "s", color=AQUA, ms=7, mec="white", label="product coupling $f\\otimes g$")
a.set_xlim(-0.03, 0.43); a.set_ylim(-0.03, 0.33); a.set_aspect("equal"); a.set_xlabel("$x=\\alpha_{11}$"); a.set_ylabel("$y=\\alpha_{12}$")
a.legend(frameon=False, fontsize=7.2, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1); a.set_title("(a) A column-binding fibre: $g_3/2=1/10$ is the smallest ratio,\nso level 1 leaves a segment; level 2 picks its midpoint", fontsize=9.5)
b = ax[0, 1]; xs = np.linspace(0.1, 0.2, 200)
b.plot(xs, xs, color=BLUE, lw=2.4, label="cells 11 and 22: both equal $x$ on the segment")
b.plot(xs, 0.3 - xs, color=ROSE, lw=2.4, label="cell 12: $3/10-x$")
b.plot(xs, np.full_like(xs, 0.1), color=ORANGE, lw=2.4, label="cells 13, 23: fixed at $1/10$ (level 1)")
b.plot(xs, 0.5 - xs, color=GREY, lw=1.2, ls=":", label="cell 21: never a facet")
b.plot(xs, np.minimum(xs, 0.3 - xs), color=INK, lw=1, ls="--", label="the least free cell"); b.plot(0.15, 0.15, "o", color=BLUE, ms=8, mec="white")
b.annotate("max-min at $x=3/20$", (0.15, 0.15), (0.155, 0.19), fontsize=8, arrowprops=dict(arrowstyle="-", color=INK2))
b.set_xlabel("position $x$ along the segment"); b.set_ylabel("cell value"); b.set_ylim(0.08, 0.42); b.legend(frameon=False, fontsize=7.2, loc="upper right")
b.set_title("(b) Level 2 along the segment: the least of the\nremaining cells is largest at the midpoint", fontsize=9.5)
c = ax[1, 0]; Ls = np.array([20, 30, 50, 75, 100, 150, 200, 300]); corr = []
for Lg in Ls:
    u = DW.softmin_point(vf, lf, float(Lg), u0=(0.15, 0.15)); corr.append(Lg * (u[0] - 0.15))
c.plot(Ls, corr, "o-", color=BLUE, lw=2, ms=5, label="$L\\,(x_L-3/20)$, positive critical point, $T=e^{-L}$")
c.axhline(math.log(2) / 2, color=INK, lw=1, ls="--"); c.text(Ls[-1], math.log(2) / 2 + 0.004, "$\\ln 2/2$", ha="right", fontsize=8.5)
c.set_xscale("log"); c.set_xlabel("$L=\\log(1/T)$"); c.set_ylabel("$L\\times$ offset from the leximin point"); c.legend(frameon=False, fontsize=7.5, loc="upper right")
c.set_title("(c) The selected point at finite $T$: $x_L=3/20+\\ln 2/(2L)+\\dots$ —\ntwo cells pull one way, one the other", fontsize=9.5)
d = ax[1, 1]; ramp = ["#b7d3f2", "#7aaee6", "#2a78d6", "#123f7a"]
for col, h in zip(ramp, (0.1, 0.12, 0.14, 0.16)):
    U1, U2, w = KB.tropical_grid(vf, [float(x) for x in lf], -300.0, h, box=((0.05, 0.25), (0.05, 0.25)), n=801); m1 = w.sum(1); a1 = U1[:, 0]
    d.plot(a1, m1 / (m1.sum() * (a1[1] - a1[0])), color=col, lw=2, label={0.1: "$h=1/10$: uniform on the segment", 0.12: "$h=0.12$: uniform on $[h,\\,3/10-h]$",
                                                                         0.14: "$h=0.14$: narrower", 0.16: "$h=0.16>3/20$: at the selected point (peak cut)"}[h])
d.axvline(0.15, color=INK2, lw=0.8, ls=":"); d.set_xlim(0.06, 0.24); d.set_ylim(0, 60); d.set_xlabel("$x=\\log y_1/\\log T$"); d.set_ylabel("density of the $x$-marginal")
d.legend(frameon=False, fontsize=7.2, loc="upper right"); d.set_title("(d) The positive thimble ($\\hbar=T^h$) along the segment:\nthe two levels appear as $h$ grows", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_leximin.pdf")); fig.savefig(os.path.join(HERE, "fig_leximin.png"), dpi=150); plt.close(fig); print("ok")
