"""Figure for Phase III Stage 1: relations and syzygies. (a) the bowtie cover and its Hilbert-Burch matrix; (b) a generator as a
disconnected fibre: two factorisations of v0 v1 v2 v3 v4 r; (c) a first syzygy as a hollow triangle: three factorisations of
v0^2 v1^2 v2^3 v3 v4 r, pairwise linked by the three generators, with no common link; (d) test degrees of freedom: the codimension as
triples are added, and its drop at a coincidence on each vertex (bowtie, strip3)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch
HERE = os.path.dirname(os.path.abspath(__file__)); D = json.load(open(os.path.join(HERE, "..", "lab", "syzygy.json")))
BLUE, ORANGE, VIOLET, AQUA, GREY, CRIT = "#2a78d6", "#eb6834", "#6250d6", "#1baf7a", "#8c8c8c", "#d03b3b"; INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 4, figsize=(16, 4.0), gridspec_kw=dict(width_ratios=[1.05, 1, 1.15, 1]))
a = ax[0]; a.axis("off"); a.set_xlim(-1.3, 1.3); a.set_ylim(-1.25, 1.1)
pos = {0: (-1.0, 0.55), 1: (-1.0, -0.15), 2: (0, 0.2), 3: (1.0, 0.55), 4: (1.0, -0.15)}
for T, col in (((0, 1, 2), BLUE), ((2, 3, 4), ORANGE)):
    a.add_patch(Polygon([pos[i] for i in T], closed=True, fc=col, alpha=0.18, ec=col, lw=1.5))
for i, p in pos.items(): a.scatter(*p, color=INK, zorder=3, s=20); a.text(p[0], p[1] + 0.09, str(i), ha="center", fontsize=8)
a.text(-1.05, -0.62, "rank", ha="center", va="center", fontsize=9)
for j, (t1, t2) in enumerate((("$m_{012}$", "$m_{234}$"), ("$c_{01}$", "$c_{34}$"), ("$c_{02}c_{12}$", "$c_{23}c_{24}$"))):
    a.text(-0.55 + 0.5 * j, -0.5, t1, ha="center", va="center", fontsize=9); a.text(-0.55 + 0.5 * j, -0.75, t2, ha="center", va="center", fontsize=9)
a.text(-0.86, -0.62, "(", ha="center", va="center", fontsize=26, color=INK2); a.text(0.73, -0.62, ")", ha="center", va="center", fontsize=26, color=INK2)
a.text(1.0, -0.62, "$\\leq 1$", ha="center", va="center", fontsize=9)
a.text(0, -1.05, "each column $\\propto v_0v_1:v_3v_4$; three minors = the generators;\nthe two rows = the two syzygies (Hilbert–Burch)", ha="center", fontsize=7.5, color=INK2)
a.set_title("(a) the bowtie: windows $\\{0,1,2\\}$, $\\{2,3,4\\}$", fontsize=9)
def box(a, xy, s, col):
    a.add_patch(FancyBboxPatch((xy[0] - 0.42, xy[1] - 0.13), 0.84, 0.26, boxstyle="round,pad=0.02", fc="white", ec=col, lw=1.4, zorder=3)); a.text(*xy, s, ha="center", va="center", fontsize=8, zorder=4)
a = ax[1]; a.axis("off"); a.set_xlim(-1, 1); a.set_ylim(-1, 1)
box(a, (-0.45, 0.25), "$c_{34}\\,m_{012}$", BLUE); box(a, (0.45, 0.25), "$c_{01}\\,m_{234}$", ORANGE)
a.text(0, 0.62, "$b=v_0v_1v_2v_3v_4\\,r$", ha="center", fontsize=9)
a.text(0, -0.25, "no common factor: $\\Delta_b$ = two disjoint edges,\n$\\tilde H_0(\\Delta_b)=\\mathbf{Q}$: a minimal generator\n$c_{34}m_{012}-c_{01}m_{234}$ (the second-triple test)", ha="center", fontsize=7.8, color=INK2)
a.set_title("(b) a relation = a disconnected fibre", fontsize=9)
a = ax[2]; a.axis("off"); a.set_xlim(-1.25, 1.25); a.set_ylim(-1.05, 1.05)
P = {"A": (0, 0.72), "B": (-0.8, -0.45), "C": (0.8, -0.45)}
lab = {"A": "$c_{01}\\,c_{02}c_{12}\\,m_{234}$", "B": "$c_{01}\\,c_{23}c_{24}\\,m_{012}$", "C": "$c_{02}c_{12}\\,c_{34}\\,m_{012}$"}
for (x, y), (z, w), s in ((P["A"], P["B"], "shared $c_{01}$:\n$g_2$"), (P["A"], P["C"], "shared $c_{02}c_{12}$:\n$g_1$"), (P["B"], P["C"], "shared $m_{012}$: $g_3$")):
    a.plot([x, z], [y, w], color=VIOLET, lw=2); a.text((x + z) / 2 + (0.0 if y != w else 0), (y + w) / 2 + (-0.17 if y == w else 0.0), s, ha="center", va="center", fontsize=7.2, color=VIOLET,
                                                          bbox=dict(fc="white", ec="none", pad=0.5))
for k, xy in P.items(): box(a, xy, lab[k], INK2)
a.text(0, 0.05, "hollow:\nno factor\ncommon\nto all three", ha="center", va="center", fontsize=7.2, color=CRIT)
a.text(0, -0.95, "$b=v_0^2v_1^2v_2^3v_3v_4\\,r$: $\\tilde H_1(\\Delta_b)=\\mathbf{Q}$, a first syzygy", ha="center", fontsize=8)
a.set_title("(c) a syzygy = a loop of relations", fontsize=9)
a = ax[3]; df = D["tests_df"]
for k, col, off in (("bowtie", BLUE, -0.18), ("strip3", ORANGE, 0.18)):
    rk = df[k]["coincidence_rank"]; xs = np.arange(len(rk)) + off
    a.bar(xs, rk, width=0.34, color=col, label=f"{k}: df {df[k]['codim']} generically")
a.set_xticks(range(7)); a.set_xlabel("vertex $i$ with $v_i=0$"); a.set_ylabel("rank of the tests' Jacobian (df)")
a.legend(frameon=False, fontsize=7.5, loc="upper right"); a.set_ylim(0, 5.3)
a.set_title("(d) df drops only at a coincidence on a cut vertex\n(bowtie: 2; strip: 2 and 4)", fontsize=9)
plt.tight_layout(); fig.savefig(os.path.join(HERE, "fig_syzygy.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_syzygy.png"), dpi=150, bbox_inches="tight"); print("ok")
