"""Figures for section 12.9 of the companion note: composition by selection and its associahedral configuration.
  fig_as_pentagon.pdf   (a) K_4 with its faces labelled by bracketings and the binary composites; (b) the 11 binary composites on the endpoint segment
  fig_as_polytope.pdf   the 2 -> 3 state example: the five bracketed composites, the five edge composites and the top inside Pi(p0, p4)"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from pce import selcomp as S, couplingfibre as CF
BLUE, ORANGE, AQUA, VIOLET, ROSE, INK, INK2, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#c2417a", "#22211f", "#52514e", "#9a9a9a"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
D4 = S.dissections(4)
CYC = [frozenset({(0, 2), (0, 3)}), frozenset({(0, 3), (1, 3)}), frozenset({(1, 3), (1, 4)}), frozenset({(1, 4), (2, 4)}), frozenset({(0, 2), (2, 4)})]
BR = {frozenset({(0, 2), (0, 3)}): "$((\\gamma_4\\gamma_3)\\gamma_2)\\gamma_1$"}
def brk(d):
    names = {frozenset({(0, 2), (0, 3)}): "((12)3)4", frozenset({(0, 3), (1, 3)}): "(1(23))4", frozenset({(1, 3), (1, 4)}): "1((23)4)",
             frozenset({(1, 4), (2, 4)}): "1(2(34))", frozenset({(0, 2), (2, 4)}): "(12)(34)"}
    return names[d]
def edge_name(e): (i, j), = tuple(e); return f"block {i}..{j}"
def cpl(p, q, t): return [[Fr(t), p[0] - Fr(t)], [q[0] - Fr(t), 1 - p[0] - q[0] + Fr(t)]]
PB = [(Fr(2, 5), Fr(3, 5)), (Fr(1, 3), Fr(2, 3)), (Fr(3, 7), Fr(4, 7)), (Fr(1, 4), Fr(3, 4)), (Fr(5, 11), Fr(6, 11))]
TB = [Fr(13, 60), Fr(7, 30), Fr(1, 80), Fr(3, 80)]; GB = [cpl(PB[i], PB[i + 1], TB[i]) for i in range(4)]
VB = {d: S.composite(GB, d)[0][0] for d in D4}; MB = S.composite(GB, frozenset(), S.markov_rule)[0][0]

fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.9), gridspec_kw={"width_ratios": [1, 1.25]})
a = ax[0]; ang = np.pi / 2 + 2 * np.pi * np.arange(5) / 5; P = np.c_[np.cos(ang), np.sin(ang)]
a.add_patch(Polygon(P, closed=True, fc="#eef3fa", ec=INK2, lw=1.2))
for k, d in enumerate(CYC):
    x, y = P[k]; a.plot(x, y, "o", color=BLUE, ms=8, zorder=5)
    a.text(1.22 * x, 1.2 * y, f"{brk(d)}\n${VB[d]}$", ha="center", va="center", fontsize=8.2)
    e = CYC[k] & CYC[(k + 1) % 5]; m = (P[k] + P[(k + 1) % 5]) / 2; a.plot(*m, "s", color=ORANGE, ms=6, zorder=5)
    (i, j), = tuple(e); a.text(0.72 * m[0], 0.72 * m[1], f"[{i}..{j}]\n${VB[e]}$", ha="center", va="center", fontsize=7.4, color=INK2)
a.plot(0, 0, "*", color=ROSE, ms=13, zorder=5); a.text(0, -0.16, f"top: ${VB[frozenset()]}$", ha="center", fontsize=8)
a.set_xlim(-1.55, 1.55); a.set_ylim(-1.45, 1.5); a.set_aspect("equal"); a.axis("off")
a.set_title("(a) $K_4$ for the binary chain, each face with the $a_{00}$ entry of its\ncomposite. Vertices: bracketings; edges: a block $[i..j]$ (one 4-way\ntable); centre: one selection in the 5-way table", fontsize=9)
b = ax[1]; lo = max(Fr(0), PB[0][0] + PB[4][0] - 1); hi = min(PB[0][0], PB[4][0])
b.plot([0.14, 0.26], [0, 0], color=INK2, lw=2)
for k, d in enumerate(CYC): b.plot(float(VB[d]), 0, "o", color=BLUE, ms=8, zorder=5); b.text(float(VB[d]), 0.08 + 0.07 * (k % 2), brk(d), ha="center", fontsize=7.5, color=BLUE)
for k, d in enumerate(CYC):
    e = CYC[k] & CYC[(k + 1) % 5]; b.plot(float(VB[e]), 0, "s", color=ORANGE, ms=6, zorder=6)
b.plot(float(VB[frozenset()]), 0, "*", color=ROSE, ms=13, zorder=7); b.plot(float(MB), 0, "D", color=AQUA, ms=7, zorder=7)
b.text(float(MB), -0.09, "Markov:\nall faces", ha="center", fontsize=7.5, color=AQUA, va="top")
b.plot([], [], "o", color=BLUE, label="bracketings (vertices)"); b.plot([], [], "s", color=ORANGE, label="edges: a block of three couplings")
b.plot([], [], "*", color=ROSE, ms=10, label="top: one selection in the 5-way table"); b.plot([], [], "D", color=AQUA, label="Markov gluing (every face)")
b.set_ylim(-0.3, 0.35); b.set_yticks([]); b.spines["left"].set_visible(False)
b.set_xlim(0.14, 0.26); b.legend(frameon=False, fontsize=7.8, loc="lower left", ncol=2)
b.set_xlabel(f"$a_{{00}}$ of the endpoint coupling; the segment $\\Pi(p_0,p_4)$ is $[{lo},{hi}]$ (zoomed to $[0.14,0.26]$)")
b.set_title("(b) The eleven composites on the endpoint segment (zoomed): selection is not associative,\nand each edge composite lies between its two bracketings", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_as_pentagon.pdf")); fig.savefig(os.path.join(HERE, "fig_as_pentagon.png"), dpi=150); plt.close(fig)

# ---------------------------------------------------------------- 2 -> 3 states
G3 = [[[Fr(1, 150), Fr(19, 150), Fr(4, 15)], [Fr(4, 25), Fr(31, 150), Fr(7, 30)]],
      [[Fr(7, 50), Fr(2, 75)], [Fr(18, 175), Fr(121, 525)], [Fr(13, 70), Fr(11, 35)]],
      [[Fr(3, 35), Fr(81, 1400), Fr(57, 200)], [Fr(4, 35), Fr(339, 1400), Fr(43, 200)]],
      [[Fr(9, 125), Fr(23, 375), Fr(1, 15)], [Fr(209, 3000), Fr(79, 750), Fr(1, 8)], [Fr(13, 120), Fr(1, 6), Fr(9, 40)]]]
V3 = {d: S.composite(G3, d) for d in D4}; pt = {d: (float(c[0][0]), float(c[0][1])) for d, c in V3.items()}
M3 = S.composite(G3, frozenset(), S.markov_rule); pm = (float(M3[0][0]), float(M3[0][1]))
f0, g0 = (Fr(2, 5), Fr(3, 5)), (Fr(1, 4), Fr(1, 3), Fr(5, 12)); Pp, _ = CF.polygon(f0, g0); PF = [(float(x), float(y)) for x, y in Pp]
fig, ax = plt.subplots(1, 2, figsize=(11.6, 5.0))
for a, zoom in zip(ax, (False, True)):
    a.add_patch(Polygon(PF, closed=True, fc="#f5f5f5", ec=INK2, lw=1))
    V = np.array([pt[d] for d in CYC]); from scipy.spatial import ConvexHull; hv = ConvexHull(V); a.add_patch(Polygon(V[hv.vertices], closed=True, fc=BLUE, alpha=0.08, ec="none"))
    for k, d in enumerate(CYC):
        e = CYC[k] & CYC[(k + 1) % 5]; q = np.array([pt[d], pt[e], pt[CYC[(k + 1) % 5]]]); a.plot(q[:, 0], q[:, 1], color=INK2, lw=0.9, zorder=3)
        a.plot(*pt[e], "s", color=ORANGE, ms=6, zorder=5)
        a.plot(*pt[d], "o", color=BLUE, ms=8, zorder=6)
        if zoom: a.annotate(brk(d), pt[d], (pt[d][0] + 0.0008, pt[d][1] + 0.0004), fontsize=7.6, color=BLUE)
    a.plot(*pt[frozenset()], "*", color=ROSE, ms=14, zorder=7); a.plot(*pm, "D", color=AQUA, ms=7, zorder=7)
    if zoom:
        xs = [pt[d][0] for d in D4] + [pm[0]]; ys = [pt[d][1] for d in D4] + [pm[1]]
        a.set_xlim(min(xs) - 0.004, max(xs) + 0.006); a.set_ylim(min(ys) - 0.003, max(ys) + 0.003)
        a.set_title("(b) zoom: $K_4$ mapped in: bracketings (blue), edge composites (orange) on the broken edges;\nthe image folds (the edges cross); the top (star) lies inside the hull, Markov (aqua) outside", fontsize=9)
    else:
        xs = [pt[d][0] for d in D4] + [pm[0]]; ys = [pt[d][1] for d in D4] + [pm[1]]
        a.add_patch(Rectangle((min(xs) - 0.004, min(ys) - 0.003), max(xs) - min(xs) + 0.01, max(ys) - min(ys) + 0.006, fc="none", ec=ROSE, lw=1))
        a.set_title("(a) the endpoint polytope $\\Pi(p_0,p_4)$, $p_0=(2/5,3/5)$, $p_4=(1/4,1/3,5/12)$;\nthe box is enlarged on the right", fontsize=9)
    a.set_xlabel("$a_{00}$"); a.set_ylabel("$a_{01}$"); a.set_aspect("equal" if not zoom else "auto")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_as_polytope.pdf")); fig.savefig(os.path.join(HERE, "fig_as_polytope.png"), dpi=150); plt.close(fig); print("ok")
