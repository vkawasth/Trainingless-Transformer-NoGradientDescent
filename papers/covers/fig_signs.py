"""Figure for Phase III, Stage 3 (real signs). (a) two disjoint triangles: one triple leaves the second triangle's labels free (two
sheets), a triple on each aligns them through the shared r; (b) the triangle's eight data sign patterns: four realised, four allowed by
every relation (there are none) but realised by nothing (the invisible odd-cycle class); (c) b_1 = visible + invisible over the zoo;
(d) the sign ladder: what one new face does to (rank_Q A, rank_2 A)."""
import json, os, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
HERE = os.path.dirname(os.path.abspath(__file__)); D = json.load(open(os.path.join(HERE, "..", "lab", "signs.json")))
BLUE, ORANGE, VIOLET, AQUA, GREY, CRIT = "#2a78d6", "#eb6834", "#6250d6", "#1baf7a", "#8c8c8c", "#d03b3b"; INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 4, figsize=(16, 4.1), gridspec_kw=dict(width_ratios=[1.3, 1, 1.15, 1.05]))
a = ax[0]; a.axis("off"); a.set_xlim(-0.2, 4.4); a.set_ylim(-1.6, 1.6)
def tri(cx, cy, fill, ec, lab):
    pts = np.array([[cx - .45, cy - .35], [cx + .45, cy - .35], [cx, cy + .4]]); a.add_patch(Polygon(pts, closed=True, fc=fill, ec=ec, lw=1.6, alpha=0.9))
    a.text(cx, cy - .05, lab, ha="center", fontsize=8)
for row, (y, both) in enumerate(((0.65, False), (-0.8, True))):
    tri(0.6, y, "#dbe7f8", BLUE, "A: triple 012"); tri(2.3, y, "#dbe7f8" if both else "white", BLUE if both else GREY, "B: triple 345" if both else "B: pairs only")
    a.text(3.35, y + 0.2, "$r$", fontsize=11, color=VIOLET); 
    a.add_patch(FancyArrowPatch((1.05, y + 0.1), (3.3, y + 0.25), arrowstyle="-|>", mutation_scale=10, color=VIOLET, lw=1.2, connectionstyle="arc3,rad=-0.45"))
    if both: a.add_patch(FancyArrowPatch((2.75, y + 0.1), (3.3, y + 0.2), arrowstyle="-|>", mutation_scale=10, color=VIOLET, lw=1.2))
    a.text(3.55, y - 0.15, "1 sheet:\nlabels aligned" if both else "2 sheets:\nB's labels free", fontsize=7.8, color=INK if both else CRIT)
a.text(0.0, 1.5, "$\\sigma_A=\\sigma_r$ from A's triple; $\\sigma_B=\\sigma_r$ only if B has one", fontsize=7.8)
a.set_title("(a) relative labels travel through the shared $r$\n(no window meets both triangles)", fontsize=9)
a = ax[1]; a.axis("off"); a.set_xlim(-0.6, 1.9); a.set_ylim(-0.5, 1.75)
P = lambda s: np.array([s[0] + 0.55 * s[2], s[1] + 0.45 * s[2]])
for s in itertools.product((0, 1), repeat=3):
    for k in range(3):
        if s[k] == 0:
            t = list(s); t[k] = 1; a.plot(*np.array([P(s), P(t)]).T, color=GREY, lw=0.8)
for s in itertools.product((0, 1), repeat=3):
    real = sum(s) % 2 == 0; a.scatter(*P(s), s=120, color=BLUE if real else CRIT, zorder=4, edgecolor="white")
    a.text(*(P(s) + np.array([0.06, 0.06])), "".join("-" if x else "+" for x in s), fontsize=7.5)
a.text(-0.55, -0.42, "signs of $(c_{01},c_{12},c_{02})$; blue: realised (even number of $-$);\nred: no relation forbids them, yet no real two-type law has them", fontsize=7.2)
a.set_title("(b) the triangle's invisible class:\n$\\mathbb{Z}/2$ torsion, Smith form $(1,1,2)$", fontsize=9)
a = ax[2]; sp = D["pair_split"]; ks = list(sp.keys()); vis = [sp[k]["visible"] for k in ks]; inv = [sp[k]["invisible"] for k in ks]
y = np.arange(len(ks))[::-1]; a.barh(y, vis, color=BLUE, label="visible: sign shadows of relations (= codim $I_B$)"); a.barh(y, inv, left=vis, color=CRIT, label="invisible: one odd-cycle class per\nnon-bipartite component")
a.set_yticks(y); a.set_yticklabels(ks, fontsize=7.5); a.set_xlabel("sign classes $b_1=\\dim H^1(G;\\mathbb{Z}/2)$"); a.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=1)
a.set_title("(c) which sign classes the relations see", fontsize=9)
a = ax[3]; cnt = D["ladder"]["counts"]; order = ["(1, 1)", "(0, 0)", "(1, 0)", "(0, 1)"]
lab = {"(1, 1)": "(1,1)\nabsorbed,\nsign fixed", "(0, 0)": "(0,0)\na test", "(1, 0)": "(1,0)\nodd cycle\nclosed", "(0, 1)": "(0,1)\ntest that\nfixes a sign"}
cols = [BLUE, VIOLET, CRIT, AQUA]; a.bar(range(4), [cnt.get(k, 0) for k in order], color=cols); a.set_yscale("log")
a.set_xticks(range(4)); a.set_xticklabels([lab[k] for k in order], fontsize=6.8); a.set_ylabel("faces added (random covers)")
for i, k in enumerate(order): a.text(i, cnt.get(k, 0) * 1.15, str(cnt.get(k, 0)), ha="center", fontsize=7.5)
a.set_title("(d) the sign ladder: change of\n$(\\mathrm{rank}_\\mathbb{Q}A,\\ \\mathrm{rank}_2A)$ per new face", fontsize=9)
plt.tight_layout(); fig.savefig(os.path.join(HERE, "fig_signs.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_signs.png"), dpi=150, bbox_inches="tight"); print("ok")
