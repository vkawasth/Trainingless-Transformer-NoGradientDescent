"""Figures for "Two latent types on arbitrary covers" (Phases I-II). Descriptive only.

fig_zoo      (a) the pair-graph zoo with b(G) and the hiding dimension b(G)+1; (b) C4: the hiding set in (log t, log lambda), a rectangle
             in (u, w) = (log lambda t, log lambda/t); (c) triangle + edge: both sign sheets; (d) the triangle at a coincidence v_2 = 0:
             the segment becomes a two-dimensional region
"""
import itertools, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import HalfspaceIntersection, ConvexHull
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from pce import covers as CV
BLUE, ORANGE, VIOLET, AQUA, GREY, CRIT = "#2a78d6", "#eb6834", "#6250d6", "#1baf7a", "#8c8c8c", "#d03b3b"; INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
def save(fig, name): fig.savefig(os.path.join(HERE, name + ".pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, name + ".png"), dpi=150, bbox_inches="tight"); plt.close(fig)

def poly2(A, b):
    z0, rad = CV.chebyshev(A, b)
    if rad <= 1e-9: return None
    hs = HalfspaceIntersection(np.c_[A, -b], z0); pts = hs.intersections; h = ConvexHull(pts); return pts[h.vertices]

fig = plt.figure(figsize=(15.5, 4.0)); gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 1, 1, 1])
# (a) zoo
a = fig.add_subplot(gs[0]); a.axis("off"); a.set_title("(a) pair graphs: $b(G)$ bipartite components; hiding dimension $b(G)+1$", fontsize=9)
zoo = [("$C_3$", 3, [(0, 1), (1, 2), (0, 2)]), ("$C_4$", 4, [(0, 1), (1, 2), (2, 3), (0, 3)]), ("$C_4$+diag", 4, [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2)]),
       ("$C_5$", 5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]), ("$P_4$", 4, [(0, 1), (1, 2), (2, 3)]), ("star", 4, [(0, 1), (0, 2), (0, 3)]),
       ("two edges", 4, [(0, 1), (2, 3)]), ("tri+edge", 5, [(0, 1), (1, 2), (0, 2), (3, 4)]), ("tri+isolated", 4, [(0, 1), (1, 2), (0, 2)]), ("$K_5$", 5, list(itertools.combinations(range(5), 2)))]
for k, (nm, m, E) in enumerate(zoo):
    cx, cy = (k % 5) * 1.25, -(k // 5) * 1.45
    if nm in ("two edges",): pos = {0: (-.25, .25), 1: (-.25, -.25), 2: (.25, .25), 3: (.25, -.25)}
    elif nm == "tri+edge": pos = {0: (-.3, .25), 1: (-.05, -.2), 2: (-.4, -.2), 3: (.3, .25), 4: (.3, -.2)}
    elif nm == "tri+isolated": pos = {0: (-.25, .25), 1: (0.0, -.2), 2: (-.4, -.2), 3: (.3, 0.0)}
    else: pos = {i: (.3 * np.cos(2 * np.pi * i / m + np.pi / 2), .3 * np.sin(2 * np.pi * i / m + np.pi / 2)) for i in range(m)}
    comps = CV.components(m, E); bip = {i: (CV.bipartition(cmp, E) or {}) for cmp in comps for i in cmp}
    for i, j in E: a.plot([cx + pos[i][0], cx + pos[j][0]], [cy + pos[i][1], cy + pos[j][1]], color=INK2, lw=1)
    for i in range(m):
        sd = bip[i].get(i, 0); col = BLUE if sd == 1 else ORANGE if sd == -1 else VIOLET
        a.scatter(cx + pos[i][0], cy + pos[i][1], s=22, color=col, zorder=3)
    b = CV.b_count(m, E); a.text(cx, cy - 0.5, f"{nm}\n$b={b}$, dim {b+1}", ha="center", va="top", fontsize=7.3)
a.set_xlim(-0.55, 5.55); a.set_ylim(-2.6, 0.45)
a.text(-0.5, -2.58, "blue/orange: the two sides of a bipartite component (scale $\\lambda^{\\pm1}$); violet: rigid (odd cycle); an isolated vertex is its own bipartite component", fontsize=6.8, color=INK2)
# (b) C4
a = fig.add_subplot(gs[1]); m, E = 4, [(0, 1), (1, 2), (2, 3), (0, 3)]; w, a1, a2 = 0.4, np.array([.8, .3, .7, .25]), np.array([.2, .7, .15, .7])
mu, c, v = CV.truth_data(w, a1, a2, E); F = CV.factorise(m, E, c)
sg = [float(np.sign(v[0] / F["comps"][0]["v0"][0]))]; A, b, _ = CV.sheet_constraints(mu, F, sg); V = poly2(A, b)
a.add_patch(Polygon(V, closed=True, fc=BLUE, alpha=0.25, ec=BLUE, lw=1.5))
t0 = np.sqrt((1 - w) / w); lam0 = v[0] / F["comps"][0]["v0"][0] * sg[0]; x0 = np.log(abs(lam0)) * (1 if F["comps"][0]["side"][0] == 1 else -1)
a.scatter([np.log(t0)], [x0], color=INK, zorder=5, s=18); a.text(np.log(t0) + 0.03, x0 + 0.03, "truth", fontsize=7.5)
a.set_xlabel("$y=\\log t$ (weight)"); a.set_ylabel("$x=\\log\\lambda$ (bipartite scale)"); a.set_aspect("equal"); a.autoscale_view()
a.set_title("(b) $C_4$: a rectangle in $u=x+y$, $w=x-y$\n(every facet has slope $\\pm1$)", fontsize=9)
# (c) triangle + edge
a = fig.add_subplot(gs[2]); m, E = 5, [(0, 1), (1, 2), (0, 2), (3, 4)]; w, a1, a2 = 0.35, np.array([.8, .3, .7, .25, .6]), np.array([.2, .7, .15, .7, .1])
mu, c, v = CV.truth_data(w, a1, a2, E); F = CV.factorise(m, E, c)
for sig, col, lab in (([1.0, 1.0], BLUE, "sheet $(+,+)$"), ([1.0, -1.0], ORANGE, "sheet $(+,-)$ (labels swapped on $\\{3,4\\}$)")):
    A, b, _ = CV.sheet_constraints(mu, F, sig); V = poly2(A, b)
    if V is not None: a.add_patch(Polygon(V, closed=True, fc=col, alpha=0.25, ec=col, lw=1.5, label=lab))
a.autoscale_view(); a.set_xlabel("$y=\\log t$"); a.set_ylabel("$x=\\log\\lambda_{\\{3,4\\}}$"); a.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
a.set_title("(c) triangle + edge: the triangle bounds $t$,\nthe edge adds a scale; two sheets", fontsize=9)
# (d) coincidence on the triangle
a = fig.add_subplot(gs[3]); m, E = 3, [(0, 1), (1, 2), (0, 2)]
mu, c, v = CV.truth_data(0.4, np.array([.8, .3, .6]), np.array([.2, .7, .1]), E); F = CV.factorise(m, E, c); A, b, _ = CV.sheet_constraints(mu, F, [float(np.sign(v[0] / F["comps"][0]["v0"][0]))])
ylo, yhi = -min(b[A[:, 0] < 0]), min(b[A[:, 0] > 0]); a.plot([ylo, yhi], [0, 0], color=VIOLET, lw=3, label="generic: a segment in $y$ (dim 1)")
mu, c, v = CV.truth_data(0.4, np.array([.8, .3, .6]), np.array([.2, .7, .6]), E); F = CV.factorise(m, E, c); A, b, _ = CV.sheet_constraints(mu, F, [float(np.sign(v[0] / F["comps"][0]["v0"][0]))])
V = poly2(A, b); a.add_patch(Polygon(V, closed=True, fc=CRIT, alpha=0.22, ec=CRIT, lw=1.5, label="$v_2=0$: $G^*=\\{01\\}$, a scale appears (dim 2)"))
a.autoscale_view(); a.set_xlabel("$y=\\log t$"); a.set_ylabel("$x=\\log\\lambda$"); a.legend(frameon=False, fontsize=7, loc="lower left")
a.set_title("(d) the zero stratum: a coincidence makes\nthe nonzero graph bipartite", fontsize=9)
plt.tight_layout(); save(fig, "fig_zoo"); print("ok")
