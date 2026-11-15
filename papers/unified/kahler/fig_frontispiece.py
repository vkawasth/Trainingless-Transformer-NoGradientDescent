"""Frontispiece: one example followed through the whole geometry (fig:frontispiece; engine/study32_frontispiece/frontispiece.json
from check_frontispiece.py, UG1-UG5). E1: f = (30, 20, 10)/60, g = (24, 21, 15)/60.
A base of margins: the slice g = g(E1) of the chamber complex, coloured by the number of vertices of Cpl(f, g); walls f_i = subset sums of g.
B the credal fibre Cpl(f, g), a simple 4-polytope with 18 vertices (3D projection), the marked vertex with its 4 edges, the path of the
  positive critical point x(L) and its tropical limit, the leximin point.
C dual bases at the marked vertex: positive cells = spanning tree of K_{3,3}; zero cells = cographic basis = the 4 edges at the vertex.
D the leximin filtration of Theorem 196: blocked cells by level, values h_j, rank increments r_j.
E the spectral side of Theorem 196: scale valuations -log|lambda|/L of Q/W at the positive critical point against 1/L, converging to
  h_j - h_1 with multiplicity r_j.
F the decision: conditional law of the column given rows {0,1}; region before evidence (equal to the product of its bands), after
  evidence (blue) and the product of its bands (dashed); inset: the Fisher-Rao image on the sphere with an asymptotic sampling cap."""
import os, json, itertools
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
HERE = os.path.dirname(os.path.abspath(__file__)); D = json.load(open(os.path.join(HERE, "..", "engine", "study32_frontispiece", "frontispiece.json")))
INK, MUTED, TEAL, CORAL, GOLD, BLUE, GREY = "#222222", "#666666", "#2b8a7e", "#c4473a", "#c9962b", "#1f5f9e", "#9e9e9e"
plt.rcParams.update({"font.size": 7.6, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "mathtext.fontset": "cm"})
CELLS = [(i, j) for i in range(3) for j in range(3)]
def tern(Q): Q = np.atleast_2d(Q); return np.column_stack([Q[:, 1] + Q[:, 2] / 2, Q[:, 2] * np.sqrt(3) / 2])
def title(ax, s, sub): ax.set_title(s, loc="left", fontsize=8.6, fontweight="bold", color=INK); ax.text(0, 1.0, sub, transform=ax.transAxes, fontsize=7, color=MUTED, va="bottom") if False else None
fig = plt.figure(figsize=(13.5, 9.2)); gs = fig.add_gridspec(2, 3, wspace=0.42, hspace=0.42)
# A ----------------------------------------------------------------
ax = fig.add_subplot(gs[0, 0]); G = np.array(D["grid"]); P = tern(G[:, :3]); cnt = G[:, 3]
sc = ax.scatter(P[:, 0], P[:, 1], c=cnt, cmap="GnBu", s=3.2, marker="s", lw=0, vmin=cnt.min() - 4, vmax=cnt.max())
for s in D["wall_levels"]:
    for i in range(3):
        a = np.zeros(3); b = np.zeros(3); a[i] = b[i] = s; o = [k for k in range(3) if k != i]; a[o[0]] = 1 - s; b[o[1]] = 1 - s
        ax.plot(*tern(np.array([a, b])).T, color="white", lw=0.9)
tri = tern(np.eye(3)); ax.add_patch(Polygon(tri, closed=True, fc="none", ec=MUTED, lw=0.8))
fE = np.array([30, 20, 10]) / 60; ax.scatter(*tern(fE).T, s=40, color=GOLD, ec=INK, zorder=5); ax.annotate("E1", tern(fE)[0], xytext=(6, 4), textcoords="offset points", fontsize=7.5, color=INK)
for k, lab in enumerate(("$f_1=1$", "$f_2=1$", "$f_3=1$")): ax.text(*tri[k], lab, fontsize=7, color=MUTED, ha=["right", "left", "center"][k], va="top" if k < 2 else "bottom")
cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.01); cb.set_label("vertices of Cpl(f, g)", fontsize=6.5); cb.ax.tick_params(labelsize=6)
ax.text(0.5, -0.1, r"walls: $\sum_{i\in I} f_i=\sum_{j\in J} g_j$   (slice $g=g_{E1}$)", transform=ax.transAxes, ha="center", fontsize=7.4)
ax.set_aspect("equal"); ax.axis("off"); ax.set_title("A  Base: the chamber complex of the margins", loc="left", fontsize=8.6, fontweight="bold", color=INK)
# B ----------------------------------------------------------------
ax = fig.add_subplot(gs[0, 1], projection="3d"); V = np.array(D["vertices"]); path = np.array(D["path"]); lex = np.array(D["leximin"])
mu = V.mean(0); _, _, vt = np.linalg.svd(V - mu); Pj = lambda X: (np.atleast_2d(X) - mu) @ vt[:3].T
PV = Pj(V); m = D["mark"]
for i, nb in enumerate(D["adj"]):
    for j in nb:
        if j > i: ax.plot(*np.array([PV[i], PV[j]]).T, color=GREY, lw=0.7)
for j in D["adj"][m]:
    ax.plot(*np.array([PV[m], PV[j]]).T, color=CORAL, lw=2.0)
    ent = list(set(D["bases"][j]) - set(D["bases"][m]))[0]; mid = PV[m] + 0.8 * (PV[j] - PV[m])
    ax.text(*mid, f"{CELLS[ent][0] + 1}{CELLS[ent][1] + 1}", color=CORAL, fontsize=6.5)
ax.scatter(*PV.T, color=INK, s=7); ax.scatter(*PV[m], color=CORAL, s=40, ec=INK)
PP = Pj(path); ax.plot(*PP.T, color=GOLD, lw=1.6, ls="-"); ax.scatter(*Pj(lex).T, color=GOLD, s=60, ec=INK, marker="*", zorder=6)

ax.tick_params(labelsize=0, length=0); ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([]); ax.view_init(18, 35)
ax.text2D(0.0, -0.02, "18 vertices = feasible spanning trees; every vertex has 4 edges (simple).\nGold: positive critical point $x(L)$ tending to the leximin point (star).",
          transform=ax.transAxes, fontsize=6.8, color=INK, va="top")
ax.set_title("B  Fibre: the credal polytope Cpl(f, g), dim 4", loc="left", fontsize=8.6, fontweight="bold", color=INK)
# C ----------------------------------------------------------------
ax = fig.add_subplot(gs[0, 2]); ax.axis("off"); pos = [tuple(c) for c in D["pos"]]; xm = D["xm"]
for k, (i, j) in enumerate(CELLS):
    on = (i, j) in pos; ax.add_patch(plt.Rectangle((j * 0.55, (2 - i) * 0.55), 0.55, 0.55, fc="#cfe8e4" if on else "white", ec=MUTED))
    ax.text(j * 0.55 + 0.275, (2 - i) * 0.55 + 0.275, xm[k] if on else "0", ha="center", va="center", fontsize=6.8, color=INK if on else CORAL)
R = {i: (2.3, 1.375 - i * 0.55) for i in range(3)}; Cc = {j: (3.5, 1.375 - j * 0.55) for j in range(3)}
for (i, j) in CELLS:
    on = (i, j) in pos; ax.plot([R[i][0], Cc[j][0]], [R[i][1], Cc[j][1]], color=TEAL if on else CORAL, lw=2 if on else 0.9, ls="-" if on else (0, (3, 2)))
for i in range(3): ax.scatter(*R[i], s=60, color="white", ec=INK, zorder=3); ax.scatter(*Cc[i], s=60, color="white", ec=INK, zorder=3)
ax.text(2.3, 1.75, "rows", ha="center", fontsize=6.5, color=MUTED); ax.text(3.5, 1.75, "columns", ha="center", fontsize=6.5, color=MUTED)
ax.text(0, -0.25, r"positive cells $T$: spanning tree of $K_{3,3}$ (graphic basis), $|T|=r+c-1=5$" "\n"
        r"zero cells $Z=E\setminus T$: cographic basis, $|Z|=(r-1)(c-1)=4$" "\n"
        r"$Z$ = tight facets $\{x_e=0\}$ = the 4 coral edges in B; normals $v_e$ span the cone", fontsize=6.9, color=INK, va="top")
ax.set_xlim(-0.1, 3.9); ax.set_ylim(-1.2, 1.9); ax.set_aspect("equal")
ax.set_title("C  Dual bases at the marked vertex", loc="left", fontsize=8.6, fontweight="bold", color=INK)
# D ----------------------------------------------------------------
ax = fig.add_subplot(gs[1, 0]); ax.axis("off"); levcol = ["#1f5f9e", "#2b8a7e", "#7fb8a8", "#d9c38a", "#e9e2cf"]; lvl = {}
for j, (h, cs) in enumerate(D["levels"]):
    for c in cs: lvl[tuple(c)] = j
for (i, jj) in CELLS:
    j = lvl[(i, jj)]; ax.add_patch(plt.Rectangle((jj * 0.6, (2 - i) * 0.6), 0.6, 0.6, fc=levcol[j], ec="white", lw=1.5))
    ax.text(jj * 0.6 + 0.3, (2 - i) * 0.6 + 0.3, f"$S_{j + 1}$", ha="center", va="center", fontsize=8, color="white" if j < 2 else INK)
y0 = 1.75
ax.text(2.1, y0 + 0.05, "level", fontsize=7, color=INK, va="top"); ax.text(2.62, y0 + 0.05, "$h_j$", fontsize=7.5, color=INK, va="top"); ax.text(3.3, y0 + 0.05, "$r_j$", fontsize=7.5, color=INK, va="top")
for j, ((h, cs), rr) in enumerate(zip(D["levels"], D["r"])):
    ax.add_patch(plt.Rectangle((2.1, y0 - 0.3 - j * 0.3), 0.14, 0.2, fc=levcol[j], ec="none")); ax.text(2.3, y0 - 0.2 - j * 0.3, f"$S_{j + 1}$", fontsize=7.2, va="center"); ax.text(2.62, y0 - 0.2 - j * 0.3, f"${h}$", fontsize=7.2, va="center"); ax.text(3.32, y0 - 0.2 - j * 0.3, f"{rr}", fontsize=7.2, va="center")
ax.text(0, -0.25, r"leximin recursion: raise the smallest free cell until blocked;" "\n" r"$S_j$ = the cells blocked at height $h_j$" "\n"
        r"$r_j$ = rank added by $\{v_k: k\in S_j\}$ to the earlier normals" "\n"
        r"here $r=(2,1,1,0,0)$: three levels remove all 4 dimensions", fontsize=6.9, color=INK, va="top")
ax.set_xlim(-0.1, 3.6); ax.set_ylim(-1.15, 1.95); ax.set_aspect("equal")
ax.set_title("D  Theorem 196, combinatorial side", loc="left", fontsize=8.6, fontweight="bold", color=INK)
# E ----------------------------------------------------------------
ax = fig.add_subplot(gs[1, 1]); Ls = [float(L) for L in D["Ls"]]; SV = np.array([D["scale_values"][str(L) if str(L) in D["scale_values"] else str(int(L))] for L in D["Ls"]])
x = 1 / np.array(Ls); keep = np.array(Ls) >= 15
for i in range(4): ax.plot(x[keep], SV[keep, i], marker="o", ms=3, color=[BLUE, BLUE, TEAL, GOLD][i], lw=1.2)
for t, lab in ((0.0, r"$0$  ($\times 2$)"), (D["target"][2], r"$h_2-h_1=\frac{3}{72}$  ($\times 1$)"), (D["target"][3], r"$h_3-h_1=\frac{9}{144}$  ($\times 1$)")):
    ax.axhline(t, color=MUTED, lw=0.6, ls="--"); ax.text(0.0712, t, lab, fontsize=7, color=INK, va="center", clip_on=False)
ax.set_xlim(-0.002, 0.07); ax.set_xlabel("$1/L$"); ax.set_ylabel(r"scale valuation $-\log|\lambda_i|/L$")
ax.text(0.03, 0.97, r"eigenvalues of $Q/W$ at the positive critical point:" "\n" r"$\lambda_i\asymp e^{-L(h_j-h_1)}$ with multiplicity $r_j$ as $L\to\infty$", transform=ax.transAxes, ha="left", va="top", fontsize=6.9, color=INK)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_title("E  Theorem 196, spectral side", loc="left", fontsize=8.6, fontweight="bold", color=INK)
# F ----------------------------------------------------------------
ax = fig.add_subplot(gs[1, 2]); Q0 = np.array(D["Q0"]); Q1 = np.array(D["Q1"])
def hullpts(Q): h = ConvexHull(tern(Q)); return tern(Q)[h.vertices]
def product(bands):
    pts = []
    for a, b in itertools.combinations(range(3), 2):
        for va in bands[a]:
            for vb in bands[b]:
                q = np.zeros(3); q[a] = va; q[b] = vb; q[3 - a - b] = 1 - va - vb
                if all(bands[k][0] - 1e-12 <= q[k] <= bands[k][1] + 1e-12 for k in range(3)): pts.append(q)
    return hullpts(np.array(pts))
ax.add_patch(Polygon(hullpts(Q0), closed=True, fc="#cfe8e4", ec=TEAL, lw=1.4, label="before evidence (= product of its bands)"))
ax.add_patch(Polygon(hullpts(Q1), closed=True, fc="none", ec=BLUE, lw=1.4, label=r"after evidence $\varphi_L$"))
ax.add_patch(Polygon(product(D["bands1"]), closed=True, fc="none", ec=CORAL, lw=1.0, ls=(0, (3, 2)), label=f"product of its bands ({D['ratio_after']:.0%} filled)"))
allp = np.vstack([tern(Q0), tern(Q1), product(D["bands1"])]); pad = 0.03
w_ = allp[:, 0].max() - allp[:, 0].min(); ax.set_xlim(allp[:, 0].min() - pad, allp[:, 0].max() + 0.75 * w_); ax.set_ylim(allp[:, 1].min() - pad, allp[:, 1].max() + pad); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.legend(frameon=False, fontsize=6.4, loc="upper left", bbox_to_anchor=(0, -0.02))
b0 = D["bands0"][0]; ax.text(0.0, 1.0, r"$q=\pi(\mathcal{M})$: column given rows $\{1,2\}$;  $\varphi_L(p)=p\circ L/\langle p,L\rangle$", transform=ax.transAxes, fontsize=6.7, color=INK, va="bottom")
ins = ax.inset_axes([0.58, 0.3, 0.44, 0.6], projection="3d"); uu, vv = np.meshgrid(np.linspace(0, np.pi / 2, 16), np.linspace(0, np.pi / 2, 16))
ins.plot_wireframe(np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv), color="#dddddd", lw=0.3)
for Q, col in ((Q0, TEAL), (Q1, BLUE)):
    h = ConvexHull(tern(Q)); V2 = Q[h.vertices]
    for i in range(len(V2)):
        t = np.linspace(0, 1, 30)[:, None]; seg = np.sqrt((1 - t) * V2[i] + t * V2[(i + 1) % len(V2)]); ins.plot(*seg.T, color=col, lw=1)
cen = np.sqrt(Q0.mean(0)); cen /= np.linalg.norm(cen); rho = 1.96 / (2 * np.sqrt(400)); e1 = np.cross(cen, [0, 0, 1]); e1 /= np.linalg.norm(e1); e2 = np.cross(cen, e1)
cap = np.array([np.cos(rho) * cen + np.sin(rho) * (np.cos(p) * e1 + np.sin(p) * e2) for p in np.linspace(0, 2 * np.pi, 60)]); ins.plot(*cap.T, color=CORAL, lw=0.8, ls=":")
ins.set_axis_off(); ins.view_init(30, 40); ins.set_title(r"$q\mapsto\sqrt{q}$; cap $\sim m^{-1/2}$", fontsize=6.3, color=MUTED, pad=0)
ax.set_title("F  Decision: the conditional region and its bands", loc="left", fontsize=8.6, fontweight="bold", color=INK, pad=14)
fig.savefig(os.path.join(HERE, "fig_frontispiece.png"), dpi=220, bbox_inches="tight"); print("ok")
