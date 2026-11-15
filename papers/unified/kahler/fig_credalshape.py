"""Study 31 (sec:credalshape; engine/study31_shape/credalshape.json from check_credalshape.py). One three-context table (median band
width of P(A = 0 | B in {0,1}, C = 0) in the P332 population). A: the credal fibre K, a 4-polytope in interaction coordinates, drawn in
the coordinates (q0, q1, s), where q = (P(A = a | block))_a is affine on K and s is the leading remaining principal direction; the
vertices realising the band of q0 are marked. B: the outcome region in the simplex, before evidence, after evidence and with margins
known within 1/100, with the product of the three coordinate bands (dashed) and the band of q0. C: the same regions under the
Fisher-Rao map q -> sqrt(q) on the positive octant of the unit sphere, with an approximate 95% sampling cap for m = 400 conditional
observations (asymptotic). D: active facets: the cells that are zero on the preimage face of each polygon vertex, before and after
evidence (dark = zero; left square C = 0, right square C = 1; rows A, columns B)."""
import os, json, itertools
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
HERE = os.path.dirname(os.path.abspath(__file__)); D = json.load(open(os.path.join(HERE, "..", "engine", "study31_shape", "credalshape.json")))["p50"]
INK, MUTED = "#222222", "#666666"; TEAL, BLUE, GREY, RED = "#2b8a7e", "#1f5f9e", "#9e9e9e", "#b3261e"
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
cells = [(a, b, c) for a in range(3) for b in range(3) for c in range(2)]
VA = np.array(D["K_vertices"]); Q0 = np.array(D["Q0"]); edges = D["K_edges"]
def tern(q):
    q = np.atleast_2d(q); return np.column_stack([q[:, 1] + q[:, 2] / 2, q[:, 2] * np.sqrt(3) / 2])
def ordered(P):
    P = np.array(P); h = ConvexHull(tern(P)); return P[h.vertices]
fig = plt.figure(figsize=(11.5, 9.2)); gs = fig.add_gridspec(2, 2, wspace=0.18, hspace=0.25)
# A
ax = fig.add_subplot(gs[0, 0], projection="3d")
X = np.column_stack([Q0[:, 0], Q0[:, 1]]); Z = VA - VA.mean(0); B2 = np.column_stack([np.ones(len(VA)), X]); res = Z - B2 @ np.linalg.lstsq(B2, Z, rcond=None)[0]
u, s, vt = np.linalg.svd(res, full_matrices=False); third = res @ vt[0]
for i, j in edges: ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], [third[i], third[j]], color=GREY, lw=0.8)
lo, hi = np.argmin(Q0[:, 0]), np.argmax(Q0[:, 0])
ax.scatter(X[:, 0], X[:, 1], third, color=INK, s=9)
for k, col, lab in ((lo, BLUE, "lower end of the band"), (hi, RED, "upper end of the band")):
    on = np.where(np.abs(Q0[:, 0] - Q0[k, 0]) < 1e-12)[0]; ax.scatter(X[on, 0], X[on, 1], third[on], color=col, s=30, label=lab)
ax.set_xlabel("q0", labelpad=-4); ax.set_ylabel("q1", labelpad=-4); ax.set_zlabel("s", labelpad=-6); ax.tick_params(labelsize=5.5, pad=-2)
ax.legend(frameon=False, fontsize=6.3, loc="upper left"); ax.view_init(22, -58)
ax.set_title(f"A  The credal fibre: a 4-polytope ({D['vertices_K']} vertices)", loc="left", fontsize=8.4, fontweight="bold", color=INK)
# B
ax = fig.add_subplot(gs[0, 1]); allp = np.vstack([D["polygonT"], D["polygon1"], D["polygon0"]]); T = tern(allp); pad = 0.035
for P_, fc, ec, ls, lab, z in ((D["polygonT"], "#e6e6e6", GREY, "--", "margins within 1/100", 1), (D["polygon0"], "#cfe8e4", TEAL, "-", "before evidence", 2), (D["polygon1"], "none", BLUE, "-", "after evidence", 3)):
    ax.add_patch(Polygon(tern(ordered(P_)), closed=True, fc=fc, ec=ec, ls=ls, lw=1.3, label=lab, zorder=z))
def band_product(P_):
    P_ = np.array(P_); cb = [[P_[:, a].min(), P_[:, a].max()] for a in range(3)]; pts = []
    for a, b_ in itertools.combinations(range(3), 2):
        for va in cb[a]:
            for vb in cb[b_]:
                q = np.zeros(3); q[a] = va; q[b_] = vb; q[3 - a - b_] = 1 - va - vb
                if all(cb[k][0] - 1e-12 <= q[k] <= cb[k][1] + 1e-12 for k in range(3)): pts.append(q)
    return ordered(pts)
ax.add_patch(Polygon(tern(band_product(D["polygon1"])), closed=True, fc="none", ec=RED, ls=(0, (3, 2)), lw=1.0, label="product of the bands, after evidence", zorder=4))
for v, txt in ((D["band0"][0], "q0 = lower end"), (D["band0"][1], "q0 = upper end")):
    seg = tern(np.array([[v, 1 - v, 0], [v, 0, 1 - v]])); ax.plot(seg[:, 0], seg[:, 1], color=MUTED, lw=0.6, ls=":")
ax.set_xlim(T[:, 0].min() - pad, T[:, 0].max() + pad); ax.set_ylim(T[:, 1].min() - pad, T[:, 1].max() + pad); ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([]); ax.legend(frameon=False, fontsize=6.3, loc="lower left")
ins = ax.inset_axes([0.73, 0.02, 0.26, 0.28]); tri = tern(np.eye(3)); ins.add_patch(Polygon(tri, closed=True, fc="none", ec=MUTED, lw=0.6))
ins.add_patch(Polygon(tern(ordered(D["polygonT"])), closed=True, fc=GREY, ec="none")); ins.set_xlim(-0.05, 1.05); ins.set_ylim(-0.05, 0.92); ins.axis("off"); ins.set_aspect("equal")
for k, lab in enumerate(("A=0", "A=1", "A=2")): ins.text(*tri[k], lab, fontsize=5.5, color=MUTED, ha="center", va="top" if k < 2 else "bottom")
ax.text(0.99, 0.99, "before evidence the region equals\nthe product of its bands (band completeness)", transform=ax.transAxes, ha="right", va="top", fontsize=6.2, color=TEAL)
ax.set_title("B  The outcome region in the simplex (zoomed)", loc="left", fontsize=8.4, fontweight="bold", color=INK)
# C
ax = fig.add_subplot(gs[1, 0], projection="3d")
uu, vv = np.meshgrid(np.linspace(0, np.pi / 2, 25), np.linspace(0, np.pi / 2, 25)); ax.plot_wireframe(np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv), color="#dddddd", lw=0.4)
def arcs(P_, col, ls, lab):
    P_ = ordered(P_); first = True
    for i in range(len(P_)):
        a, b_ = P_[i], P_[(i + 1) % len(P_)]; t = np.linspace(0, 1, 40)[:, None]; seg = np.sqrt((1 - t) * a + t * b_)
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=col, ls=ls, lw=1.3, label=lab if first else None); first = False
arcs(D["polygonT"], GREY, "--", "margins within 1/100"); arcs(D["polygon0"], TEAL, "-", "before evidence"); arcs(D["polygon1"], BLUE, "-", "after evidence")
cen = np.sqrt(np.array(D["polygon0"]).mean(0)); cen /= np.linalg.norm(cen); rho = 1.96 / (2 * np.sqrt(400))
e1 = np.cross(cen, [0, 0, 1]); e1 /= np.linalg.norm(e1); e2 = np.cross(cen, e1); ph = np.linspace(0, 2 * np.pi, 100)
cap = np.array([np.cos(rho) * cen + np.sin(rho) * (np.cos(p) * e1 + np.sin(p) * e2) for p in ph]); ax.plot(cap[:, 0], cap[:, 1], cap[:, 2], color=RED, lw=1.0, ls=":", label="sampling cap, m = 400 (approx.)")
ax.set_xlabel("sqrt q0", labelpad=-4); ax.set_ylabel("sqrt q1", labelpad=-4); ax.set_zlabel("sqrt q2", labelpad=-6); ax.tick_params(labelsize=5.5, pad=-2)
c0 = cen; ax.set_xlim(c0[0] - 0.2, c0[0] + 0.2); ax.set_ylim(c0[1] - 0.2, c0[1] + 0.2); ax.set_zlim(c0[2] - 0.2, c0[2] + 0.2); ax.view_init(30, 40)
ax.legend(frameon=False, fontsize=6.2, loc="upper left")
ax.set_title("C  The Fisher-Rao image on the sphere", loc="left", fontsize=8.4, fontweight="bold", color=INK)
# D
sub = gs[1, 1].subgridspec(4, 8, hspace=0.15, wspace=0.25, height_ratios=[1, 1, 2.2, 1.2]); fig.text(0.53, 0.465, "D  Active facets at each polygon vertex", fontsize=8.4, fontweight="bold", color=INK)
def glyph(axg, zc, title, col):
    G = np.zeros((3, 7)); G[:] = np.nan
    for k in range(18):
        a, b_, c_ = cells[k]; G[a, b_ + 4 * c_] = 1.0 if k in zc else 0.15
    axg.imshow(G, cmap="Greys", vmin=0, vmax=1); axg.set_xticks([]); axg.set_yticks([])
    for s_ in axg.spines.values(): s_.set_color(col)
    axg.set_title(title, fontsize=5.8, color=col, pad=2)
for row, (key, col, lab) in enumerate((("active0", TEAL, "before"), ("active1", BLUE, "after"))):
    for k, v in enumerate(D[key][:8]):
        axg = fig.add_subplot(sub[row, k]); glyph(axg, set(v["zero_cells"]), f"{lab} {k + 1}", col)
axn = fig.add_subplot(sub[2:, :]); axn.axis("off")
axn.text(0, 0.95, "Each glyph: the 18 cells, left square C = 0, right square C = 1, rows A, columns B; dark = zero cell (active facet).\n\n" f"Before evidence the region has {len(D['active0'])} vertices; evidence (cell weights) gives {len(D['active1'])}.\n"
         "At the level of the six block cells evidence is a projective map, an isometry of the Hilbert metric,\n"
         "so the block-level shape is unchanged; summing over B to get the outcome q adds vertices and changes it.", fontsize=6.6, color=INK, va="top")
fig.savefig(os.path.join(HERE, "fig_credalshape.png"), dpi=220, bbox_inches="tight"); print("ok")
