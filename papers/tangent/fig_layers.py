"""Figure for the section "Layers" of "A tangent structure for injected data": a picture for each layer.
(a) strata of Delta_3 and the layer a direction enters; (b) the merge downstairs: families entering one product law keep different rhat;
(c) the same upstairs, after blowing up: rhat is the coordinate r on the exceptional line; (d) the triangle's explanation segment
against log lam: a corner where the multi-map's domain ends; (e) a branch ending at a positivity wall; (f) the pipeline's domain along a
mixing chord: a hole."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
HERE = os.path.dirname(os.path.abspath(__file__)); ENG = os.path.join(HERE, "..", "engine"); sys.path.insert(0, ENG)
from pce import layers as LY, injection as IJ
D = json.load(open(os.path.join(ENG, "..", "lab", "layers.json")))
BLUE, ORANGE, VIOLET, AQUA, GREY, CRIT = "#2a78d6", "#eb6834", "#6250d6", "#1baf7a", "#8c8c8c", "#d03b3b"; INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
V3 = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]]); bary = lambda p: np.asarray(p) @ V3
fig, ax = plt.subplots(2, 3, figsize=(14.5, 8.4))
# (a)
a = ax[0, 0]; a.set_aspect("equal"); a.axis("off"); a.add_patch(Polygon(V3, closed=True, fc="#eef4fc", ec="none"))
for i, j in ((0, 1), (1, 2), (0, 2)): a.plot(*np.array([V3[i], V3[j]]).T, color=VIOLET, lw=3)
a.scatter(*V3.T, color=ORANGE, s=40, zorder=5)
x = np.array([.5, .5, 0]); p = bary(x); a.scatter(*p, color=INK, s=25, zorder=6)
for u, col, lab in ((np.array([.25, -.25, 0]), VIOLET, "$u_1$ stays on the edge:\nlayer = edge, $w_3\\geq0$"), (np.array([-.2, 0, .2]), BLUE, "$u_2$ enters:\nlayer = interior, $w$ free")):
    a.add_patch(FancyArrowPatch(p, bary(x + u), arrowstyle="-|>", mutation_scale=12, color=col, lw=1.8))
    a.text(*(bary(x + u) + (np.array([-0.38, 0.06]) if col == VIOLET else np.array([-0.05, 0.08]))), lab, fontsize=7.3, color=col)
a.text(0.5, -0.27, "strata: 3 vertices (orange), 3 open edges (violet), the open interior (blue).\nEach open stratum is a tangent category; the corner fibre (29 points)\n= 15 (edge layer) + 14 (interior layer), not 5 x 5 = 25", ha="center", fontsize=7.3)
a.set_xlim(-0.1, 1.1); a.set_ylim(-0.42, 0.95); a.set_title("(a) layer 1: strata, and the layer a direction enters", fontsize=9)
# (b)
a = ax[0, 1]; mer = D["merge"]; th = np.array([0.6, -0.5, 0.45]); th /= np.linalg.norm(th)
from pce.lab import law, moments
mu = np.array([.32, .26, .28])
for wv, col in ((0.6, BLUE), (0.3, ORANGE), (0.45, AQUA), (0.75, VIOLET)):
    t = np.sqrt((1 - wv) / wv); pts = []
    for rho in np.linspace(0.25, 0.0, 30):
        P = law(wv, mu + rho * th * t, mu - rho * th / t); _, M, mS = moments(P); pts.append((M[0, 1], mS((0, 1, 2))))
    pts = np.array(pts); a.plot(pts[:, 0], pts[:, 1], color=col, lw=2, label=f"$w={wv}$: $\\hat r={t - 1 / t:+.3f}$ all the way in")
a.scatter([0], [0], color=INK, s=30, zorder=5); a.text(0.0005, -0.0012, "the merge\n(a product law)", fontsize=7.5)
a.set_xlabel("pair covariance $M_{01}$"); a.set_ylabel("triple moment $m_{012}$"); a.legend(frameon=False, fontsize=7, loc="upper left")
a.set_title("(b) layer 2, downstairs: families entering one law\nkeep different $\\hat r$; no limit at the merge", fontsize=9)
# (c)
a = ax[0, 2]
for wv, col in ((0.6, BLUE), (0.3, ORANGE), (0.45, AQUA), (0.75, VIOLET)):
    t = np.sqrt((1 - wv) / wv); a.plot([0, 0.25], [t - 1 / t] * 2, color=col, lw=2)
a.axvline(0, color=CRIT, lw=3); a.text(0.008, -1.4, "exceptional line $\\rho=0$:\none point per value of $r$", color=CRIT, fontsize=7.5)
a.set_xlabel("$\\rho=|v|$ (distance to the merge)"); a.set_ylabel("$r=t-1/t$ (= $\\hat r$)"); a.set_ylim(-1.7, 1.4)
a.set_title("(c) layer 2, upstairs: blown up, $\\hat r$ is a smooth\ncoordinate; the singular point became a line", fontsize=9)
# (d)
a = ax[1, 0]; f = np.array(D["fold"]["curve"]); ll = np.log(f[:, 0])
a.fill_between(ll, f[:, 1], f[:, 2], color=BLUE, alpha=0.3); a.plot(ll, f[:, 1], color=BLUE, lw=1.5); a.plot(ll, f[:, 2], color=BLUE, lw=1.5)
ls = np.log(D["fold"]["lam_star"]); a.axvline(ls, color=CRIT, lw=1, ls="--"); a.scatter([ls], [D["fold"]["y_star"]], color=CRIT, zorder=5, s=25)
a.text(ls + 0.04, 0.12, f"domain ends at $\\lambda^*={D['fold']['lam_star']:.4f}$:\nno explanation to the left", color=CRIT, fontsize=7.5, ha="left")
a.set_xlabel("$\\log\\lambda$ ($c_{01}$ scaled by $\\lambda$)"); a.set_ylabel("explanations $y=\\log t$")
a.set_title("(d) layer 3: the multi-map data $\\to$ explanations\nhas a corner where its domain ends", fontsize=9)
# (e)
a = ax[1, 1]; br = D["branch"]; wv = np.array(br["w"])
rr = np.array(br["radius_rel"]); a.plot(wv, np.where(rr > -0.5, rr, np.nan), color=ORANGE, lw=2, label="relative-sign sheet (labels swapped on {2,3})")
a.plot(wv, br["radius_true"], color=BLUE, lw=2, label="true sheet"); a.axvline(br["w_star"], color=CRIT, lw=1, ls="--")
a.text(br["w_star"] + 0.005, 0.05, f"branch dies at\n$w^*={br['w_star']:.4f}$", color=CRIT, fontsize=7.5)
a.set_xlabel("weight $w$ of the truth (two disjoint edges)"); a.set_ylabel("size of the sheet (Chebyshev radius)"); a.legend(frameon=False, fontsize=7.2, loc="upper left")
a.set_title("(e) layer 3: a branch of the multi-map\nends at a positivity wall", fontsize=9)
# (f)
a = ax[1, 2]; h = D["hole"]; P, Q = np.array(h["P"]), np.array(h["Q"]); es = np.linspace(0, 1, 401)
d = np.array([LY.pipeline_domain(IJ.as_law((1 - e) * P + e * Q, 3)) for e in es])
a.plot(es, d * 1e4, color=VIOLET, lw=2); a.axhline(0, color=INK2, lw=0.8); a.set_yscale("symlog", linthresh=0.01); a.axvspan(h["shares"][0], h["shares"][1], color=CRIT, alpha=0.15)
a.text(np.mean(h["shares"]), 2.0, "hole: the pipeline\nis undefined here", ha="center", color=CRIT, fontsize=7.5)
a.set_xlabel("share $\\varepsilon$ of $Q$ mixed into $P$"); a.set_ylabel("$M_{01}M_{02}M_{12}$ ($\\times10^{-4}$, symlog)")
a.set_title("(f) restriction criteria do not mix: both laws are\nin the domain, part of the chord is not", fontsize=9)
plt.tight_layout(); fig.savefig(os.path.join(HERE, "fig_layers.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_layers.png"), dpi=150, bbox_inches="tight"); print("ok")
