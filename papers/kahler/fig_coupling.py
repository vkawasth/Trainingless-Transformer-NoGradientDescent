"""Figures for section 11 of the companion note: one coupling polytope through every layer. Running example f = (2/5, 3/5),
g = (3/10, 43/100, 27/100) (the architecture note's example). Four figures:
  fig_cpl_glance.pdf   the table and its fibre: the leximin coupling, the tables as lattice points, the three distinguished couplings
  fig_cpl_tierA.pdf    Tier A: the moment map is the expected table of Fisher's noncentral hypergeometric family
  fig_cpl_thimble.pdf  the positive thimble, tropically: from the uniform law on couplings to the leximin coupling
  fig_cpl_walls.pdf    margin space: Kahler-chamber walls f_i = g_j against leximin walls f_i/3 = g_j/2, and the one-count move"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from pce import couplingfibre as CF, kahlerbridge as KB
BLUE, ORANGE, AQUA, VIOLET, ROSE, INK, INK2, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#c2417a", "#22211f", "#52514e", "#9a9a9a"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
f, g = CF.EX_F, CF.EX_G; P, A = CF.polygon(f, g); PF = np.array([[float(x), float(y)] for x, y in P])
V, L = CF.cell_functions(f[0], g[0], g[1]); vk = [V[k] for k in (0, 1, 2, 3, 5)]; lk = [float(L[k]) for k in (0, 1, 2, 3, 5)]
PROD = (0.4 * 0.3, 0.4 * 0.43); LEX = (2 / 15, 2 / 15); SIG1 = (0.13, 0.13); SIG2 = (0.2, 0.1)
def pent(ax, fc="#f3f3f3", lw=1.2):
    ax.add_patch(Polygon(PF, closed=True, fc=fc, ec=INK2, lw=lw, zorder=1))
def marks(ax, legend=True, ms=8):
    ax.plot(*PROD, "s", color=AQUA, ms=ms, mec="white", zorder=5, label="product coupling $f\\otimes g$ (Tier A, odds ratios 1)")
    ax.plot(*LEX, "o", mfc="none", mec=BLUE, mew=2.2, ms=ms + 4, zorder=6, label="leximin coupling (Tier B: the positive critical point)")
    ax.plot(*SIG1, "v", color=VIOLET, ms=ms - 2, mec="white", zorder=7, label="signed fibre, $y_1,y_2<0$ (just below the leximin)")
    ax.plot(*SIG2, "^", color=ROSE, ms=ms - 1, mec="white", zorder=5, label="signed fibre, $y_2<0$")
OUT = lambda name: os.path.join(HERE, name)

# ---------------------------------------------------------------- figure 1: the table and its fibre
fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.6), gridspec_kw=dict(width_ratios=[0.8, 1.2]))
a = ax[0]; a.set_axis_off(); a.set_xlim(0, 5.2); a.set_ylim(-1.0, 3.6)
lex = CF.leximin([0.4, 0.6], [0.3, 0.43, 0.27]); cw, ch = 1.15, 0.85
for i in range(2):
    for j in range(3):
        x0, y0 = 0.4 + j * cw, 2.0 - i * ch; hl = (i == 0)
        a.add_patch(Rectangle((x0, y0), cw, ch, fc="#dbe8f8" if hl else "white", ec=INK2, lw=1))
        a.text(x0 + cw / 2, y0 + ch / 2 + 0.12, f"cell {i + 1}{j + 1}", ha="center", fontsize=7.5, color=INK2)
        a.text(x0 + cw / 2, y0 + ch / 2 - 0.16, f"{lex[i, j]:.4f}", ha="center", fontsize=10, color=INK)
    a.text(0.4 + 3 * cw + 0.12, 2.0 - i * ch + ch / 2, f"$f_{i + 1}$ = {[0.4, 0.6][i]}", va="center", fontsize=9)
for j in range(3): a.text(0.4 + j * cw + cw / 2, 2.0 - ch - 0.25 - 0.05, f"$g_{j + 1}$ = {[0.3, 0.43, 0.27][j]}", ha="center", fontsize=9)
a.text(0.4, 3.25, "(a) The leximin coupling of $f$ and $g$", fontsize=10, color=INK)
a.text(0.4, 0.55, "No coupling can have its smallest cell above $\\min(f_i/3,\\ g_j/2)$.\nHere row 1 is the scarcest per cell ($f_1/3=2/15$), so the best\ncoupling spreads it evenly (shaded). This is the coupling that\nthe positive critical point of $W$ selects. Cell 23 is close behind\n($g_3/2=0.135$): a near tie that section 11.5 exploits.",
       fontsize=8, color=INK2, va="top")
b = ax[1]; pent(b)
T = CF.tables(100, f, g); S = np.array([s for s, _ in T]) / 100; b.plot(S[:, 0], S[:, 1], ".", color=GREY, ms=2.2, zorder=2, label="the 715 tables with these margins ($n=100$)")
marks(b)
lab = {"11": ((-0.018, 0.27), "$\\alpha_{11}=0$", 90), "12": ((0.215, -0.012), "$\\alpha_{12}=0$", 0), "13": ((0.16, 0.25), "$\\alpha_{13}=0$", -45),
       "21": ((0.31, 0.05), "$\\alpha_{21}=0$", 90), "23": ((0.055, 0.06), "$\\alpha_{23}=0$", -45)}
for k, (xy, t, rot) in lab.items(): b.text(*xy, t, fontsize=8, color=INK2, rotation=rot, ha="center", va="center")
b.set_xlim(-0.04, 0.36); b.set_ylim(-0.04, 0.44); b.set_aspect("equal"); b.set_xlabel("$x=\\alpha_{11}$"); b.set_ylabel("$y=\\alpha_{12}$")
b.legend(frameon=False, fontsize=7.2, loc="upper left", bbox_to_anchor=(1.0, 1.0)); b.set_title("(b) The fibre $\\mathrm{Cpl}(f,g)$: a pentagon whose sides are the cells that can vanish", fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT("fig_cpl_glance.pdf")); fig.savefig(OUT("fig_cpl_glance.png"), dpi=150); plt.close(fig)

# ---------------------------------------------------------------- figure 2: Tier A
Ssc = [s for s, _ in T]; lw_ = np.array([CF.hypergeom_weights(t) for _, t in T]); c = np.exp(lw_ - lw_.max())
fig, ax = plt.subplots(1, 2, figsize=(11.2, 6.0))
a = ax[0]; pent(a, fc="white")
for u1 in np.linspace(-3, 3, 13):
    pts = np.array([KB.moment(Ssc, c, (u1, u2)) for u2 in np.linspace(-6, 6, 241)]) / 100; a.plot(pts[:, 0], pts[:, 1], color=BLUE, lw=0.8, alpha=0.75)
for u2 in np.linspace(-3, 3, 13):
    pts = np.array([KB.moment(Ssc, c, (u1, u2)) for u1 in np.linspace(-6, 6, 241)]) / 100; a.plot(pts[:, 0], pts[:, 1], color=ORANGE, lw=0.8, alpha=0.75)
a.plot([], [], color=BLUE, label="$u_1$ fixed (log odds ratio of cells 11, 23 vs 13, 21)"); a.plot([], [], color=ORANGE, label="$u_2$ fixed (cells 12, 23 vs 13, 22)")
marks(a, ms=7); a.set_xlim(-0.04, 0.36); a.set_ylim(-0.04, 0.44); a.set_aspect("equal"); a.set_xlabel("expected $\\alpha_{11}$"); a.set_ylabel("expected $\\alpha_{12}$")
a.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
a.set_title("(a) The moment map is the expected table: odds-ratio\ncoordinates fill the pentagon; odds ratios 1 give $f\\otimes g$", fontsize=9.5)
b = ax[1]; pent(b, fc="white"); q0 = c / c.sum(); _, ulex, qlex = KB.legendre(Ssc, c, 100 * np.array(LEX))
Sp = np.array(Ssc) / 100
h1_ = b.scatter(Sp[:, 0], Sp[:, 1], s=0.8 + 900 * q0, color=AQUA, alpha=0.6, lw=0, label="odds ratios 1: the hypergeometric law (Fisher's exact test)")
h2_ = b.scatter(Sp[:, 0], Sp[:, 1], s=0.8 + 900 * qlex, color=BLUE, alpha=0.45, lw=0, label=f"odds ratios ({np.exp(ulex[0]):.2f}, {np.exp(ulex[1]):.2f}): mean at the leximin")
marks(b, legend=False, ms=7); b.set_xlim(-0.04, 0.36); b.set_ylim(-0.04, 0.44); b.set_aspect("equal"); b.set_xlabel("$\\alpha_{11}$"); b.set_ylabel("$\\alpha_{12}$")
b.legend(handles=[h1_, h2_], frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1); b.set_title("(b) Two laws on the 715 tables (dot area = probability)", fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT("fig_cpl_tierA.pdf")); fig.savefig(OUT("fig_cpl_tierA.png"), dpi=150); plt.close(fig)

# ---------------------------------------------------------------- figure 3: the thimble, tropically
hs = [0.0, 0.07, 0.12, 2 / 15]; fig, ax = plt.subplots(1, 4, figsize=(13.2, 3.7)); box = ((-0.02, 0.32), (-0.02, 0.42))
for a, h in zip(ax, hs):
    U1, U2, w = KB.tropical_grid(vk, lk, -400.0, h, box=box, n=501)
    a.imshow((w / w.max()).T, origin="lower", extent=(box[0][0], box[0][1], box[1][0], box[1][1]), cmap="Blues", vmin=0, vmax=1, aspect="equal")
    a.add_patch(Polygon(PF, closed=True, fc="none", ec=INK2, lw=1))
    if h < 2 / 15 - 1e-9:
        Ph = []
        import itertools
        for i, j in itertools.combinations(range(5), 2):
            M = np.array([vk[i], vk[j]], float)
            if abs(np.linalg.det(M)) < 1e-12: continue
            x = np.linalg.solve(M, np.array([lk[i] + h, lk[j] + h]))
            if np.all(np.array(vk, float) @ x - np.array(lk) >= h - 1e-12): Ph.append(x)
        Ph = np.array(Ph); cen = Ph.mean(0); Ph = Ph[np.argsort(np.arctan2(Ph[:, 1] - cen[1], Ph[:, 0] - cen[0]))]
        a.add_patch(Polygon(Ph, closed=True, fc="none", ec=ORANGE, lw=1.2, ls="--"))
    a.plot(*LEX, "o", color=BLUE, ms=5, mec="white"); a.plot(*PROD, "s", color=AQUA, ms=5, mec="white")
    if h >= 2 / 15 - 1e-9:
        a.plot(*LEX, "o", mfc="none", mec=ORANGE, mew=1.8, ms=14); a.annotate("all the mass", LEX, (0.17, 0.30), fontsize=8, color=INK2, arrowprops=dict(arrowstyle="->", color=INK2))
    if h == 0.0:
        a.annotate("product $f\\otimes g$", PROD, (0.17, 0.33), fontsize=7.5, color="white", arrowprops=dict(arrowstyle="-", color="white"))
    a.set_xlim(*box[0]); a.set_ylim(*box[1]); a.set_xticks([0, 0.1, 0.2, 0.3]); a.set_yticks([0, 0.2, 0.4])
    a.set_title({0.0: "$h=0$: uniform on all couplings\n(Duistermaat-Heckman)", 0.07: "$h=0.07$: uniform on $Q_h$\n(every cell $\\geq 0.07$)",
                 0.12: "$h=0.12$: uniform on $Q_h$\n(every cell $\\geq 0.12$)", 2 / 15: "$h=h^*=2/15$: all mass at\nthe leximin coupling"}[h], fontsize=9)
fig.suptitle("The positive thimble $e^{-W/\\hbar}\\,dy/y$ with $\\hbar=T^h$, read in $\\log y/\\log T$ ($\\log T=-400$): as $h$ grows it shrinks from all couplings to the leximin one",
             fontsize=9.5, y=1.0)
fig.tight_layout(); fig.savefig(OUT("fig_cpl_thimble.pdf"), bbox_inches="tight"); fig.savefig(OUT("fig_cpl_thimble.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- figure 4: two kinds of walls in margin space (slice g1 = 0.3)
g1 = 0.3; F1, G2 = np.meshgrid(np.linspace(0.005, 0.995, 600), np.linspace(0.005, 0.695, 420), indexing="ij")
R = np.stack([F1 / 3, (1 - F1) / 3, np.full_like(F1, g1 / 2), G2 / 2, (0.7 - G2) / 2]); cls = R.argmin(0)
cols = [BLUE, ORANGE, AQUA, VIOLET, ROSE]; names = ["row 1 binds: leading cells 11,12,13", "row 2 binds: 21,22,23", "column 1 binds: 11,21",
                                                    "column 2 binds: 12,22", "column 3 binds: 13,23"]
from matplotlib.colors import ListedColormap
fig, ax = plt.subplots(1, 2, figsize=(11.6, 5.8), gridspec_kw=dict(width_ratios=[1.25, 0.75]))
a = ax[0]; a.imshow(cls.T, origin="lower", extent=(0, 1, 0, 0.7), cmap=ListedColormap(cols), alpha=0.28, aspect="auto", interpolation="nearest")
ff_ = np.linspace(0, 1, 2)
for k, (xs, ys) in enumerate([((0.3, 0.3), (0, 0.7)), ((0.7, 0.7), (0, 0.7)), ((0, 0.7), (0, 0.7)), ((1, 0.3), (0, 0.7)), ((0.7, 0), (0, 0.7)), ((0.3, 1), (0, 0.7))]):
    a.plot(xs, ys, color=INK2, lw=1.3, label="Kähler-chamber walls $f_i=g_j$ (blow-ups)" if k == 0 else None)
for k in range(5):
    a.plot([], [], "s", color=cols[k], alpha=0.5, ms=9, label=names[k])
a.contour(F1, G2, cls, levels=np.arange(-0.5, 5), colors=[INK], linewidths=1.0, linestyles="--")
a.plot([], [], "--", color=INK, lw=1, label="leximin walls: two ratios $f_i/3$, $g_j/2$ tie (the leading set jumps)")
a.plot(0.4, 0.43, "o", color="white", mec=INK, ms=8, mew=1.6, zorder=6); a.annotate("the example", (0.4, 0.43), (0.48, 0.56), fontsize=8, arrowprops=dict(arrowstyle="-", color=INK2))
a.plot(0.4, 0.44, "x", color=INK, ms=7, mew=1.6, zorder=6); a.annotate("one count moved\n(column 3 to 2)", (0.4, 0.44), (0.06, 0.6), fontsize=8, arrowprops=dict(arrowstyle="-", color=INK2))
a.set_xlim(0, 1); a.set_ylim(0, 0.7); a.set_xlabel("$f_1$"); a.set_ylabel("$g_2$  (with $g_1=0.3$, $g_3=0.7-g_2$)")
a.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2); a.set_title("(a) Margin space: two kinds of wall, and they are different", fontsize=9.5)
b = ax[1]; Z = np.linspace(0.36, 0.48, 400)
for k, (lab_, fn) in enumerate([("row 1: $f_1/3$", lambda z: np.full_like(z, 0.4 / 3)), ("column 3: $g_3/2$", lambda z: (0.7 - z) / 2), ("column 1: $g_1/2$", lambda z: np.full_like(z, 0.15))]):
    b.plot(Z, fn(Z), color=[BLUE, ROSE, AQUA][k], lw=2, label=lab_)
b.plot(Z, np.minimum(0.4 / 3, (0.7 - Z) / 2), color=INK, lw=1, ls=":", label="$h^*$ = the smallest")
b.axvline(0.43, color=GREY, lw=1); b.axvline(0.44, color=GREY, lw=1, ls="--"); b.axvline(13 / 30, color=INK, lw=0.8, ls="--")
b.text(13 / 30 + 0.001, 0.1385, "leximin\nwall\n$13/30$", fontsize=7, va="bottom")
b.axvline(0.40, color=INK2, lw=1.6); b.text(0.401, 0.162, "blow-up wall\n$f_1=g_2$", fontsize=7.5)
b.text(0.4305, 0.157, "example\n$g_2=0.43$", fontsize=7.5, ha="right"); b.text(0.4415, 0.157, "one count\n$g_2=0.44$", fontsize=7.5)
b.set_xlabel("$g_2$ (at $f_1=0.4$, $g_1=0.3$)"); b.set_ylabel("ratio"); b.set_ylim(0.115, 0.17); b.legend(frameon=False, fontsize=7.4, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
b.set_title("(b) Zoom at $f_1=0.4$: the leximin wall is a third of\na count away, the blow-up wall three counts", fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT("fig_cpl_walls.pdf")); fig.savefig(OUT("fig_cpl_walls.png"), dpi=150); plt.close(fig)
print("ok")
