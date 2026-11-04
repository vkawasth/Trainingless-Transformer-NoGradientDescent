"""Figures for section 12.7 of the companion note: the column-binding example classified exactly.
  fig_cb_newton.pdf   the Newton polygon of the quintic and the two branch groups (and the roots' sizes against tau)
  fig_cb_objects.pdf  the five objects over their two fibres, with exact leading holonomies
  fig_cb_hom.pdf      the 5 x 5 Hom table as a picture"""
import os, sys
import numpy as np
import mpmath as mp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, Arc, Rectangle
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from pce import colbind as CB, couplingfibre as CF
BLUE, ORANGE, AQUA, VIOLET, ROSE, INK, INK2, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#c2417a", "#22211f", "#52514e", "#9a9a9a"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
# ---------------------------------------------------------------- Newton polygon
fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.4)); a = ax[0]
pts, hull, edges = CB.newton_polygon(); P = np.array(pts)
a.plot(P[:, 0], P[:, 1], "o", color=INK2, ms=7, zorder=3)
for (x, y), lab in zip(pts, ["$s^5$", "$\\tau^2s^4$", "$-2\\tau^3s^3$", "$-2\\tau^5s^2$", "$\\tau^8$"]): a.annotate(lab, (x, y), (x + 0.12, y + 0.25), fontsize=8.5)
H = np.array(hull); a.plot(H[:2, 0], H[:2, 1], color=ROSE, lw=3, label="edge 1: slope $-5/3$, 3 roots, $s\\approx2^{-1/3}\\omega\\,\\tau^{5/3}$")
a.plot(H[1:, 0], H[1:, 1], color=BLUE, lw=3, label="edge 2: slope $-3/2$, 2 roots, $s\\approx\\pm\\sqrt{2}\\,\\tau^{3/2}$")
a.set_xlabel("power of $s$"); a.set_ylabel("power of $\\tau$ in the coefficient"); a.set_xlim(-0.3, 5.8); a.set_ylim(-0.5, 9.2)
a.legend(frameon=False, fontsize=7.8, loc="upper right"); a.set_title("(a) The Newton polygon of $q(s)=s^5+\\tau^2s^4-2\\tau^3s^3-2\\tau^5s^2+\\tau^8$:\nits lower edges give every valuation and leading coefficient", fontsize=9.5)
b = ax[1]; taus = [mp.mpf(10) ** -k for k in np.linspace(2, 40, 39)]
for tau in taus:
    br = CB.branches(tau, 300)
    for y1, y2 in br:
        grp = abs(mp.log(abs(y2)) / mp.log(tau) - 1.5) < abs(mp.log(abs(y2)) / mp.log(tau) - 5 / 3)
        b.plot(float(-mp.log10(tau)), float(-mp.log10(abs(y2))), "o", ms=3, color=BLUE if grp else ROSE)
xs = np.linspace(2, 40, 2); b.plot(xs, 1.5 * xs - np.log10(np.sqrt(2)), color=BLUE, lw=1, ls="--", label="$|y_2|=\\sqrt{2}\\,\\tau^{3/2}$ (2 branches: the leximin fibre)")
b.plot(xs, 5 / 3 * xs + np.log10(2) / 3, color=ROSE, lw=1, ls="--", label="$|y_2|=2^{-1/3}\\tau^{5/3}$ (3 branches: the fibre $(2/15,1/6)$)")
b.set_xlabel("$\\log_{10}(1/\\tau)$, with $\\tau=T^{1/10}$"); b.set_ylabel("$\\log_{10}(1/|y_2|)$"); b.legend(frameon=False, fontsize=7.8, loc="upper left")
b.set_title("(b) The five roots follow the two predicted lines exactly;\nthe offsets are the moduli of the leading coefficients", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_cb_newton.pdf")); fig.savefig(os.path.join(HERE, "fig_cb_newton.png"), dpi=150); plt.close(fig)
# ---------------------------------------------------------------- objects over fibres
fig, a = plt.subplots(figsize=(7.4, 5.6)); f, g = (Fr(2, 5), Fr(3, 5)), (Fr(1, 2), Fr(3, 10), Fr(1, 5)); Pp, _ = CF.polygon(f, g); PF = [(float(x), float(y)) for x, y in Pp]
a.add_patch(Polygon(PF, closed=True, fc="#f5f5f5", ec=INK2, lw=1.1)); a.plot([0.1, 0.2], [0.2, 0.1], color=ORANGE, lw=4, alpha=0.6, label="the candidate segment (level 1)")
objs = [("(0.15, 0.15)", (0.15, 0.15), (0.33, 0.24), [("positive", BLUE, "o", "$(\\frac{\\sqrt{2}}{2},\\sqrt{2})$"), ("real $(-,-)$", ROSE, "s", "$(-\\frac{\\sqrt{2}}{2},-\\sqrt{2})$")]),
        ("(2/15, 1/6)", (2 / 15, 1 / 6), (0.16, 0.37), [("real $(-,+)$", ROSE, "s", "$(-2^{1/3},2^{-1/3})$"), ("complex", GREY, "D", "$(-2^{1/3}\\bar\\omega,2^{-1/3}\\omega)$"),
                                                         ("complex", GREY, "D", "$(-2^{1/3}\\omega,2^{-1/3}\\bar\\omega)$")])]
for lab, u, (bx, by), items in objs:
    a.plot(*u, "x", color=INK, ms=8, mew=1.8, zorder=6); dx = 0.1; w = dx * len(items) + 0.01; h = 0.065
    a.add_patch(FancyBboxPatch((bx - w / 2, by - h / 2), w, h, boxstyle="round,pad=0.006", fc="white", ec=INK2, lw=0.9, zorder=6))
    a.annotate("", u, (bx, by - h / 2 - 0.006 if by > u[1] else by + h / 2), arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    a.text(bx, by + h / 2 + 0.008, f"fibre {lab}", ha="center", fontsize=8.5)
    for k, (nm, c_, mk, hl) in enumerate(items):
        x = bx - w / 2 + 0.005 + dx * (k + 0.5); a.plot(x, by + 0.012, mk, color=c_, ms=8, zorder=8)
        a.add_patch(Arc((x, by + 0.024), 0.018, 0.018, theta1=-50, theta2=230, color=c_, lw=1, zorder=8)); a.text(x, by - 0.008, hl, fontsize=6.4, ha="center", va="top", zorder=9)
a.set_xlim(-0.03, 0.47); a.set_ylim(-0.03, 0.44); a.set_aspect("equal"); a.set_xlabel("$x=\\alpha_{11}$"); a.set_ylabel("$y=\\alpha_{12}$")
a.set_title("The five objects of the column-binding example over their two fibres\n(labels: exact leading holonomies; $\\omega=e^{2\\pi i/3}$; loops: End $=\\mathrm{Cl}_2$)", fontsize=9.5)
a.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_cb_objects.pdf")); fig.savefig(os.path.join(HERE, "fig_cb_objects.png"), dpi=150); plt.close(fig)
# ---------------------------------------------------------------- the Hom table
names = ["$P$ positive", "$N$ real $(-,-)$", "$R$ real $(-,+)$", "$C_\\omega$ complex", "$C_{\\bar\\omega}$ complex"]; fib = [0, 0, 1, 1, 1]
diag = ["$\\det\\sim\\sqrt{2}\\,\\tau^{1/2}$", "$\\det\\sim-\\sqrt{2}\\,\\tau^{1/2}$", "$\\det\\sim-c\\,\\tau^{2/3}$", "$\\det\\sim-c\\,\\bar\\omega^2\\tau^{2/3}$", "$\\det\\sim-c\\,\\omega^2\\tau^{2/3}$"]
fig, a = plt.subplots(figsize=(8.2, 6.6))
for i in range(5):
    for j in range(5):
        if i == j: fc, txt = "#dbe8f8", "$\\mathrm{Cl}(Q)\\cong M_2$\n" + diag[i] + "\nreduced: rank 1"
        elif fib[i] != fib[j]: fc, txt = "#f3f3f3", "0\ndisjoint fibres"
        else: fc, txt = "#fde9df", "0\nsame fibre,\ndistinct $m_0$"
        a.add_patch(Rectangle((j, 4 - i), 1, 1, fc=fc, ec="white", lw=2)); a.text(j + 0.5, 4 - i + 0.5, txt, ha="center", va="center", fontsize=7.4)
a.set_xticks(np.arange(5) + 0.5); a.set_xticklabels(names, rotation=20, fontsize=8); a.set_yticks(np.arange(5) + 0.5); a.set_yticklabels(names[::-1], fontsize=8)
a.set_xlim(0, 5); a.set_ylim(0, 5); a.set_aspect("equal"); a.tick_params(length=0)
for sp_ in a.spines.values(): sp_.set_visible(False)
a.plot([2, 2], [0, 5], color=INK2, lw=1.2); a.plot([0, 5], [3, 3], color=INK2, lw=1.2)
a.set_title("Hom between the five objects ($c=3\\cdot2^{-4/3}$): only the diagonal survives.\nFirst two: the leximin fibre; last three: the fibre $(2/15,1/6)$", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_cb_hom.pdf")); fig.savefig(os.path.join(HERE, "fig_cb_hom.png"), dpi=150); plt.close(fig); print("ok")
