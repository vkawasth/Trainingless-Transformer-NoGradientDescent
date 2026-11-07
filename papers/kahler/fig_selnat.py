"""Figures for sections 12.13-12.14 of the companion note (module pce/selnat.py).
  fig_selcat.pdf  the small category on the umbrella chain: (a) objects and couplings, the two composites; (b) the tensor square; (c) the
                  coarse-graining square. Selection breaks both squares, Markov gluing keeps both.
  fig_selrob.pdf  how little changes a decision: (a) what the data determine (the interval), and the two composites; (b) margin-preserving
                  edits of the forecast-umbrella data; (c) changes that cost no data at all"""
import os, sys, random
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from pce import selnat as N
from pce import selcomp as S
BLUE, ORANGE, AQUA, VIOLET, ROSE, INK, INK2, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#c2417a", "#22211f", "#52514e", "#9a9a9a"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
E = N.UMBRELLA; p, q, r, t1, t2, tau = E["p"], E["q"], E["r"], E["t1"], E["t2"], E["tau"]; g, d = N.umbrella()
G = [[0], [1, 2]]

# ------------------------------------------------------------------ fig_selcat
def square(ax, x, y, M, title, w=0.62, col=BLUE, fs=7.2):
    n, m = len(M), len(M[0]); h = w * n / m
    for i in range(n):
        for j in range(m):
            v = float(M[i][j]); ax.add_patch(Rectangle((x + j * w / m, y + (n - 1 - i) * h / n), w / m, h / n, facecolor=col, alpha=0.12 + 0.8 * min(1, v / 0.5), edgecolor="white", lw=1.5))
            ax.text(x + (j + .5) * w / m, y + (n - 1 - i + .5) * h / n, str(M[i][j]), ha="center", va="center", fontsize=fs, color=INK if v < 0.3 else "white")
    ax.text(x + w / 2, y + h + 0.06, title, ha="center", va="bottom", fontsize=8, color=INK2)

def arrow(ax, a, b, txt="", col=INK2, off=(0, 0.05), ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, color=col, lw=1.2, linestyle=ls, shrinkA=4, shrinkB=4))
    if txt: ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], txt, ha="center", va="bottom", fontsize=7.6, color=col)

fig, ax = plt.subplots(1, 3, figsize=(13.0, 4.1), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
a = ax[0]; a.set_xlim(0, 3.2); a.set_ylim(-0.15, 2.35); a.axis("off")
for k, (nm, pr) in enumerate((("forecast rain", p), ("umbrella carried", q), ("wet", r))):
    x0 = 0.1 + k * 1.1; a.add_patch(Rectangle((x0, 2.0), 0.8 * float(pr), 0.12, color=BLUE)); a.add_patch(Rectangle((x0 + 0.8 * float(pr), 2.0), 0.8 * (1 - float(pr)), 0.12, color=GREY, alpha=.35))
    a.text(x0 + 0.4, 2.17, f"{'XYZ'[k]}: P({nm}) = {pr}", ha="center", fontsize=7.8, color=INK)
square(a, 0.5, 1.0, g, r"$\gamma_1\in\Pi(X,Y)$", w=0.78); square(a, 1.6, 1.0, d, r"$\gamma_2\in\Pi(Y,Z)$", w=0.78)
mk = N.markov(g, d); se = N.sel(g, d)
square(a, 0.5, -0.1, mk, "Markov $\\gamma_1D^{-1}\\gamma_2$", col=AQUA, w=0.78, fs=6.8); square(a, 1.6, -0.1, se, "selection $\\sigma(\\gamma_1,\\gamma_2)$", w=0.78, fs=6.8)
a.text(1.39, 0.86, "two composites in $\\Pi(X,Z)$", ha="center", fontsize=8.2, color=INK)
a.set_title("(a) Objects are laws, morphisms are couplings;\nthe same data compose in two ways", fontsize=9)

def sq(a, corners, labels, vals, title):
    a.set_xlim(-0.2, 2.2); a.set_ylim(-0.55, 2.15); a.axis("off")
    (A, B, C, D) = corners
    for (x, y), t in zip(corners, labels): a.text(x, y, t, ha="center", va="center", fontsize=8.2, color=INK, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREY, lw=0.8))
    arrow(a, (A[0] + .35, A[1]), (B[0] - .35, B[1]), "select", BLUE); arrow(a, (A[0], A[1] - .22), (C[0], C[1] + .22), "", INK2)
    arrow(a, (B[0], B[1] - .22), (D[0], D[1] + .22), "", INK2); arrow(a, (C[0] + .35, C[1]), (D[0] - .35, D[1]), "select", BLUE, off=(0, -0.2))
    a.text(1.0, -0.12, vals[0], ha="center", fontsize=8.2, color=BLUE); a.text(1.0, -0.38, vals[1], ha="center", fontsize=8.2, color=AQUA)
    a.set_title(title, fontsize=9)
gq = N.cpl(Fr(1, 4), Fr(1, 2), Fr(1, 5)); dq = N.cpl(Fr(1, 2), Fr(3, 5), Fr(1, 10))
L1 = N.sel(N.kron(g, gq), N.kron(d, dq))[0][0]; R1 = N.kron(N.sel(g, d), N.sel(gq, dq))[0][0]
sq(ax[1], [(0, 1.7), (2, 1.7), (0, 0.4), (2, 0.4)], ["two chains", "two composites", "product chain", "composite of\nthe product"],
   [f"selection: {R1} (select, then $\\otimes$) $\\neq$ {L1} ($\\otimes$, then select)", "Markov gluing: the two paths agree"],
   "(b) Adding an independent system:\nthe tensor square")
ax[1].text(-0.12, 1.05, "$\\otimes$", fontsize=11, color=INK2); ax[1].text(2.05, 1.05, "$\\otimes$", fontsize=11, color=INK2)
d3 = [[Fr(1, 10), Fr(1, 10), Fr(2, 15)], [Fr(1, 5), Fr(1, 5), Fr(4, 15)]]
L2 = N.sel(g, N.merge_cols(d3, G))[0][0]; R2 = N.merge_cols(N.sel(g, d3), G)[0][0]
sq(ax[2], [(0, 1.7), (2, 1.7), (0, 0.4), (2, 0.4)], ["$\\gamma_1$, fine $\\gamma_2$\n($Z$ has 3 states)", "fine composite", "$\\gamma_1$, coarse $\\gamma_2$\n(merge 2 states)", "coarse composite"],
   [f"selection: {R2} (select, then merge) $\\neq$ {L2} (merge, then select)", "Markov gluing: the two paths agree"], "(c) Forgetting a distinction at the end:\nthe coarse-graining square")
ax[2].text(-0.16, 1.05, "merge", fontsize=7.6, color=INK2, rotation=90, va="center"); ax[2].text(2.1, 1.05, "merge", fontsize=7.6, color=INK2, rotation=90, va="center")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_selcat.pdf")); fig.savefig(os.path.join(HERE, "fig_selcat.png"), dpi=150); plt.close(fig)

# ------------------------------------------------------------------ fig_selrob
fl = lambda x: float(x / p)
lo, hi = N.identified(g, d); us, um = N.sel(g, d)[0][0], N.markov(g, d)[0][0]
fig, ax = plt.subplots(1, 3, figsize=(13.0, 3.9), gridspec_kw=dict(width_ratios=[1, 1.15, 1.15]))
a = ax[0]
a.add_patch(Rectangle((fl(lo), 0.35), fl(hi) - fl(lo), 0.3, color=GREY, alpha=0.3, lw=0))
a.text((fl(lo) + fl(hi)) / 2, 0.69, "every composite compatible with the data", ha="center", bbox=dict(fc="white", ec="none", pad=1), fontsize=7.6, color=INK2)
a.plot([fl(um)], [0.5], "o", color=AQUA, ms=8); a.plot([fl(us)], [0.5], "D", color=BLUE, ms=7)
a.text(fl(um) + 0.01, 0.3, f"Markov\n{float(um / p):.3f}", ha="left", va="top", fontsize=7.6, color=AQUA)
a.text(fl(us) + 0.01, 0.3, f"selection\n{float(us / p):.3f}", ha="left", va="top", fontsize=7.6, color=BLUE)
a.axvline(0.5, color=ROSE, ls="--", lw=1.1); a.text(0.505, 0.93, r"threshold $\tau=1/2$", fontsize=7.6, color=ROSE)
for x0, y0, col, lab in ((fl(us), 0.83, BLUE, f"worst error {float((hi - lo) / 2 / p):.3f}"), (fl(um), 0.08, AQUA, f"worst error {float((hi - um) / p):.3f}")):
    a.annotate("", (fl(hi), y0), (x0, y0), arrowprops=dict(arrowstyle="<->", color=col, lw=1)); a.text((x0 + fl(hi)) / 2, y0 + 0.02, lab, ha="center", fontsize=7.2, color=col)
a.set_xlim(0.2, 1.05); a.set_ylim(-0.12, 1.02); a.set_yticks([]); a.spines["left"].set_visible(False)
a.set_xlabel("P(wet | forecast rain) of the composite")
a.set_title("(a) What the data determine: an interval.\nSelection is its centre (least worst-case error)", fontsize=9)

b = ax[1]; ss = [Fr(k, 2000) for k in range(-433, 233)]
ss = [s for s in ss if 0 <= t1 + s <= min(p, q)]
X = [float(2 * abs(s)) * (1 if s > 0 else -1) * 100 for s in ss]
b.fill_between(X, [fl(N.identified(N.cpl(p, q, t1 + s), d)[0]) for s in ss], [fl(N.identified(N.cpl(p, q, t1 + s), d)[1]) for s in ss], color=GREY, alpha=0.25, lw=0, label="compatible composites")
b.plot(X, [fl(S.u_markov(p, q, r, t1 + s, t2)) for s in ss], color=AQUA, lw=1.8, label="Markov")
b.plot(X, [fl(S.u_bin(p, q, r, t1 + s, t2)) for s in ss], color=BLUE, lw=1.8, label="selection")
b.axhline(0.5, color=ROSE, ls="--", lw=1.1); b.axvline(0, color=GREY, lw=0.6)
for s, col, txt, dy in ((Fr(-1, 76), AQUA, "Markov flips\nat 2.6%", -0.17), (Fr(-9, 140), BLUE, "selection flips\nat 12.9%", 0.12), (Fr(1, 12), INK2, "data force\n'wet' at 16.7%", -0.2)):
    x = float(2 * abs(s)) * (1 if s > 0 else -1) * 100; b.plot(x, 0.5, "o", color=col, ms=5); b.text(x, 0.5 + dy, txt, ha="center", fontsize=7.2, color=col)
b.set_xlabel("edit of the forecast-umbrella records (% moved; sign = direction)"); b.set_ylabel("P(wet | forecast rain)")
b.legend(frameon=False, fontsize=7.2, loc="lower right")
b.set_title("(b) Poisoning inside the data: margin-preserving\nedits needed to flip each decision", fontsize=9)

c = ax[2]
ref = []
for x in [Fr(k, 120) for k in range(21)]:
    for y in [Fr(k, 120) for k in range(21)]:
        gf = [[x, g[0][0] - x, g[0][1]], [Fr(1, 6) - x, g[1][0] - (Fr(1, 6) - x), g[1][1]]]; df = [[y, Fr(1, 6) - y], [d[0][0] - y, d[0][1] - (Fr(1, 6) - y)], d[1]]
        if min(min(rr) for rr in gf) < 0 or min(min(rr) for rr in df) < 0: continue
        L, U = N.identified(gf, df); ref.append((fl(N.markov(gf, df)[0][0]), fl(S.select_binary(gf, df)), fl(L), fl(U)))
fine = []
for i in range(1, 8):
    for j in range(1, 8):
        a_, b_ = t2 * Fr(i, 8), (r - t2) * Fr(j, 8); df = [[a_, t2 - a_, q - t2], [b_, r - t2 - b_, 1 - q - r + t2]]
        fine.append((fl(N.merge_cols(N.markov(g, df), [[0, 1], [2]])[0][0]), fl(N.merge_cols(N.sel(g, df), [[0, 1], [2]])[0][0])))
random.seed(5); ind = []
for _ in range(14):
    A_ = Fr(random.randint(1, 9), 10); B_ = Fr(random.randint(1, 9), 10); C_ = Fr(random.randint(1, 9), 10)
    l1, h1 = max(0, A_ + B_ - 1), min(A_, B_); l2, h2 = max(0, B_ + C_ - 1), min(B_, C_)
    x = l1 + (h1 - l1) * Fr(random.randint(0, 10), 10); y = l2 + (h2 - l2) * Fr(random.randint(0, 10), 10)
    P = N.sel(N.kron(g, N.cpl(A_, B_, x)), N.kron(d, N.cpl(B_, C_, y))); Pm = N.markov(N.kron(g, N.cpl(A_, B_, x)), N.kron(d, N.cpl(B_, C_, y)))
    ind.append((fl(Pm[0][0] + Pm[0][1] + Pm[1][0] + Pm[1][1]), fl(P[0][0] + P[0][1] + P[1][0] + P[1][1])))
rowsD = [("reveal a hidden distinction\nin the middle (195 ways)", ref), ("record 'wet' at a finer\nresolution (49 ways)", fine), ("attach an independent\nsystem (14 ways)", ind)]
for k, (lab, R) in enumerate(rowsD):
    y = 2 - k
    if k == 0: c.add_patch(Rectangle((min(z[2] for z in R), y - 0.32), max(z[3] for z in R) - min(z[2] for z in R), 0.64, color=GREY, alpha=0.18, lw=0))
    for j, (col, off) in enumerate(((AQUA, 0.12), (BLUE, -0.12))):
        v = [z[j] for z in R]
        if max(v) - min(v) < 1e-12: c.plot([v[0]], [y + off], "o" if j == 0 else "D", color=col, ms=6)
        else: c.plot([min(v), max(v)], [y + off] * 2, color=col, lw=5, solid_capstyle="round", alpha=0.85)
    c.text(0.27, y + 0.34, lab.replace("\n", " "), ha="left", va="bottom", fontsize=7.4, color=INK)
c.plot(fl(um), 2.95, "o", color=AQUA, ms=6); c.plot(fl(us), 2.95, "D", color=BLUE, ms=6); c.text(0.27, 3.08, "the data as recorded", fontsize=7.4, color=INK, va="bottom")
c.axvline(0.5, color=ROSE, ls="--", lw=1.1)
c.plot([], [], color=AQUA, lw=5, label="Markov (range)"); c.plot([], [], color=BLUE, lw=5, label="selection (range)")
c.legend(frameon=False, fontsize=7.2, loc="lower right", bbox_to_anchor=(1.0, -0.02)); c.set_yticks([]); c.spines["left"].set_visible(False); c.set_xlim(0.26, 1.02); c.set_ylim(-0.55, 3.4)
c.set_xlabel("P(wet | forecast rain)")
c.set_title("(c) Changes that cost no data: structure\nunderneath moves one rule or the other", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_selrob.pdf")); fig.savefig(os.path.join(HERE, "fig_selrob.png"), dpi=150); plt.close(fig)
print("ok", min(z[0] for z in ref), max(z[0] for z in ref), min(z[1] for z in ind), max(z[1] for z in ind))
