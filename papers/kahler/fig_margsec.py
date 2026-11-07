"""Figure for sections 12.15-12.18 of the companion note (module pce/margsec.py).
  fig_cycle.pdf  (a) the face table of D for the 4-cycle: how many faces of each dimension have a fibre of each dimension; (b) the symmetric
                 CHSH path: the identified interval of the unobserved P(A0 = 0, A1 = 0), the forced section value, and contextual data past S = 2;
                 (c) three generic paths: disagreement of the two sections and fibre width, toward a CH facet, a positivity facet and a 6-face;
                 (d) Simpson: conditional and marginal covariances under selection and Markov gluing, against what the data allow"""
import os, sys, random
from fractions import Fraction as Fr
from collections import Counter
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from pce import margsec as G
from pce import selcomp as S
from pce import selnat as N
BLUE, ORANGE, AQUA, VIOLET, ROSE, INK, INK2, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#c2417a", "#22211f", "#52514e", "#9a9a9a"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
F = G.facets_D(); faces = G.faces_D(F); Lset = {(tuple(c), c0) for c, c0 in G.local_ineqs()}
ch = [f for f in F if (tuple(map(Fr, f[0])), Fr(f[1])) not in Lset]; pos = [f for f in F if (tuple(map(Fr, f[0])), Fr(f[1])) in Lset]
fig, ax = plt.subplots(2, 2, figsize=(12.0, 8.4))

a = ax[0, 0]; tab = Counter((G.face_dim(V), G.fibre_dim_face(V)) for V in faces)
Mx = np.zeros((8, 9))
for (fd, fb), n in tab.items(): Mx[fb, fd] = n
a.imshow(np.log1p(Mx), cmap="Blues", origin="lower", aspect="auto")
for (fd, fb), n in tab.items(): a.text(fd, fb, str(n), ha="center", va="center", fontsize=8, color="white" if n > 60 else INK)
a.set_xticks(range(9)); a.set_yticks(range(8)); a.set_xlabel("dimension of the face $G$ of $D$"); a.set_ylabel("dimension of the fibre over its interior")
a.text(8, 7.45, "interior", ha="center", fontsize=7.4, color=INK2); a.text(7, 4.45, "positivity\nfacets", ha="center", fontsize=7.2, color=INK2, va="bottom")
a.text(7, 0.45, "CH facets", ha="center", fontsize=7.2, color=INK2, va="bottom")
a.set_title("(a) The 4-cycle: fibre dimension $=|V(G)|-1-\\dim G$ over each face.\nCH facets and all 1752 of their faces carry a single joint law", fontsize=9)

b = ax[0, 1]; lam = np.linspace(0, 1, 401)
b.axvspan(0.5, 1, color=ROSE, alpha=0.1, lw=0); b.text(0.75, 0.47, "contextual: in $L$, not in $D$\n(no joint law; PR box at 1)", ha="center", fontsize=7.6, color=ROSE)
m = lam <= 0.5
b.fill_between(lam[m], lam[m] / 2, (1 - lam[m]) / 2, color=GREY, alpha=0.3, lw=0, label="compatible values (identified interval)")
b.plot(lam[m], 0.25 + 0 * lam[m], color=BLUE, lw=2, label="every equivariant section (maximum entropy = leximin)")
b.plot(0.5, 0.25, "o", color=INK, ms=6); b.annotate("CH facet $S=2$:\nfibre is one point", (0.5, 0.25), (0.26, 0.42), fontsize=7.6, color=INK, arrowprops=dict(arrowstyle="->", color=INK2))
b.set_xlabel("$\\lambda$ (correlations $\\lambda,\\lambda,\\lambda,-\\lambda$; CHSH value $S=4\\lambda$)"); b.set_ylabel("$P(A_0=0,\\,A_1=0)$, never observed")
b.set_xlim(0, 1); b.set_ylim(0, 0.55); b.legend(frameon=False, fontsize=7.4, loc="lower left")
b.set_title("(b) The symmetric CHSH path: the data pin the unobserved\nquantity more tightly, until no joint law is left", fontsize=9)

c = ax[1, 0]; random.seed(2)
def combo(V):
    w = [Fr(random.randint(1, 9)) for _ in V]; t = sum(w); return [sum(wi * G.VERTS[k][r] for wi, k in zip(w, V)) / t for r in range(8)]
d0 = combo(range(16)); f3 = [V for V in faces if G.face_dim(V) == 6 and len(V) == 10][0]
grid = [Fr(k, 20) for k in range(19)] + [Fr(19, 20), Fr(39, 40), Fr(99, 100), Fr(1)]
for (name, V, col) in (("toward a CH facet (fibre $\\to$ point)", ch[0][2], BLUE), ("toward a positivity facet (fibre 4)", pos[0][2], ORANGE), ("toward a 6-face (fibre 3)", f3, VIOLET)):
    d1 = combo(sorted(V)); xs = []; tv = []; wd = []
    for l in grid:
        dd = [(1 - l) * x + l * y for x, y in zip(d0, d1)]; me = G.maxent(dd, iters=200000, tol=1e-13); lx = np.array([float(v) for v in G.leximin(dd)])
        xs.append(float(l)); tv.append(0.5 * np.abs(me - lx).sum()); wd.append(G.cell_width(dd))
    c.plot(xs, tv, color=col, lw=1.9, label=name); c.plot(xs, wd, color=col, lw=1.0, ls="--")
c.plot([], [], color=INK2, lw=1.9, label="solid: disagreement of the two sections (TV)"); c.plot([], [], color=INK2, lw=1, ls="--", label="dashed: fibre width (largest range of a cell)")
c.set_xlabel("position on a straight path from an interior point to the face"); c.set_ylabel("probability")
c.legend(frameon=False, fontsize=7.2, loc="upper left"); c.set_ylim(0, 0.52)
c.set_title("(c) Rules are forced to agree only where the fibre collapses:\nat CH facets, not at structural zeros", fontsize=9)

e = ax[1, 1]
g = [[Fr(5, 24), Fr(11, 24)], [Fr(1, 24), Fr(7, 24)]]; dd = [[Fr(3, 23), Fr(11, 92)], [Fr(15, 22), Fr(3, 44)]]
p = g[0][0] + g[0][1]; r = dd[0][0] + dd[1][0]; sl = []
for bb in range(2):
    q = g[0][bb] + g[1][bb]; rb, sb = g[0][bb], dd[bb][0]; mid = (max(Fr(0), rb + sb - q) + min(rb, sb)) / 2; sl.append(float(mid / q - (rb / q) * (sb / q)))
comp = float(S.select_binary(g, dd) - p * r); mk = float(N.markov(g, dd)[0][0] - p * r); lo, hi = N.identified(g, dd)
xs = [0, 1, 2.2]; e.bar([0 - 0.18, 1 - 0.18], sl, width=0.34, color=BLUE, label="selection")
e.bar([0 + 0.18, 1 + 0.18], [0, 0], width=0.34, color=AQUA, label="Markov gluing (zero by construction)"); e.plot([0.18, 1.18], [0, 0], "s", color=AQUA, ms=5)
e.bar([2.2 - 0.18], [comp], width=0.34, color=BLUE); e.bar([2.2 + 0.18], [mk], width=0.34, color=AQUA)
e.add_patch(Rectangle((2.62, float(lo - p * r)), 0.22, float(hi - lo), color=GREY, alpha=0.4, lw=0)); e.text(2.73, float(hi - p * r) + 0.003, "data allow", ha="center", fontsize=7.2, color=INK2)
e.axhline(0, color=INK2, lw=0.8); e.set_xticks(xs); e.set_xticklabels(["given $Y$ = yes", "given $Y$ = no", "marginal ($Y$ ignored)"])
e.set_ylabel("covariance of $X$ and $Z$"); e.legend(frameon=False, fontsize=7.4, loc="lower left"); e.set_xlim(-0.5, 3.0)
e.set_title("(d) Simpson's reversal made by a rule: selection gives positive\nassociation in each slice and negative overall; the data allow either", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_cycle.pdf")); fig.savefig(os.path.join(HERE, "fig_cycle.png"), dpi=150); plt.close(fig); print("ok")
