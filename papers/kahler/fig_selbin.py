"""Figure for section 12.10 of the companion note: selection on binary chains.
  fig_sb.pdf  (a) uniform margins: correlations compose by sign-times-min under selection, by product under Markov gluing; (b) the exact associator
              along one parameter; (c) the selected composite along a path of a 3 x 2 x 3 chain, across changes of the facet set"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from pce import selcomp as S
BLUE, ORANGE, AQUA, VIOLET, ROSE, INK, INK2, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#c2417a", "#22211f", "#52514e", "#9a9a9a"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
fig, ax = plt.subplots(1, 3, figsize=(13.0, 3.9))
a = ax[0]; r2 = np.linspace(-1, 1, 401)
for r1, col in ((0.8, BLUE), (-0.5, ROSE)):
    a.plot(r2, np.sign(r1 * r2) * np.minimum(abs(r1), abs(r2)), color=col, lw=2, label=f"selection, $\\rho_1={r1}$")
    a.plot(r2, r1 * r2, color=col, lw=1.2, ls="--", label=f"Markov, $\\rho_1={r1}$")
a.axhline(0, color=GREY, lw=0.6); a.axvline(0, color=GREY, lw=0.6)
a.set_xlabel("$\\rho_2$"); a.set_ylabel("correlation of the composite"); a.legend(frameon=False, fontsize=7.4, loc="upper left")
a.set_title("(a) Uniform margins: selection is the tropical\nproduct, sign$(\\rho_1\\rho_2)\\min(|\\rho_1|,|\\rho_2|)$; Markov multiplies", fontsize=9)
b = ax[1]; ps = [Fr(2, 5), Fr(1, 3), Fr(3, 7), Fr(1, 4)]; t1, t2 = Fr(13, 60), Fr(7, 30)
tt = [Fr(k, 840) for k in range(0, 211)]; A = [S.associator_bin(ps, [t1, t2, t]) for t in tt]
b.plot([float(t) for t in tt], [float(x) for x in A], color=INK, lw=1.8)
for k in [Fr(1, 30), Fr(13, 120), Fr(2, 15), Fr(3, 20), Fr(143, 840), Fr(41, 210), Fr(7, 30)]: b.plot(float(k), float(S.associator_bin(ps, [t1, t2, k])), "o", color=ORANGE, ms=4)
b.axhline(0, color=GREY, lw=0.8); b.axvspan(13 / 120, 2 / 15, color=AQUA, alpha=0.25, label="$A=0$ on $[13/120,\\,2/15]$")
b.plot(5 / 28, 0, "D", color=AQUA, ms=6, label="and at $5/28$"); b.plot(1 / 80, -41 / 960, "*", color=ROSE, ms=10, label="the $K_3$ example: $-41/960$")
b.axvline(3 / 28, color=GREY, ls=":", lw=1); b.text(3 / 28, 0.022, " independence", fontsize=7.4, color=INK2)
b.set_xlabel("$t_3$ (the $00$ entry of $\\gamma_3$)"); b.set_ylabel("associator $A$ ($00$ entry)"); b.legend(frameon=False, fontsize=7.4, loc="lower right")
b.set_title("(b) The exact associator along one coupling:\npiecewise linear, 7 kinks (orange), slopes $0,\\pm 1/4, 1/2$", fontsize=9)
c = ax[2]; rng = np.random.default_rng(12)
def rc(pp, qq, moves=3):
    g = [[pp[i] * qq[j] for j in range(len(qq))] for i in range(len(pp))]
    for _ in range(moves):
        i, k = rng.choice(len(pp), 2, replace=False); j, l = rng.choice(len(qq), 2, replace=False); d = min(g[i][l], g[k][j]) * Fr(int(rng.integers(2, 10)), 10)
        g[i][j] += d; g[k][l] += d; g[i][l] -= d; g[k][j] -= d
    return g
pA = (Fr(1, 5), Fr(3, 10), Fr(1, 2)); pB = (Fr(2, 5), Fr(3, 5)); pC = (Fr(1, 4), Fr(1, 3), Fr(5, 12))
ga = rc(pA, pB, 4); gb = rc(pA, pB, 4); g2 = rc(pB, pC, 4); ss = []; ys = []; prevF = None; ch = []
for k in range(201):
    s = Fr(k, 200); g1 = [[(1 - s) * ga[i][j] + s * gb[i][j] for j in range(2)] for i in range(3)]; F = tuple(S.facet_cells([g1, g2]))
    if prevF is not None and F != prevF: ch.append(float(s))
    prevF = F; cc = S.select([g1, g2]); ss.append(float(s)); ys.append([float(cc[0][0]), float(cc[1][1]), float(cc[2][2])])
ys = np.array(ys)
for i, col in enumerate([BLUE, ORANGE, VIOLET]): c.plot(ss, ys[:, i], color=col, lw=1.6, label=f"entry $({i}{i})$")
for x in ch: c.axvline(x, color=ROSE, lw=0.8, ls="--")
c.plot([], [], color=ROSE, lw=0.8, ls="--", label="facet set changes")
c.set_xlabel("position $s$ on the path of $\\gamma_1$"); c.set_ylabel("selected composite"); c.legend(frameon=False, fontsize=7.4)
c.set_title("(c) A $3\\times2\\times3$ chain: the composite moves\ncontinuously across changes of the facet set", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_sb.pdf")); fig.savefig(os.path.join(HERE, "fig_sb.png"), dpi=150); plt.close(fig); print("ok", ch)
