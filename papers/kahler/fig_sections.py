"""Figure for the remark 'three layers' (section 12.15) of the companion note (module pce/margsec.py).
  fig_sections.pdf  (a) the tensor-square defect of the power-family section phi_q, zero only at Shannon (q = 1); (b) the chain-rule cocycle
                    c H on a symmetric fibre: c > 0 has one maximiser (a section), c < 0 has two (no section)"""
import os, sys
from fractions import Fraction as Fr
import numpy as np, mpmath as mp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from pce import margsec as G
from pce import selnat as N
BLUE, ORANGE, AQUA, VIOLET, ROSE, INK, INK2, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#c2417a", "#22211f", "#52514e", "#9a9a9a"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
gq = N.cpl(Fr(1, 4), Fr(1, 2), Fr(1, 5)); dq = N.cpl(Fr(1, 2), Fr(3, 5), Fr(1, 5)); gu, du = N.umbrella()
def tdefect(q):
    c1, s1 = G.chain_power_section([gu, du], q); c2, s2 = G.chain_power_section([gq, dq], q); cp, sp = G.chain_power_section([N.kron(gu, gq), N.kron(du, dq)], q)
    prod = {}
    for a1, v1 in zip(c1, s1):
        for a2, v2 in zip(c2, s2): prod[tuple(2 * a + b for a, b in zip(a1, a2))] = v1 * v2
    return float(mp.fsum(abs(v - prod[c]) for c, v in zip(cp, sp)) / 2)
QS = [1, Fr(99, 100), Fr(19, 20), Fr(9, 10), Fr(3, 4), Fr(1, 2), Fr(1, 4), 0, Fr(-1, 2), -1, -2, -4, -8]
DEF = [tdefect(q) for q in QS]
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
a = ax[0]; a.plot([float(q) for q in QS], DEF, "o-", color=BLUE, lw=1.8, ms=4)
a.axhline(0.0752, color=ROSE, ls="--", lw=1); a.text(-7.8, 0.0765, "facet-leximin selection: 0.0752", fontsize=7.6, color=ROSE)
a.annotate("Shannon ($q=1$): defect 0", (1, 0), (-2.5, 0.012), fontsize=7.8, color=INK, arrowprops=dict(arrowstyle="->", color=INK2))
a.annotate("Burg ($q=0$)", (0, DEF[QS.index(0)]), (-3.2, 0.035), fontsize=7.8, color=INK, arrowprops=dict(arrowstyle="->", color=INK2))
a.set_xlabel("$q$ in $\\phi_q(x)=(x^q-x)/(q(1-q))$"); a.set_ylabel("tensor-square defect (total variation)"); a.set_ylim(0, 0.09)
a.set_title("(a) Only the Shannon member of the power family\nrespects independent systems", fontsize=9)
b = ax[1]; xs = np.linspace(1e-6, 0.25 - 1e-6, 400)
def H(x): v = np.array([x, 0.25 - x, 0.25 - x, x] * 2); return -(v * np.log(v)).sum()
Hs = np.array([H(x) for x in xs])
b.plot(xs, Hs, color=BLUE, lw=1.9, label="$c>0$: maximise $H$, one maximiser (a section)")
b.plot(xs, -Hs + Hs.max() + Hs.min(), color=ORANGE, lw=1.9, label="$c<0$: maximise $-H$ (shifted): two maximisers on this line, four in the fibre")
b.plot(0.125, Hs.max(), "o", color=BLUE, ms=6); b.plot([xs[0], xs[-1]], [Hs.max(), Hs.max()], "o", color=ORANGE, ms=6)
b.set_xlabel("position $x$ along the symmetric fibre (both slices moved together)"); b.set_ylabel("value of $cH$ (nats)")
b.legend(frameon=False, fontsize=7.4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
b.set_title("(b) The chain-rule cocycle $cH$: the sign of $c$ (concavity),\nnot the cocycle, makes the choice unique", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_sections.pdf")); fig.savefig(os.path.join(HERE, "fig_sections.png"), dpi=150); plt.close(fig); print("ok", DEF)
