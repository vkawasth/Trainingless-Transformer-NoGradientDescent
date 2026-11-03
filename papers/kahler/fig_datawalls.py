"""Figure for section 6 of the note: soft evidence moves lambda. (a) F_1 along alpha: the max-min level h* and the selected valuation u*,
with the outcome wall at alpha = 1/3 inside the Kähler cone. (b) A data path from alpha = 1/5 with evidence against term 1 (r = e1)."""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from pce import kahlerbridge as KB, datawalls as DW
BLUE, ORANGE, INK2, GREY = "#2a78d6", "#eb6834", "#52514e", "#8c8c8c"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
fig, ax = plt.subplots(1, 2, figsize=(10.6, 3.9))
al = np.linspace(0.01, 0.99, 197); hs = [DW.balance_lp(*KB.F1(a))[0] for a in al]
u1 = np.where(al < 1 / 3, 1 / 3, (1 + al) / 4); u2 = np.where(al < 1 / 3, 1 / 3, (1 - al) / 2)
A = ax[0]; A.plot(al, u1, color=BLUE, lw=2, label="$u_1^*$"); A.plot(al, u2, color=ORANGE, lw=2, label="$u_2^*$")
for a in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    val = KB.valuation(*KB.F1(a), [1e-30])[0]; A.plot([a, a], val, "o", color="white", mec=INK2, ms=5)
A.axvline(1 / 3, color=GREY, ls="--", lw=1); A.text(1 / 3 + 0.01, 0.47, "outcome wall\n$\\alpha=1/3$", fontsize=8, color=INK2)
A.text(0.04, 0.05, "leading set $\\{v_1,v_2,v_3\\}$\ncandidates: a point", fontsize=8, color=INK2); A.text(0.6, 0.27, "leading set $\\{v_2,v_4\\}$\ncandidates: a segment", fontsize=8, color=INK2)
A.set_xlim(0, 1); A.set_ylim(0, 0.55); A.set_xlabel("$\\alpha$ (Kähler cone $0<\\alpha<1$)"); A.legend(frameon=False, fontsize=7.6, loc="upper left")
A.set_title("(a) $F_1$: the selected valuation is continuous;\nthe leading set and candidate dimension jump", fontsize=9.5)
v, lam0 = KB.F1(0.2); ss = np.linspace(0, 0.6, 121); U = []
for s in ss:
    lam = DW.shift(lam0, [1, 0, 0, 0], s); l1, l2, l3, l4 = lam
    U.append(np.linalg.solve(np.array([[1, -1], [2, 1]], float), np.array([l1 - l2, l1 - l3])) if DW.balance_lp(v, lam)[2] == (0, 1, 2)
             else np.array([(l1 - l3 - (l2 - l4) / 2) / 2, (l2 - l4) / 2]))
U = np.array(U); B = ax[1]; B.plot(ss, U[:, 0], color=BLUE, lw=2, label="$u_1^*(s)$"); B.plot(ss, U[:, 1], color=ORANGE, lw=2, label="$u_2^*(s)$")
for s in (0.0, 0.1, 0.3, 0.4, 0.5, 0.6):
    val = KB.valuation(v, DW.shift(lam0, [1, 0, 0, 0], s), [1e-30])[0]; B.plot([s, s], val, "o", color="white", mec=INK2, ms=5)
B.axvline(0.2, color=GREY, ls="--", lw=1); B.text(0.205, 0.02, "$s^*=1/5$: $n^*=N/5$", fontsize=8, color=INK2)
B.set_xlabel("evidence $s=n/N$ against term 1 ($r=e_1$)"); B.set_ylim(-0.02, 0.45); B.legend(frameon=False, fontsize=7.6, loc="center right")
B.set_title("(b) a data path from $\\alpha=1/5$: affine on each side,\nkink at the wall (circles: positive critical point, $T=10^{-30}$)", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_datawalls.pdf")); fig.savefig(os.path.join(HERE, "fig_datawalls.png"), dpi=150); plt.close(fig)
