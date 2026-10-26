"""fig_noncomm.py -- non-commutativity as structure (note Section 9; data from engine/check_noncomm.py -> noncomm_results.json).
(a) order effects against the commutator norm (random projections): all below the diagonal; (b) the CHSH family: as the angle between A0
and A1 opens, ||[A0, A1]|| grows and lam* falls from 1 to 2 - sqrt 2; (c) Zeno dragging: success of steering |j, j> through N intermediate
projections, against the direct jump (N = 1); (d) the loop phase arg tr(P1 P2 P3) against the solid angle: a line of slope 1/2."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "noncomm_results.json")))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 4, figsize=(7.8, 2.4)); a, b, c, d = ax
O = np.array(R["order"]); a.plot(O[:, 0], O[:, 1], "o", ms=2, color="#2a78d6", alpha=0.6); a.plot([0, 0.5], [0, 0.5], color="#888", lw=0.8, ls="--")
a.set_xlabel("$\\|[P,Q]\\|$"); a.set_ylabel("|order effect|"); a.set_title("(a) order is bounded\nby the commutator", loc="left", fontsize=8)
S = np.array(R["chsh_sweep"]); b.plot(S[:, 2], S[:, 1], color="#c8423b", lw=1.5); b.plot(S[-1, 2], S[-1, 1], "o", color="black", ms=4)
b.text(S[-1, 2], S[-1, 1] + 0.03, "Tsirelson\n$2-\\sqrt{2}$", ha="right", fontsize=6.5); b.set_xlabel("$\\|[A_0,A_1]\\|$ (projections)"); b.set_ylabel("$\\lambda^*$")
b.set_ylim(0.5, 1.03); b.set_title("(b) commutators decide\nthe cover; gluing fails", loc="left", fontsize=8)
for (j, rows), col in zip(R["drag"].items(), ["#2a78d6", "#2e9e6b", "#eb6834"]):
    rows = np.array(rows); c.semilogx(rows[:, 0], rows[:, 1], "o-", ms=3, color=col, label=f"$j={float(j):g}$")
c.set_xlabel("intermediate projections $N$"); c.set_ylabel("success of steering"); c.set_ylim(0, 1.02); c.legend(frameon=False, fontsize=6.5)
c.set_title("(c) probability paths\n(Zeno dragging, 90°)", loc="left", fontsize=8)
B = np.array(R["bargmann"]); d.plot(B[:, 0], B[:, 1], "o", ms=2, color="#9b59b6", alpha=0.7); xx = np.linspace(-2 * np.pi, 2 * np.pi, 10)
d.plot(xx, xx / 2, color="#888", lw=0.8, ls="--"); d.set_xlabel("signed solid angle $\\Omega$"); d.set_ylabel("arg tr$(P_1P_2P_3)$")
d.set_title("(d) loops leave a phase\n($\\Omega/2$, spin 1/2)", loc="left", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_noncomm.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_noncomm.png"), dpi=130, bbox_inches="tight")
print("ok")
