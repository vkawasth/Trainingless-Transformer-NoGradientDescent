"""fig_qglue.py -- quantum graded gluing (note Section 11; data from engine/check_qglue.py -> qglue_results.json, plus a slice computed here).
(a) a two-dimensional real slice of the pure positivity domain D_1(f) for three random pairwise windows on three qubits: the domain (shaded)
is star-shaped about 0 and bounded by pieces of the determinantal walls det f_w = 0 (one colour per window), as Prop. 2.21 and Thm 3.18
say; (b) pairwise Werner windows: the quantum fraction against the singlet weight p, the formula min(1, 3(1-p)/2) and the Z-basis
classical lam*; (c) the dephasing bound: for random quantum families, the classical lam* of the tables in 20 random product bases against
the quantum fraction (all on or above the diagonal); (d) pure single-actor explanations against mixed ones for the Werner windows."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from pce.qglue import f_eval
R = json.load(open(os.path.join(HERE, "qglue_results.json")))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 4, figsize=(8.4, 2.5)); a, b, c, d = ax
rng = np.random.default_rng(5)
def rstate(dd, mix=0.3):
    G = rng.normal(size=(dd, dd)) + 1j * rng.normal(size=(dd, dd)); Rr = G @ G.conj().T; Rr /= np.trace(Rr).real; return (1 - mix) * Rr + mix * np.eye(dd) / dd
wins = [(k, rstate(4)) for k in ([0, 1], [1, 2], [0, 2])]
u1 = rng.normal(size=8) + 1j * rng.normal(size=8); u1 /= np.linalg.norm(u1); u2 = rng.normal(size=8) + 1j * rng.normal(size=8)
u2 -= np.vdot(u1, u2) * u1; u2 /= np.linalg.norm(u2)
g = np.linspace(-0.85, 0.85, 241); Xg, Yg = np.meshgrid(g, g); inside = np.ones_like(Xg, bool); mins = []
for k, rho in wins:
    M = np.zeros_like(Xg)
    for i in range(len(g)):
        for j in range(len(g)):
            x = Xg[i, j] * u1 + Yg[i, j] * u2; M[i, j] = np.linalg.eigvalsh(f_eval(rho, k, 3, x.reshape(8, 1, 1))).min()
    mins.append(M); inside &= M >= 0
a.contourf(Xg, Yg, inside.astype(float), levels=[0.5, 1.5], colors=["#cfe0f3"])
for M, col in zip(mins, ["#2a78d6", "#eb6834", "#2e9e6b"]): a.contour(Xg, Yg, M, levels=[0], colors=[col], linewidths=1)
a.plot(0, 0, "k+", ms=6); a.set_aspect("equal"); a.set_xlabel("$s_1$"); a.set_ylabel("$s_2$")
a.set_title("(a) $D_1(f)$ on a slice; walls\n$\\det f_w=0$ per window", loc="left", fontsize=8)
W = np.array(R["werner"]); b.plot(W[:, 0], W[:, 3], color="#888", lw=1, ls="--", label="$\\min(1,\\,3(1-p)/2)$")
b.plot(W[:, 0], W[:, 1], "o", color="#9b59b6", ms=3.5, label="quantum $\\lambda_Q$"); b.plot(W[:, 0], W[:, 2], "x", color="black", ms=4, label="classical $\\lambda^*$ (Z basis)")
b.axvline(1 / 3, color="#c8423b", lw=0.7, ls=":"); b.set_xlabel("singlet weight $p$"); b.set_ylabel("fraction saved"); b.legend(frameon=False, fontsize=6)
b.set_title("(b) Werner triangle:\nmonogamy", loc="left", fontsize=8)
Pp = np.array(R["dephasing_pairs"]); c.plot(Pp[:, 0], Pp[:, 1], ".", ms=3, color="#2a78d6"); lo, hi = Pp.min() - 0.01, Pp.max() + 0.01
c.plot([lo, hi], [lo, hi], color="#888", lw=0.8, ls="--"); c.set_xlabel("quantum fraction $\\lambda_Q$"); c.set_ylabel("classical $\\lambda^*$ in a product basis")
c.set_title("(c) dephasing bound:\n$\\lambda_Q\\leq\\lambda^*$ in each sampled basis", loc="left", fontsize=8)
PV = np.array(R["pure_vs_mixed"]); xs = np.arange(len(PV)); d.bar(xs - 0.18, PV[:, 1], 0.36, color="#eb6834", label="pure (level 1), lower bound")
d.bar(xs + 0.18, PV[:, 2], 0.36, color="#9b59b6", label="mixed (SDP)"); d.set_xticks(xs); d.set_xticklabels([f"p={v:g}" for v in PV[:, 0]])
d.set_ylabel("fraction saved"); d.legend(frameon=False, fontsize=6); d.set_title("(d) one pure actor vs mixed", loc="left", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_qglue.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_qglue.png"), dpi=130, bbox_inches="tight")
print("ok")
