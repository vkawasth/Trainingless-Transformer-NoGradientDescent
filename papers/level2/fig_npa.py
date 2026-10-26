"""fig_npa.py -- the categorical home, computed (note Section 12; data from engine/check_npa.py -> npa_results.json).
(a) isotropic PR boxes: the classical lam* and the dimension-free quantum fraction (NPA level 1+AB) against the correlation strength v,
with the closed forms; (b) random no-signalling boxes: lam* against the quantum fraction at levels 2, 1+AB and 1; (c) holonomy against
finite dimension: the best unitary residual ||UV - qVU|| for q = e^(2 pi i/3) and for an irrational phase, by dimension."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "npa_results.json")))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.6, 2.6))
I = np.array(R["isotropic"]); v = np.linspace(0.5, 1, 200); r2 = 1 / np.sqrt(2)
a.plot(v, 2 * (1 - v), color="#888", lw=0.8, ls="--"); a.plot(v, np.where(v <= r2, 1, (1 - v) / (1 - r2)), color="#888", lw=0.8, ls="--")
a.plot(I[:, 0], I[:, 1], "o", ms=3.5, color="#2a78d6", label="classical $\\lambda^*$"); a.plot(I[:, 0], I[:, 2], "s", ms=3.5, color="#9b59b6", label="quantum (NPA, no dimension)")
a.axvline(r2, color="#c8423b", lw=0.7, ls=":"); a.text(r2 + 0.005, 0.05, "$1/\\sqrt{2}$", color="#c8423b", fontsize=7)
a.set_xlabel("PR strength $v$"); a.set_ylabel("fraction saved"); a.legend(frameon=False, fontsize=6.3, loc="lower left")
a.set_title("(a) isotropic boxes", loc="left", fontsize=8.5)
C = np.array(R["chain"])
for k, (lab, col, mk) in enumerate([("NPA level 2", "#2e9e6b", "o"), ("NPA level 1+AB", "#9b59b6", "s"), ("NPA level 1", "#eb6834", "^")]):
    b.plot(C[:, 0], C[:, k + 1], mk, ms=4 - k * 0.6, color=col, label=lab, mfc="none" if k else None)
b.plot([0.5, 1], [0.5, 1], color="#888", lw=0.8, ls="--"); b.set_xlabel("classical $\\lambda^*$"); b.set_ylabel("quantum fraction")
b.legend(frameon=False, fontsize=6.3, loc="lower right"); b.set_title("(b) the hierarchy sits above $\\lambda^*$", loc="left", fontsize=8.5)
H = R["holonomy"]; labs = list(H); vals = [H[k] for k in labs]
cols = ["#2e9e6b" if "n=3" in k and "/3" in k else ("#c8423b" if "sqrt2" in k else "#eb6834") for k in labs]
c.bar(range(len(vals)), vals, color=cols); c.set_xticks(range(len(vals)))
c.set_xticklabels([k.replace("q=e^(2pi i/3), ", "$q^3=1$\n").replace("q=e^(2pi i sqrt2), ", "irrational\n") for k in labs], fontsize=6.3)
c.text(1, 0.04, "$\\approx 0$", ha="center", fontsize=7, color="#2e9e6b"); c.set_ylabel("best $\\|UV-qVU\\|$ (unitary)"); c.set_title("(c) holonomy vs dimension", loc="left", fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_npa.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_npa.png"), dpi=130, bbox_inches="tight")
print("ok")
