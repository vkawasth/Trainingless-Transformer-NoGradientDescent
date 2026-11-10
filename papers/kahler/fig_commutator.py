"""Figure for 'What braids and knots carry': the commutator braid. (a) Strand paths of A B A^-1 B^-1 in the plane. (b) The winding
angles (theta_12, theta_23) for two orderings with the same moves: the commutator encloses area (2 pi)^2, the reordering encloses none."""
import os, sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from pce import braidknot as BK

fig, ax = plt.subplots(1, 2, figsize=(8.6, 4.0))
P = BK.commutator_path([BK.TWIST[x] for x in "A B a b".split()], N=200)
cols = ["#1f5f9e", "#b03a2e", "#c27c0e"]
tt = np.linspace(0, 4, len(P))
for i in range(3):
    front = P[:, i].imag >= 0                                          # draw the part in front of the plane thicker
    ax[0].plot(P[:, i].real, tt, color=cols[i], lw=0.8, alpha=0.6)
    ax[0].plot(np.where(front, P[:, i].real, np.nan), tt, color=cols[i], lw=2.2, label=f"strand {i+1}")
for k, lab in enumerate(["$A$", "$B$", "$A^{-1}$", "$B^{-1}$"]): ax[0].text(2.25, k + 0.5, lab, fontsize=9, va="center")
ax[0].set_xlabel("position (real part)", fontsize=9); ax[0].set_ylabel("time", fontsize=9); ax[0].set_xlim(-0.6, 2.6)
ax[0].legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
ax[0].set_title("(a) $ABA^{-1}B^{-1}$ as a braid (thick: in front)", fontsize=9.5, loc="left")
for lab, ls, col in (("A B a b", "-", "#222"), ("A a B b", "--", "#888")):
    Q = BK.commutator_path([BK.TWIST[x] for x in lab.split()], N=200)
    th = lambda i, j: np.unwrap(np.angle(Q[:, j] - Q[:, i])) / (2 * np.pi)
    name = "$ABA^{-1}B^{-1}$: area 1" if lab == "A B a b" else "$AA^{-1}BB^{-1}$: area 0"
    ax[1].plot(th(0, 1) - th(0, 1)[0], th(1, 2) - th(1, 2)[0], ls=ls, color=col, lw=1.6, label=name)
ax[1].set_xlabel(r"$\theta_{12}/2\pi$", fontsize=9); ax[1].set_ylabel(r"$\theta_{23}/2\pi$", fontsize=9)
ax[1].legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2); ax[1].set_title("(b) winding angles", fontsize=9.5, loc="left")
for a in ax: a.spines[["top", "right"]].set_visible(False); a.tick_params(labelsize=8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_commutator.pdf")); fig.savefig(os.path.join(HERE, "fig_commutator.png"), dpi=150)
