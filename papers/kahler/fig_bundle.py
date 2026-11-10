"""Figure for 'What braids and knots carry': the bundle of suspicion heights (residual without each source) in three histories, with
the smallest explanation and the suspect set. Uses pce/bundlebraid."""
import os, sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from pce import bundlebraid as BB, watchtoy as WT

W = WT.setup(); g = W["gB"]
cases = [("redundant_liar", "(a) a single liar, R2"), ("cascade", "(b) liar, framers, a recruit"), ("efficient", "(c) lockstep coalition")]
HI = {"R2": "#b03a2e", "S3": "#1f5f9e", "R1": "#3b8bc9", "S4": "#c27c0e"}
fig, ax = plt.subplots(2, 3, figsize=(10.5, 5.2), sharex=True, gridspec_kw=dict(height_ratios=[2.3, 1]))
for j, (nm, title) in enumerate(cases):
    S, V, rows = BB.features(BB.history(nm, W), g); t = np.arange(len(S))
    for i, x in enumerate(BB.BASE):
        y = S[:, i] + 0.015 * (i - 3)
        ax[0, j].plot(t, y, color=HI.get(x, "#a6a6a6"), lw=2.0 if x in HI else 1.1, zorder=3 if x in HI else 2)
    lab = {}
    for i, x in enumerate(BB.BASE):
        if x in HI: lab[x] = S[-1, i]
    used = []
    for x, yv in sorted(lab.items(), key=lambda kv: kv[1]):
        gap = 0.045 * max(1.1, S.max()); yy = yv if not used or yv - used[-1] > gap else used[-1] + gap; used.append(yy)
        ax[0, j].text(20.4, yy, x, color=HI[x], va="center", fontsize=8.5)
    ax[0, j].axvline(15, color="#666", ls=":", lw=0.9)
    ax[0, j].set_title(title, fontsize=9.5, loc="left"); ax[0, j].set_xlim(0, 22.5)
    me = [r["min_expl"] for r in rows]
    ax[1, j].step(t, me, where="post", color="#222", lw=1.8)
    ax[1, j].set_ylim(-0.4, 4.6); ax[1, j].set_yticks([0, 1, 2, 3, 4]); ax[1, j].set_yticklabels(["0", "1", "2", "3", "none"])
    ax[1, j].set_xlabel("step", fontsize=9)
    sus = []
    for s in S:
        m = s.min(); sus.append(tuple(x for x, v in zip(BB.BASE, s) if v <= m + 0.01) if s.max() > 0.05 else ())
    prev = None
    for k, sset in enumerate(sus):
        if sset != prev and sset:
            ax[0, j].annotate("suspect: " + ", ".join(sset), xy=(k, S[k].min()), xytext=(k + 0.4, S[k].min() - 0.12 * (1 + 0.5 * (j == 1))),
                              fontsize=7.5, color="#333", arrowprops=dict(arrowstyle="-", color="#888", lw=0.6))
        prev = sset
ax[0, 0].set_ylabel("residual without the source", fontsize=9)
ax[1, 0].set_ylabel("smallest\nexplanation", fontsize=9)
for a in ax.flat: a.spines[["top", "right"]].set_visible(False); a.tick_params(labelsize=8)
for j in range(3): ax[0, j].set_ylim(-0.35, max(ax[0, j].get_ylim()[1], 1.1))
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_bundle.pdf")); fig.savefig(os.path.join(HERE, "fig_bundle.png"), dpi=150)
