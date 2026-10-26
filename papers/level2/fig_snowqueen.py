"""Figure for Study 8 (Snow Queen): the origin fingerprint of the two response sets, with 95% bootstrap intervals."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "..", "engine", "study8_snowqueen", "results_study8.json")))
items = [("Delta", "direct influence\n$\\sum_q s_q$"), ("c", "CbD degree $c$"), ("lam_classical", "classical $\\lambda^*$"),
         ("lam_quantum", "quantum $\\lambda_Q$")]
fig, ax = plt.subplots(figsize=(7.2, 3.2))
w = 0.36; x = np.arange(len(items))
for j, (ds, lab, col) in enumerate((("D2", "correct responses only", "#3a6ea5"), ("D3", "all responses", "#c0622f"))):
    pt = [R[ds]["point"][k] for k, _ in items]; lo = [R[ds]["ci95"][k][0] for k, _ in items]; hi = [R[ds]["ci95"][k][1] for k, _ in items]
    err = np.array([np.maximum(0, np.array(pt) - lo), np.maximum(0, np.array(hi) - pt)])
    ax.bar(x + (j - 0.5) * w, pt, w, yerr=err, capsize=3, color=col, label=lab)
    for xi, v, h in zip(x + (j - 0.5) * w, pt, hi): ax.text(xi, max(v, h) + 0.025, f"{abs(v):.2f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in items], fontsize=8); ax.set_ylim(0, 1.05); ax.set_ylabel("fraction / degree")
ax.legend(fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Snow Queen (Cervantes & Dzhafarov 2018): origin fingerprint, 95% bootstrap intervals", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_snowqueen.pdf")); fig.savefig(os.path.join(HERE, "fig_snowqueen.png"), dpi=160)
