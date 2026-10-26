"""Figure for Study 8b: each table's evidence split into classical, direct influence, quantum-explainable contextuality and beyond quantum."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "..", "engine", "study8_snowqueen", "results_study8b.json")))
parts = [("lam", "explained classically ($\\lambda^*$)", "#8c8c8c"), ("DI", "direct influence", "#3a6ea5"),
         ("QC", "contextual, quantum-explainable", "#c0622f"), ("BQ", "contextual, beyond quantum", "#6b2d5c")]
fig, ax = plt.subplots(figsize=(7.2, 2.6))
labels = {"D2": "correct responses only", "D3": "all responses"}
for r, ds in enumerate(("D3", "D2")):
    left = 0.0; p = R[ds]["point"]
    for k, lab, col in parts:
        v = max(0.0, p[k]); ax.barh(r, v, left=left, color=col, height=0.55, label=lab if r == 0 else None)
        if v > 0.035: ax.text(left + v / 2, r, f"{v:.2f}", ha="center", va="center", color="white", fontsize=8)
        left += v
ax.set_yticks([0, 1]); ax.set_yticklabels([labels["D3"], labels["D2"]], fontsize=8); ax.set_xlim(0, 1); ax.set_xlabel("share of every table")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.42))
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_snowqueen2.pdf")); fig.savefig(os.path.join(HERE, "fig_snowqueen2.png"), dpi=160)
