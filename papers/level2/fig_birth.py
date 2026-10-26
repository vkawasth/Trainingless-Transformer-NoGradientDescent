"""Figure for Section 15 (birth of contextuality): the birth curve and where contextual models are found."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); R = json.load(open(os.path.join(HERE, "birth_results.json")))
BLUE, ORANGE, GREY = "#3a6ea5", "#c0622f", "#8c8c8c"
fig, ax = plt.subplots(1, 2, figsize=(9, 3.0))
c = np.array(R["curve"]); t = np.linspace(0, 0.6, 200)
ax[0].plot(t, 2 * np.sqrt(1 + np.sin(t) ** 2), color=BLUE, label="2 sqrt(1 + sin^2 t)"); ax[0].plot(c[:, 0], c[:, 1], "o", color=ORANGE, ms=4, label="computed")
ax[0].plot(t, 2 + t ** 2, color=GREY, ls="--", lw=1, label="2 + t^2"); ax[0].axhline(2, color="k", lw=0.6)
ax[0].set_xlabel("reflection angle t on both sides"); ax[0].set_ylabel("largest CHSH value"); ax[0].legend(fontsize=7, frameon=False)
ax[0].set_title("(a) leaving a birth point", fontsize=9)
nb = R["nearby"]; labs = ["birth points", "singular,\nnot birth", "smooth"]; vals = [nb["birth"]["share"], nb["singular_not_birth"]["share"], nb["smooth"]["share"]]
ax[1].bar(labs, vals, color=[ORANGE, BLUE, GREY]); ax[1].set_ylabel("share of nearby models violating CHSH")
for i, v in enumerate(vals): ax[1].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
ax[1].set_title(f"(b) near the classical points of Rep$_4$ ({R['rep4']['points']} points)", fontsize=9)
for a in ax: a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_birth.pdf")); fig.savefig(os.path.join(HERE, "fig_birth.png"), dpi=150)
