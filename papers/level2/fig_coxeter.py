"""Figure for Section 15: singular points of Rep_3 and quantum gaps, graph by graph."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "coxeter_results.json")))
T = [t for t in R["all_graphs"]]
order = sorted(range(len(T)), key=lambda i: (not T[i]["chordal"], T[i]["nodes"], T[i]["edges"]))
BLUE, ORANGE = "#3a6ea5", "#c0622f"
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
x = np.arange(len(T))
share = [T[i]["obstructed"] / T[i]["points"] for i in order]; col = [BLUE if T[i]["chordal"] else ORANGE for i in order]
ax[0].bar(x, share, color=col); ch = [T[i]["chordal"] for i in order]
ax[0].plot(x[np.array(ch)], np.zeros(sum(ch)), "o", color=BLUE, ms=3, clip_on=False)
ax[0].set_xlabel("graph (all isomorphism classes on 4 and 5 vertices)"); ax[0].set_ylabel("share of obstructed points in Rep$_3$")
ax[0].set_xticks([]); ax[0].set_title("(a) singular points of the 3x3 representation scheme", fontsize=9)
gap = [T[i]["quantum_gap"] if T[i]["quantum_gap"] is not None else 0.0 for i in order]
ax[1].bar(x, gap, color=col); ax[1].plot(x[np.array(ch)], np.array(gap)[np.array(ch)], "o", color=BLUE, ms=3, clip_on=False); ax[1].set_xticks([]); ax[1].set_xlabel("same graphs, same order")
ax[1].set_ylabel("quantum - classical maximum"); ax[1].set_title("(b) quantum gap (NPA level 2)", fontsize=9)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
    a.bar([0], [0], color=BLUE, label="chordal"); a.bar([0], [0], color=ORANGE, label="not chordal"); a.legend(fontsize=7, frameon=False, loc="upper center")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_coxeter.pdf")); fig.savefig(os.path.join(HERE, "fig_coxeter.png"), dpi=150)
