"""Figure for Phase III, Stage 4 (strata). (a) the bowtie's strata: latent rank and the tests' degrees of freedom; (b) lobes: which
coincidences cost test degrees of freedom on four pair graphs; (c) r = 0: the misfit of the explanation with triangle B's labels flipped,
against r: zero only at equal weights."""
import json, os, sys, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from pce import cover_strata as STR
D = json.load(open(os.path.join(HERE, "..", "lab", "strata.json")))
BLUE, ORANGE, VIOLET, AQUA, GREY, CRIT = "#2a78d6", "#eb6834", "#6250d6", "#1baf7a", "#8c8c8c", "#d03b3b"; INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.0), gridspec_kw=dict(width_ratios=[1.15, 1.25, 1]))
a = ax[0]; T = D["bowtie_strata"]; x = np.arange(len(T))
a.bar(x - 0.2, [t["latent_rank"] for t in T], width=0.4, color=BLUE, label="latent rank (of 6)"); a.bar(x + 0.2, [t["tests_df"] for t in T], width=0.4, color=ORANGE, label="tests' df (of 2)")
a.set_xticks(x); import re
lab = lambda st: re.sub(r"v_(\d)", r"$v_\1$", st).replace(" = 0", "=0")
a.set_xticklabels([lab(t["stratum"]) for t in T], rotation=40, ha="right", fontsize=7.5)
a.legend(frameon=False, fontsize=7.5, loc="upper right"); a.set_ylim(0, 7.5)
a.set_title("(a) the bowtie with its triples, stratum by stratum:\nonly the shared variable 2 and the merge cost", fontsize=9)
a = ax[1]; a.axis("off"); a.set_xlim(0, 8.4); a.set_ylim(-0.6, 3.2)
graphs = [("bowtie", {0: (0, 1), 1: (0, 0), 2: (0.8, 0.5), 3: (1.6, 1), 4: (1.6, 0)}, [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (2, 4)], 2),
          ("windmill", {0: (1, 0.5), 1: (0.3, 1.1), 2: (0.2, 0.4), 3: (1.7, 1.1), 4: (1.8, 0.4), 5: (0.7, -0.2), 6: (1.3, -0.2)}, [(0, 1), (1, 2), (0, 2), (0, 3), (3, 4), (0, 4), (0, 5), (5, 6), (0, 6)], 0),
          ("two 4-cycles", {0: (0.9, 0.5), 1: (0.4, 1), 2: (0, 0.5), 3: (0.4, 0), 4: (1.4, 1), 5: (1.8, 0.5), 6: (1.4, 0)}, [(0, 1), (1, 2), (2, 3), (0, 3), (0, 4), (4, 5), (5, 6), (0, 6)], 0),
          ("triangle + 4-cycle", {0: (0.8, 0.5), 1: (0.2, 1), 2: (0.2, 0), 3: (1.3, 1), 4: (1.8, 0.5), 5: (1.3, 0)}, [(0, 1), (1, 2), (0, 2), (0, 3), (3, 4), (4, 5), (0, 5)], 0)]
L = D["lobes"]; keymap = {"bowtie": "bowtie", "windmill": "windmill", "two 4-cycles": "two 4-cycles at 0", "triangle + 4-cycle": "triangle + 4-cycle at 0"}
for k, (nm, pos, E, c) in enumerate(graphs):
    ox = (k % 2) * 4.2 + 0.3; oy = 1.7 if k < 2 else 0.0
    for i, j in E: a.plot([ox + pos[i][0], ox + pos[j][0]], [oy + pos[i][1], oy + pos[j][1]], color=INK2, lw=1)
    for i, p in pos.items(): a.scatter(ox + p[0], oy + p[1], s=26 if i != c else 60, color=CRIT if i == c else BLUE, zorder=3)
    info = L[keymap[nm]] if nm != "triangle + 4-cycle" else L["triangle + 4-cycle at 0"]
    a.text(ox + 2.0, oy + 0.6, f"{nm}\nlobes at red: {info['lobes']}\ndf {info['df']} $\\to$ {info['df_at_centre']}", fontsize=7.5)
a.set_title("(b) a coincidence at the red vertex costs max(0, lobes $-1$) df:\nlobes = non-bipartite pieces meeting there (cut alone is not enough)", fontsize=9)
a = ax[2]; m = 6; F = [tuple(e) for e in sorted({e for W in [(0, 1, 2), (3, 4, 5)] for e in itertools.combinations(W, 2)})] + [(0, 1, 2), (3, 4, 5)]
v = np.array([0.4, -0.3, 0.5, 0.35, -0.45, 0.3]); rs = np.linspace(-1, 1, 401); flip = np.r_[1, 1, 1, -1, -1, -1, 1]
mis = [np.abs(STR.phi(m, F, np.r_[v, r] * flip) - STR.phi(m, F, np.r_[v, r])).max() for r in rs]
a.plot(rs, mis, color=VIOLET, lw=2); a.scatter([0], [0], color=CRIT, zorder=5, s=25)
a.text(0.05, 0.002, "r = 0: the flipped labelling\nfits too (4 explanations)", color=CRIT, fontsize=7.5)
a.set_xlabel("$r=t-1/t$ (0 = equal weights)"); a.set_ylabel("misfit of triangle B's flipped labels")
a.set_title("(c) equal weights un-align labels\nthat travelled through $r$", fontsize=9)
plt.tight_layout(); fig.savefig(os.path.join(HERE, "fig_strata.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_strata.png"), dpi=150, bbox_inches="tight"); print("ok")
