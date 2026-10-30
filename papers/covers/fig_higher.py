"""Figure for Phase III Stage 2 and the three-type proof of concept. (a) the moment space of the standardised latent on the positivity
interval: (M3, M4) with M4 >= M3^2 + 1 (the Hankel trinomial); two types are its lower boundary; (b) dimensions: two types, collinear
three types and general three types along the window ladder, with S_3 descent; (c) decision ranges for a three- and a four-variable
event under two types, collinear three types (exact) and general three types (inner bounds)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog
HERE = os.path.dirname(os.path.abspath(__file__)); D = json.load(open(os.path.join(HERE, "..", "lab", "higher.json")))
BLUE, ORANGE, VIOLET, AQUA, GREY, CRIT = "#2a78d6", "#eb6834", "#6250d6", "#1baf7a", "#8c8c8c", "#d03b3b"; INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.1))
MS = D["moment_space"]; lo, hi = MS["tau_interval"]; g = np.linspace(lo, hi, 3001); Aeq = np.vstack([np.ones_like(g), g, g ** 2])
m3s = np.linspace(MS["M3_range"][0] + 1e-4, MS["M3_range"][1] - 1e-4, 60); low, up = [], []
for m3 in m3s:
    A = np.vstack([Aeq, g ** 3]); b = [1, 0, 1, m3]
    low.append(linprog(g ** 4, A_eq=A, b_eq=b, bounds=[(0, None)] * len(g), method="highs").fun); up.append(-linprog(-g ** 4, A_eq=A, b_eq=b, bounds=[(0, None)] * len(g), method="highs").fun)
a = ax[0]; a.fill_between(m3s, low, up, color=ORANGE, alpha=0.25, label="three or more collinear types")
a.plot(m3s, m3s ** 2 + 1, color=BLUE, lw=2.2, label="two types: $M_4=M_3^2+1$ (Hankel $=0$)")
t = np.sqrt(0.4 / 0.6); r0 = t - 1 / t; a.scatter([r0], [r0 ** 2 + 1], color=INK, zorder=5, s=18); a.text(r0 + 0.02, r0 ** 2 + 1 - 0.06, "truth", fontsize=7.5)
a.plot([r0, r0], [MS["M4_given_M3"][0], MS["M4_given_M3"][1]], color=CRIT, lw=1.6); a.text(r0 + 0.03, 1.45, "$M_4$ free given $M_3$\nfor three types", color=CRIT, fontsize=7.3)
a.set_xlabel("$M_3=E\\tau^3$ (a triple window)"); a.set_ylabel("$M_4=E\\tau^4$ (a four-window)"); a.legend(frameon=False, fontsize=7.3, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
a.set_title("(a) the latent moment space: two types are\nits lower boundary; the trinomial is that boundary", fontsize=9)
a = ax[1]; lad = D["three_type"]["ladder"]; cd = D["collinear_dims"]
labels = ["pairs", "+1 triple", "+2", "+3", "+4", "all\ntriples", "full\ntable"]; gen = [17 - x for x in lad[:5]] + [17 - lad[10], 17 - lad[11]]
x = np.arange(len(labels)); a.bar(x - 0.27, gen, width=0.26, color=ORANGE, label="three types, general (17 parameters)")
two = [1, 0, 0, 0, 0, 0, 0]; a.bar(x, two, width=0.26, color=BLUE, label="two types")
a.bar([0 + 0.27, 1 + 0.27], [cd[0], cd[1]], width=0.26, color=VIOLET, label="three collinear types (sizes 3, 4, 5\nremove one each: 3, 2, 1, 0)")
a.set_xticks(x); a.set_xticklabels(labels, fontsize=7.3); a.set_ylabel("dimension of the hiding set"); a.legend(frameon=False, fontsize=7, loc="upper right")
d = D["descent"]; a.text(2.6, 1.55, f"$S_3$ descent: one four-window has a\n{14 - d['single_window_rank']}-dim fibre; two overlapping ones\nglue to a point ({d['glued_rank']} of 17)", fontsize=7.3, color=INK2)
a.set_title("(b) dimensions along the window ladder ($m=5$)", fontsize=9)
a = ax[2]
rows = [("E3 | pairs", (0.006769, 0.01525), (0.006769, 0.01525), (0.001318, 0.016042)), ("E3 | +triple", (0.013, 0.013), (0.013, 0.013), (0.013, 0.013)),
        ("E4 | pairs", (0.002135, 0.010198), (0.002135, 0.010198), (0.000424, 0.021332)), ("E4 | +triple", (0.0086, 0.0086), (0.007475, 0.0086), (0.006848, 0.009958))]
for k, (nm, r2, rc, rg) in enumerate(rows):
    y = len(rows) - 1 - k
    for off, rr, col, lab in ((0.22, r2, BLUE, "two types"), (0.0, rc, VIOLET, "three collinear (exact)"), (-0.22, rg, ORANGE, "three general (inner bound)")):
        a.plot(rr, [y + off] * 2, color=col, lw=5, solid_capstyle="butt", label=lab if k == 0 else None)
        if rr[0] == rr[1]: a.scatter([rr[0]], [y + off], color=col, s=18, zorder=4)
a.set_yticks(range(len(rows))); a.set_yticklabels([r[0] for r in rows][::-1]); a.set_xlabel("probability of the event")
a.legend(frameon=False, fontsize=7.2, loc="lower right"); a.set_title("(c) decision ranges (truth: two types): E3 = all of 0,1,2 harmed,\nE4 = all of 0..3; with the triple 012, E4 is pinned only for two types", fontsize=9)
plt.tight_layout(); fig.savefig(os.path.join(HERE, "fig_higher.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_higher.png"), dpi=150, bbox_inches="tight"); print("ok")
