"""The 3 x 3 test (sec:threebythree; numbers from engine/threebythree_results.json and the TT output). A: how the leximin recursion removes
the four fibre dimensions over 600 searched margins; patterns with a level of 3, or four levels of 1, are excluded by prop:firstlevel.
B: normal-fan types of sampled chambers (2000 margins) by number of facets; the only Fano type is P^2 x P^2. C: the spectral
correspondence on example E1: eigenvalue valuations of Q/W at the positive critical point against 1/L, extrapolating to h_j - h_1."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); R = json.load(open(os.path.join(HERE, "..", "engine", "threebythree_results.json")))
INK, MUTED, GRID = "#222222", "#666666", "#d9d9d9"; RED, BLUE, AMBER, TEAL = "#b3261e", "#1f5f9e", "#c27c0e", "#2b8a7e"
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.5), gridspec_kw=dict(wspace=0.34))
ax = axs[0]; pats = {eval(k): v for k, v in R["patterns"].items()}
order = [(2, 1, 1), (2, 2), (4,), (1, 1, 1, 1), (3, 1), (1, 3), (1, 1, 2)]
vals = [pats.get(p, 0) for p in order]; cols = [BLUE if p in pats else GRID for p in order]
ax.bar(range(len(order)), vals, color=cols)
for i, (p, v) in enumerate(zip(order, vals)):
    ax.text(i, v + 6, str(v) if v else "excluded", ha="center", fontsize=6.3, color=INK if v else MUTED, rotation=0 if v else 90, va="bottom")
ax.set_xticks(range(len(order))); ax.set_xticklabels(["-".join(map(str, p)) for p in order], fontsize=6.6)
ax.set_xlabel("dimensions removed by successive levels"); ax.set_ylabel("margins (of 600)")
ax.set_title("A  Selection depth: never four single steps", loc="left", fontsize=8.2, fontweight="bold", color=INK)
ax = axs[1]
fan = [(6, 9, True, 107), (7, 12, False, 210), (7, 12, False, 116), (7, 12, False, 160), (8, 15, False, 104), (8, 15, False, 226),
       (8, 15, False, 180), (8, 15, False, 286), (9, 18, False, 157), (9, 18, False, 102), (9, 18, False, 64), (9, 18, False, 65), (9, 18, False, 223)]
x = 0; ticks = []; labs = []
for nfac in (6, 7, 8, 9):
    group = [t for t in fan if t[0] == nfac]; start = x
    for t in group:
        ax.bar(x, t[3], color=RED if t[2] else BLUE, width=0.8); x += 1
    ticks.append((start + x - 1) / 2); labs.append(f"{nfac} facets\n{group[0][1]} vertices"); x += 1.6
ax.text(0, 112, "P2 x P2\n(Fano)", ha="center", fontsize=6.3, color=RED)
ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=6.4); ax.set_ylabel("sampled margins (of 2000)")
ax.set_title("B  13 chamber types; one is Fano", loc="left", fontsize=8.2, fontweight="bold", color=INK)
ax = axs[2]; sp = R["spectral"]["E1"]; Ls = [60, 120, 240]; V = np.array([sp["by_L"][str(L)] for L in Ls]); inv = np.array([1 / L for L in Ls])
for i in range(4):
    ax.plot(inv, V[:, i], "o-", color=[BLUE, BLUE, TEAL, AMBER][i], ms=3, lw=1.1)
    fit = np.polyfit(inv, V[:, i], 1); xs = np.linspace(0, inv.max(), 50); ax.plot(xs, np.polyval(fit, xs), color=MUTED, lw=0.6, ls=":")
for t, lab in zip(sorted(set(round(v, 6) for v in sp["target"])), ["h1 - h1 = 0 (level 1 removes 2)", "h2 - h1 = 1/24 (level 2 removes 1)", "h3 - h1 = 1/16 (level 3 removes 1)"]):
    ax.axhline(t, color=GRID, lw=0.8); ax.text(0.0165, t - 0.004, lab, fontsize=6.2, color=INK, ha="right", va="top")
ax.set_xlabel("1 / L"); ax.set_ylabel("eigenvalue valuation of Q/W"); ax.set_xlim(-0.0005, inv.max() * 1.05)
ax.set_title("C  Spectrum tracks the levels (E1)", loc="left", fontsize=8.2, fontweight="bold", color=INK)
fig.savefig(os.path.join(HERE, "fig_threebythree.png"), dpi=220, bbox_inches="tight"); print("ok")
