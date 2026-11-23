"""The running example of the opening chain (fig:triangle; engine/study33_triangle/triangle.json from check_triangle.py, CY1-CY6).
A: three covers. The 4-cycle of CHSH (contextual for l = 0.9: empty fibre), the odd binary triangle with all pairs forced to differ
(empty fibre), and the compatible 3 x 3 x 2 triangle T (a 4-dimensional fibre). B: along a line in the fibre of T through the
maximum-entropy law (s = 0) and the leximin law (s = 1), extended to the boundary, all three observed pair tables stay exactly the same.
The value of information of a signal is constant when the signal sits in an observed context with the decision variables (Case A,
signal C) and varies when it does not (Case B, signal B); the action taken after seeing B = 0 changes at a wall."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); D = json.load(open(os.path.join(HERE, "..", "engine", "study33_triangle", "triangle.json")))
INK, MUTED, TEAL, CORAL, GOLD, BLUE = "#222222", "#666666", "#2b8a7e", "#c4473a", "#c9962b", "#1f5f9e"
plt.rcParams.update({"font.size": 7.8, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "mathtext.fontset": "cm"})
fig = plt.figure(figsize=(12, 4.2)); gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.25], wspace=0.18)
ax = fig.add_subplot(gs[0, 0]); ax.axis("off"); ax.set_xlim(-0.15, 3.4); ax.set_ylim(-0.75, 1.35)
def graph(x0, nodes, edges, col, ls, title, sub):
    P = {k: (x0 + v[0], v[1]) for k, v in nodes.items()}
    for a, b, lab in edges:
        ax.plot([P[a][0], P[b][0]], [P[a][1], P[b][1]], color=col, lw=2, ls=ls)
        if lab: ax.text((P[a][0] + P[b][0]) / 2, (P[a][1] + P[b][1]) / 2, lab, fontsize=6.6, color=col, ha="center", va="center", bbox=dict(fc="white", ec="none", pad=0.4))
    for k, (x, y) in P.items(): ax.scatter(x, y, s=170, color="white", ec=INK, zorder=3); ax.text(x, y, k, ha="center", va="center", fontsize=7, zorder=4)
    ax.text(x0 + 0.45, 1.25, title, ha="center", fontsize=7.8, fontweight="bold", color=INK); ax.text(x0 + 0.45, -0.45, sub, ha="center", va="top", fontsize=6.8, color=INK)
graph(0.05, {"$A_0$": (0, 0.9), "$B_0$": (0.9, 0.9), "$A_1$": (0.9, 0.0), "$B_1$": (0, 0.0)},
      [("$A_0$", "$B_0$", "+"), ("$B_0$", "$A_1$", "+"), ("$A_1$", "$B_1$", "$-$"), ("$B_1$", "$A_0$", "+")], CORAL, "-", "4-cycle (CHSH)",
      "correlations $(\\lambda,\\lambda,\\lambda,-\\lambda)$\n$\\lambda=0.9$: Clauser–Horne violated,\nfibre empty (contextual)\n$\\lambda=0.5$: fibre nonempty")
graph(1.2, {"A": (0, 0.0), "B": (0.9, 0.0), "C": (0.45, 0.8)}, [("A", "B", "$\\neq$"), ("B", "C", "$\\neq$"), ("A", "C", "$\\neq$")], CORAL, "-", "odd triangle",
      "binary, every pair differs:\n$A\\neq B\\neq C\\Rightarrow A=C$\nfibre empty")
graph(2.35, {"A": (0, 0.0), "B": (0.9, 0.0), "C": (0.45, 0.8)}, [("A", "B", "AB"), ("B", "C", "BC"), ("A", "C", "AC")], TEAL, "-", "triangle T (3x3x2)",
      "pair tables compatible:\nfibre is a 4-polytope;\nwhat no pair sees: 4\ninteraction coordinates")
ax.set_title("A  Can the local tables be glued?", loc="left", fontsize=8.6, fontweight="bold", color=INK)
ax = fig.add_subplot(gs[0, 1]); s = np.array(D["s"]); vA = np.array(D["voiA"]); vB = np.array(D["voiB"]); w = D["wall"]; cond = np.array(D["cond"]); t = D["t"]
act = cond > t + 1e-12
ax.axvspan(s.min(), w, color="#eef7f7", lw=0); ax.axvspan(w, s.max(), color="#fbeeee", lw=0)
ax.text((s.min() + w) / 2, 0.0405, "after seeing B = 0: act", ha="center", fontsize=6.8, color=TEAL); ax.text((w + s.max()) / 2, 0.0405, "after seeing B = 0: pass", ha="center", fontsize=6.8, color=CORAL)
ax.plot(s, vA, color=BLUE, lw=2, label="Case A: signal C (in context AC with the decision)"); ax.plot(s, vB, color=GOLD, lw=2, label="Case B: signal B (never observed with A and C together)")
ax.axvline(w, color=MUTED, lw=0.8, ls="--")
for x0, lab, mk in ((0, "maximum entropy", "o"), (1, "leximin", "D")):
    i = int(np.argmin(np.abs(s - x0)))
    for v in (vA[i], vB[i]): ax.scatter(s[i], v, color=INK, marker=mk, s=26, zorder=5)
    ax.annotate(lab, (s[i], vA[i]), xytext=(0, 7), textcoords="offset points", fontsize=6.8, color=INK, ha="center")
ax.set_xlabel("position $s$ along an interaction direction of the fibre (all three pair tables fixed)"); ax.set_ylabel("value of information")
ax.set_ylim(-0.002, 0.044); ax.set_xlim(s.min(), s.max()); ax.legend(frameon=False, fontsize=6.8, loc="center left", bbox_to_anchor=(0, 0.55))
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.set_title("B  Same observations, different worth of a signal", loc="left", fontsize=8.6, fontweight="bold", color=INK)
fig.savefig(os.path.join(HERE, "fig_triangle.png"), dpi=220, bbox_inches="tight"); print("ok")
