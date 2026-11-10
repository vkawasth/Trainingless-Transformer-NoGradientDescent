"""Figure for 'Coherent miscalibration' (module pce/miscal.py).
  fig_miscal.pdf  (a) the decision maker's naive value of B along four matched paths: the coalition path A (S2 and S3 move together), the
                  honest drift B0 of the world to the coalition's own endpoint law, an honest coupling drift B1 to the same value of B, and a
                  generic honest drift B2 of matched displacement; (b) the residual of the reports together with the independent anchor
                  (X3, X5): only the coalition path disagrees with it"""
import os, sys
import numpy as np
from scipy.optimize import brentq
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from pce import miscal as MC, bridgetoy as BT, tamper as T
ROSE, INK, VIOLET, GREEN, GREY, INK2 = "#c2417a", "#22211f", "#6250d6", "#2f8f5b", "#9a9a9a", "#52514e"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
J0 = np.full(5, 0.7); field = np.array([0.15, 0, 0, 0, 0]); P = MC.ising(J0, field); _, gA, gB = BT.world()
hon = MC.reports(P); anch = [hon["S1"], hon["S4"], hon["S5"]]; AC = (2, 4)
pathA, p1 = MC.coalition_path(P, ["S2", "S3"], anch, gB, 0.35); lams = np.linspace(0, 1, 21)
lawsB0 = [(1 - l) * P + l * p1 for l in lams]
j1 = brentq(lambda j: gB @ MC.ising(np.r_[0.7, 0.7, j, 0.7, 0.7], field) - 0.35, -3, 0.7); pathB1, lawsB1 = MC.world_path(J0, np.r_[0.7, 0.7, j1, 0.7, 0.7], field)
sA = MC.stats(pathA, [BT.table(P, AC)] * 21, gB, AC)
def disp_one(j):
    pth, _ = MC.world_path(J0, np.r_[j, 0.7, 0.7, 0.7, 0.7], field, T_=1); return MC.stats(pth, [BT.table(P, AC)] * 2, gB, AC)["displacement"]
j2 = brentq(lambda j: disp_one(j) - sA["displacement"], -0.5, 0.0); pathB2, lawsB2 = MC.world_path(J0, np.r_[j2, 0.7, 0.7, 0.7, 0.7], field)
series = {"A: coalition S2 + S3": (pathA, [BT.table(P, AC)] * 21, ROSE, "-"), "B0: honest drift to the same law": ([MC.reports(p) for p in lawsB0], [BT.table(p, AC) for p in lawsB0], INK, "--"),
          "B1: honest coupling drift": (pathB1, [BT.table(p, AC) for p in lawsB1], VIOLET, "-"), "B2: generic honest drift": (pathB2, [BT.table(p, AC) for p in lawsB2], GREEN, "-")}
fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"wspace": 0.25})
for name, (path, an, col, ls) in series.items():
    v = [MC.naive_value(list(r.values()), gB)[0] for r in path]
    res = [T.residual(5, list(r.values()) + [("anchor", AC, an[t])]) for t, r in enumerate(path)]
    a.plot(range(21), v, color=col, ls=ls, lw=2 if ls == "-" else 1.6, label=name); b.plot(range(21), res, color=col, ls=ls, lw=2 if ls == "-" else 1.6)
a.axhline(0.5, color=INK2, lw=0.8, ls=":"); a.text(20, 0.52, "threshold", ha="right", fontsize=8, color=INK2)
a.set_xlabel("step"); a.set_ylabel("decision maker's value of B"); a.set_title("(a) Four matched paths: A and B0 coincide", fontsize=9.5, loc="left")
a.legend(frameon=False, fontsize=8, loc="lower left")
b.set_xlabel("step"); b.set_ylabel("residual with the anchor (X3, X5)"); b.set_title("(b) Only the independent anchor separates A from B0", fontsize=9.5, loc="left")
fig.savefig(os.path.join(HERE, "fig_miscal.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_miscal.png"), dpi=150, bbox_inches="tight")
