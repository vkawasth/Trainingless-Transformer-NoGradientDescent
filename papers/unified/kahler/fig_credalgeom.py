"""Credal geometry, mixing and flow on the sleep world (sec:credalgeom; numbers from engine/credalgeom_results.json, check_credalgeom.py).
A: the band ladder for reaching N2 (O = 2) as conditions are added, from the sources alone and with the lab anchors: bands are undefined
from the sources once the condition includes P = 0, and a more specific band can be wider (dilation). B: the credal set morphing under the
combined flow (evidence for T = 1 plus mixing of C and T): sampled extreme laws in the plane of two conditional probabilities at t = 0, 2
and 6. C: the ceiling (Hilbert diameter) and the log-odds widths of two bands over time, for evidence only (A), mixing only (B) and both
(C): with both, the primary band widens while the ceiling falls."""
import os, json, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); R = json.load(open(os.path.join(HERE, "..", "engine", "credalgeom_results.json")))
INK, MUTED, GRID = "#222222", "#666666", "#d9d9d9"; RED, BLUE, AMBER, TEAL = "#b3261e", "#1f5f9e", "#c27c0e", "#2b8a7e"
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.6), gridspec_kw=dict(width_ratios=[1.1, 1, 1.1], wspace=0.34))
ax = axs[0]; order = ["all", "T=1", "T=1,P=0", "T=1,P=0,C=1", "P=0,C=1"]; labels = ["none", "T=1", "T=1, P=0", "T=1, P=0, C=1", "P=0, C=1"]
for k, (key, lab) in enumerate(zip(order, labels)):
    for s, col, off in (("sources", MUTED, -0.15), ("anchors", BLUE, 0.15)):
        d = R["ladder"][s][key]; y = len(order) - 1 - k + off
        if d["undefined"] in (True, "True"):
            ax.text(0.5, y, "undefined (condition can have probability 0)", color=col, fontsize=6.2, ha="center", va="center", style="italic")
        else:
            lo, hi = float(d["lo"]), float(d["hi"]); ax.plot([lo, hi], [y, y], color=col, lw=4, solid_capstyle="butt")
ax.set_yticks(range(len(order))); ax.set_yticklabels(labels[::-1]); ax.set_xlim(0, 1); ax.set_xlabel("band of P(N2 | condition)")
ax.plot([], [], color=MUTED, lw=4, label="sources only"); ax.plot([], [], color=BLUE, lw=4, label="with lab anchors"); ax.legend(fontsize=6.3, frameon=False, loc="upper right")
ax.set_title("A  Bands as conditions are added", loc="left", fontsize=8.2, fontweight="bold", color=INK)
ax = axs[1]; cells = [tuple(c) for c in R["cells"]]; V = np.array(R["verts"]); mu, nu = 0.1, 0.1; beta = 1.0
L = np.zeros((24, 24))
for i, (P_, C, T, O) in enumerate(cells):
    for j, (P2, C2, T2, O2) in enumerate(cells):
        if (P2, T2, O2) == (P_, T, O) and C2 != C: L[i, j] = mu
        if (P2, C2, O2) == (P_, C, O) and T2 != T: L[i, j] = nu
L -= np.diag(L.sum(1)); a = beta * np.array([1.0 if c[2] == 1 else 0.0 for c in cells])
from scipy.linalg import expm
def coords(y):
    f = lambda pred: sum(y[i] for i, c in enumerate(cells) if pred(*c))
    x1 = f(lambda P_, C, T, O: T == 1 and P_ == 0 and C == 1 and O == 2) / f(lambda P_, C, T, O: T == 1 and P_ == 0 and C == 1)
    x2 = f(lambda P_, C, T, O: P_ == 0 and C == 1 and O == 2) / f(lambda P_, C, T, O: P_ == 0 and C == 1)
    return x1, x2
from scipy.spatial import ConvexHull
for t, col in ((0, MUTED), (2, TEAL), (6, BLUE)):
    M = expm(t * (L + np.diag(a))); P = np.array([coords((v @ M) / (v @ M).sum()) for v in V])
    ax.scatter(P[:, 0], P[:, 1], s=6, color=col, alpha=0.7, lw=0); h = ConvexHull(P); poly = P[np.r_[h.vertices, h.vertices[0]]]
    ax.plot(poly[:, 0], poly[:, 1], color=col, lw=1.0, label=f"t = {t}")
ax.plot([0, 1], [0, 1], color=GRID, lw=0.8)
ax.set_xlabel("P(N2 | T=1, P=0, C=1)"); ax.set_ylabel("P(N2 | P=0, C=1)"); ax.legend(fontsize=6.3, frameon=False, loc="upper left")
ax.set_xlim(0.35, 0.9); ax.set_ylim(0.25, 0.85)
ax.set_title("B  The credal set morphs under the flow", loc="left", fontsize=8.2, fontweight="bold", color=INK)
ax = axs[2]; cols = dict(A=AMBER, B=MUTED, C=RED); names = dict(A="evidence only", B="mixing only", C="evidence + mixing")
for nm in ("A", "B", "C"):
    rows = R["out"][nm]; ts = [r["t"] for r in rows]; D = [r["diam"] for r in rows]
    w1 = [r["P=0,C=1"]["hi_lo"] - r["P=0,C=1"]["lo_lo"] for r in rows]
    ax.plot(ts, D, color=cols[nm], lw=1.8, label=f"ceiling, {names[nm]}"); ax.plot(ts, w1, color=cols[nm], lw=1.2, ls="--", marker="o", ms=2.5)
ax.plot([], [], color=INK, ls="--", lw=1.2, label="band P(N2 | P=0, C=1), log-odds width")
ax.set_xlabel("flow time t"); ax.set_ylabel("Hilbert diameter / band width (log-odds)"); ax.legend(fontsize=6.0, frameon=False, loc="lower left")
ax.set_ylim(1.5, 2.05)
ax.set_title("C  The band rises under a falling ceiling", loc="left", fontsize=8.2, fontweight="bold", color=INK)
fig.savefig(os.path.join(HERE, "fig_credalgeom.png"), dpi=220, bbox_inches="tight"); print("ok")
