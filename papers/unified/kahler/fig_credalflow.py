"""Credal sets under evidence (sec:credalflow). A: a credal polygon on three states carried by soft evidence (the tilt toward state 1),
with the decision wall <c, p> = 0; the polygon keeps its Hilbert shape and its vertices. B: the bracket of the payoff over K_t, the time
after which the decision is determinate, and the explicit bound of thm:crossing. C: dilation as a revealed conserved spread: the bracket of
P(X = 1) - 1/2 under evidence about a correlated signal Y tends to the conserved conditional bracket on the top cell.
Numbers computed with engine/pce/credalflow.py."""
import os, sys, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from pce import credalflow as CF
INK, MUTED, GRID = "#222222", "#666666", "#d9d9d9"; RED, BLUE, AMBER, TEAL = "#b3261e", "#1f5f9e", "#c27c0e", "#2b8a7e"
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 3, figsize=(11, 3.5), gridspec_kw=dict(width_ratios=[1.05, 1, 1], wspace=0.3))
V = [np.array(v) for v in ([0.10, 0.30, 0.60], [0.18, 0.22, 0.60], [0.12, 0.48, 0.40], [0.22, 0.38, 0.40])]
c = np.array([1.0, -0.15, -0.35]); a = np.array([1.2, 0.0, -0.4])
def xy(p): return np.array([p[1] + 0.5 * p[0], math.sqrt(3) / 2 * p[0]])      # state 0 at the top, 1 bottom right, 2 bottom left
ax = axs[0]; tri = np.array([xy(np.eye(3)[i]) for i in (0, 1, 2, 0)]); ax.plot(tri[:, 0], tri[:, 1], color=MUTED, lw=0.8)
for i, lab in enumerate(["state 1 (favoured)", "state 2", "state 3"]):
    q = xy(np.eye(3)[i]); ax.text(q[0], q[1] + (0.04 if i == 0 else -0.07), lab, ha="center", fontsize=6.6, color=MUTED)
# wall <c, p> = 0 inside the simplex
pts = []
for i, j in ((0, 1), (0, 2), (1, 2)):
    ci, cj = c[i], c[j]
    if ci * cj < 0:
        s = cj / (cj - ci); p = np.zeros(3); p[i] = s; p[j] = 1 - s; pts.append(xy(p))
pts = np.array(pts); ax.plot(pts[:, 0], pts[:, 1], color=RED, lw=1.2, ls="--"); ax.text(pts[0][0] + 0.02, pts[0][1], "decision wall", color=RED, fontsize=6.5)
from scipy.spatial import ConvexHull
cols = [MUTED, TEAL, BLUE, INK]
for t, col in zip([0, 1, 2, 3.5], cols):
    P = np.array([xy(CF.tilt(v, a, t)) for v in V]); h = ConvexHull(P); poly = P[np.r_[h.vertices, h.vertices[0]]]
    ax.fill(poly[:, 0], poly[:, 1], color=col, alpha=0.18); ax.plot(poly[:, 0], poly[:, 1], color=col, lw=1.0)
    ax.text(P[:, 0].max() + 0.03, P[:, 1].mean(), f"t = {t:g}", fontsize=6.3, color=col, va="center")
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("A  Evidence carries the credal set", loc="left", fontsize=8.2, fontweight="bold", color=INK, pad=14)
ax = axs[1]; ts = np.linspace(0, 5, 400); br = np.array([CF.bracket(V, c, a, t) for t in ts])
ax.fill_between(ts, br[:, 0], br[:, 1], color=BLUE, alpha=0.2, lw=0); ax.plot(ts, br[:, 0], color=BLUE, lw=1.4, label="lower payoff"); ax.plot(ts, br[:, 1], color=BLUE, lw=0.9, ls="--", label="upper payoff")
ax.axhline(0, color=RED, lw=0.9, ls="--"); tl = CF.last_nonpositive(V, c, a, 8.0); B = CF.crossing_bound(V, c, a)
ax.axvline(tl, color=INK, lw=0.9); ax.text(tl + 0.05, br[:, 1].max() * 0.9, f"determinate\nafter t = {tl:.2f}", fontsize=6.4)
ax.axvline(B, color=AMBER, lw=0.9, ls=":"); ax.text(B + 0.05, br[:, 0].min() * 0.7, f"bound {B:.2f}", fontsize=6.4, color=AMBER)
ax.set_xlabel("evidence (t)"); ax.set_ylabel("payoff bracket over K_t"); ax.legend(fontsize=6.3, frameon=False, loc="lower right")
ax.set_title("B  Indeterminate, then determinate", loc="left", fontsize=8.2, fontweight="bold", color=INK)
ax = axs[2]
def law(rho): q = 0.25 * (1 + rho); r = 0.25 * (1 - rho); return np.array([q, r, r, q])
Vd = [law(-0.8), law(0.8)]; cd = np.array([-0.5, -0.5, 0.5, 0.5]); ad = np.array([0.0, 1.0, 0.0, 1.0]); ts = np.linspace(0, 10, 300)
bd = np.array([CF.bracket(Vd, cd, ad, t) for t in ts])
ax.fill_between(ts, bd[:, 0], bd[:, 1], color=TEAL, alpha=0.22, lw=0); ax.plot(ts, bd[:, 0], color=TEAL, lw=1.3); ax.plot(ts, bd[:, 1], color=TEAL, lw=1.3)
ax.axhline(0.4, color=INK, lw=0.8, ls=":"); ax.axhline(-0.4, color=INK, lw=0.8, ls=":"); ax.text(10, 0.42, "conserved spread of P(X=1 | Y=1)", fontsize=6.3, ha="right")
ax.set_xlabel("evidence for Y = 1 (t)"); ax.set_ylabel("bracket of P(X=1) - 1/2"); ax.set_ylim(-0.5, 0.55)
ax.set_title("C  Dilation reveals a conserved spread", loc="left", fontsize=8.2, fontweight="bold", color=INK)
fig.savefig(os.path.join(HERE, "fig_credalflow.png"), dpi=220, bbox_inches="tight"); print("ok", tl, B)
