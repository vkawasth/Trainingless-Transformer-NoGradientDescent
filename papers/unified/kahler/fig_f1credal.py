"""Worked example of sec:credalflow on F_1: a credal Kahler class alpha in [0.36, 0.45] under evidence (engine/check_f1credal.py).
A: the interval of wall values c moves rigidly; the outcome is indeterminate exactly while it straddles c = 0 (rates), and one data
stream at N = 400 follows it. B: a linear payoff on the posterior over terms, g = p_4 - p_3, over the credal hull at N = 10: its bracket
crosses 0 inside the tropical window. C: an adversary forcing the set across the wall by s = 1, against the honest stream that keeps it
on the positive side, and what it pays."""
import os, sys, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from pce import kahlerbridge as KB, credalflow as CF
INK, MUTED, GRID = "#222222", "#666666", "#d9d9d9"; RED, BLUE, AMBER, TEAL = "#b3261e", "#1f5f9e", "#c27c0e", "#2b8a7e"
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
a = np.array([2, -1, 2, -3], float); P = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.5, 0.2, 0.3], [0.2, 0.3, 0.5]]); LP = np.log(P); LA = a @ LP
Q = np.array([0.5, 0.3, 0.2]); d = Q @ LA; H = -(LP @ Q); AL = (0.36, 0.45)
lam = {al: np.array(KB.F1(al)[1], float) for al in AL}; c0 = {al: float(-a @ lam[al]) for al in AL}; cmin, cmax = min(c0.values()), max(c0.values())
fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw=dict(wspace=0.32))
ax = axs[0]; s = np.linspace(0, 0.6, 300)
ax.fill_between(s, cmin - s * d, cmax - s * d, color=BLUE, alpha=0.18, lw=0, label="credal interval (rates)")
rng = np.random.default_rng(3); N = 400; x = rng.choice(3, size=int(0.6 * N), p=Q); S = np.cumsum(LA[x]); sd = np.arange(1, len(x) + 1) / N
ax.plot(sd, cmin - S / N, color=BLUE, lw=0.8); ax.plot(sd, cmax - S / N, color=BLUE, lw=0.8, label="one data stream, N = 400")
ax.axhline(0, color=RED, lw=1, ls="--"); ax.text(0.6, -0.035, "outcome wall (alpha = 1/3)", color=RED, fontsize=6.4, ha="right", va="top")
ax.axvspan(cmin / d, cmax / d, color=AMBER, alpha=0.12); ax.text((cmin / d + cmax / d) / 2, cmax * 0.95, f"indeterminate\n{cmin / d:.3f} < s <= {cmax / d:.3f}", ha="center", fontsize=6.4, color=AMBER)
ax.set_xlabel("evidence s = n / N"); ax.set_ylabel("wall value c = 3 alpha_eff - 1"); ax.legend(fontsize=6.2, frameon=False, loc="lower left")
ax.set_title("A  The outcome: a rigid transit", loc="left", fontsize=8.2, fontweight="bold", color=INK)
ax = axs[1]; Nn = 10; cg = np.array([0.0, 0.0, -1.0, 1.0])
def prior(al): l = Nn * lam[al]; l -= l.max(); p = np.exp(l); return p / p.sum()
V = [prior(al) for al in AL]; ss = np.linspace(0, 3, 400)
def rel(v, t): q = CF.tilt(v, -Nn * H, t); return (q[3] - q[2]) / (q[3] + q[2])   # same sign as p_4 - p_3
br = np.array([[min(rel(v, t) for v in V), max(rel(v, t) for v in V)] for t in ss])
ax.fill_between(ss, br[:, 0], br[:, 1], color=TEAL, alpha=0.25, lw=0); ax.plot(ss, br[:, 0], color=TEAL, lw=1.2); ax.plot(ss, br[:, 1], color=TEAL, lw=1.2)
w0, w1 = AL[0] / (H[3] - H[2]), AL[1] / (H[3] - H[2]); ax.axvspan(w0, w1, color=AMBER, alpha=0.12); ax.axhline(0, color=RED, lw=0.9, ls="--")
ax.text((w0 + w1) / 2, br[:, 1].max() * 0.85, f"tropical window\n{w0:.2f} < s <= {w1:.2f}", ha="center", fontsize=6.4, color=AMBER)
ax.set_xlabel("evidence s = n / N"); ax.set_ylabel("bracket of (p_4 - p_3) / (p_4 + p_3), N = 10")
ax.set_title("B  A posterior payoff: soft becomes hard", loc="left", fontsize=8.2, fontweight="bold", color=INK)
ax = axs[2]; Qh = np.array([0.3, 0.4, 0.3]); dh = Qh @ LA
def tilt_to(m):
    lo, hi = 0.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2; q = Qh * np.exp(mid * LA); q /= q.sum()
        if q @ LA < m: lo = mid
        else: hi = mid
    q = Qh * np.exp(hi * LA); return q / q.sum()
Qa = tilt_to(cmax + 0.05); da = Qa @ LA; kl = float(Qa @ np.log(Qa / Qh)); s = np.linspace(0, 1.2, 300)
ax.fill_between(s, cmin - s * dh, cmax - s * dh, color=MUTED, alpha=0.2, lw=0, label="honest stream")
ax.fill_between(s, cmin - s * da, cmax - s * da, color=RED, alpha=0.2, lw=0, label="attack")
ax.axhline(0, color=RED, lw=1, ls="--"); ax.axvline(1.0, color=INK, lw=0.8, ls=":")
ax.text(0.03, -0.42, f"attack pays {kl:.4f} per observation\n(bound {(da - dh) ** 2 / (2 * np.abs(LA).max() ** 2):.4f}); 7.1 in total at N = 200,\nwhere a likelihood-ratio test catches it 98.5% of the time", fontsize=6.2)
ax.set_xlabel("evidence s = n / N"); ax.set_ylabel("wall value c"); ax.legend(fontsize=6.2, frameon=False, loc="upper left")
ax.set_title("C  Forcing the outcome costs divergence", loc="left", fontsize=8.2, fontweight="bold", color=INK)
fig.savefig(os.path.join(HERE, "fig_f1credal.png"), dpi=220, bbox_inches="tight"); print("ok")
