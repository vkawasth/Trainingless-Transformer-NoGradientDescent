"""Figure for the note "Conditioned states as toric Kahler data": the F_1 polygon at two blow-up sizes, with the leading-order candidates and
the valuations of the critical points of W."""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import mpmath as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
BLUE, ORANGE, VIOLET, AQUA, GREY, INK2 = "#2a78d6", "#eb6834", "#6250d6", "#1baf7a", "#8c8c8c", "#52514e"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
mp.mp.dps = 80
def crit(a):
    T = mp.mpf(10) ** -60; out = []
    for w in mp.polyroots([1, 0, 0, -mp.sqrt(T), -T ** (1 - mp.mpf(a))], maxsteps=400, extraprec=400):
        y2 = w * w; y1 = mp.sqrt(T) / w; u = (float(mp.log(abs(y1)) / mp.log(T)), float(mp.log(abs(y2)) / mp.log(T)))
        real = abs(mp.im(y1)) < 1e-50; pos = real and mp.re(y1) > 0 and mp.re(y2) > 0; out.append((u, "positive" if pos else ("real, $y_1<0$" if real else "complex")))
    return out
from pce import kahlerbridge as KB
fig = plt.figure(figsize=(10.6, 8.2)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.82])
ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
for axx, a in zip(ax[:2], (0.2, 0.6)):
    V = [(0, 0), (1, 0), (a, 1 - a), (0, 1 - a)]
    axx.add_patch(Polygon(V, closed=True, fc="#f3f3f3", ec=INK2, lw=1.2))
    if a < 1 / 3:
        axx.plot([1 / 3], [1 / 3], "D", color=ORANGE, ms=11, mfc="none", mew=2, label="leading terms balance (candidates)")
    else:
        axx.plot([(1 - a) / 2, a], [(1 - a) / 2] * 2, color=ORANGE, lw=4, alpha=0.7, label="leading terms balance (candidates)")
        axx.plot([(1 - a) / 2, a], [(1 - a) / 2] * 2, "o", ms=8, mfc="white", mec=ORANGE, mew=2, label="endpoints: no positive balance")
    seen = set()
    for u, kind in crit(a):
        key = (round(u[0], 3), round(u[1], 3), kind)
        if key in seen: continue
        seen.add(key); col = {"positive": BLUE, "real, $y_1<0$": VIOLET, "complex": GREY}[kind]
        off = {"positive": (0, 0), "real, $y_1<0$": (0.012, 0.012), "complex": (-0.012, -0.012)}[kind]
        axx.plot(u[0] + off[0], u[1] + off[1], "o", color=col, ms=7, label=f"critical point: {kind}")
    axx.set_xlim(-0.05, 1.05); axx.set_ylim(-0.05, 1.0); axx.set_aspect("equal"); axx.set_xlabel("$u_1$"); axx.set_ylabel("$u_2$")
    axx.text(0.5, -0.035, "$p_2\\geq0$", ha="center", fontsize=8, color=INK2); axx.text(0.02, (1 - a) + 0.015, "$p_2\\leq1-\\alpha$", fontsize=8, color=INK2)
    axx.legend(frameon=False, fontsize=7.4, loc="upper right")
    axx.set_title(f"({'a' if a < 0.5 else 'b'}) $F_1$ with $\\alpha={a:g}$: " + ("one candidate, and a signed\ncritical point off it" if a < 1 / 3 else
                  "a segment of candidates;\nhigher order selects one point"), fontsize=9.5)
axx = ax[2]; v, lam = KB.F1(0.6); ramp = ["#b7d3f2", "#7aaee6", "#2a78d6", "#123f7a"]
for col, (h, lt) in zip(ramp, ((0.0, -200.0), (0.2, -200.0), (0.3, -200.0), (0.45, -100.0))):
    U1, U2, w = KB.tropical_grid(v, lam, lt, h, box=((-0.05, 1.05), (-0.05, 0.5)), n=1101); m = w.sum(1); a1 = U1[:, 0]
    axx.plot(a1, m / (m.sum() * (a1[1] - a1[0])), color=col, lw=2, label=f"$h={h:g}$" + {0.0: " (uniform on $Q$)", 0.2: " $=h^*$ (the segment)", 0.3: " (next level)", 0.45: " (at $u^*$)"}[h])
axx.axvline(0.4, color=INK2, lw=0.8, ls=":"); axx.set_xlim(-0.05, 1.05); axx.set_ylim(0, 8); axx.set_xlabel("$u_1=\\log y_1/\\log T$"); axx.set_ylabel("density of the $u_1$-marginal")
axx.legend(frameon=False, fontsize=7.4, loc="upper right")
axx.set_title("(c) $F_1$, $\\alpha=0.6$: the thimble law with $\\hbar=T^h$;\nits support shrinks to $u^*_1=0.4$ as $h$ grows", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_kahler.pdf")); fig.savefig(os.path.join(HERE, "fig_kahler.png"), dpi=150); plt.close(fig)
