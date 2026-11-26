"""Figure for Study 91 (held-out 3x4 walks, seed 9102): fig91_heldout.png
Left: real critical values near the tie that carries no recorded collision (path 0, L = 20), with the substep holding a multiple caustic.
Right: collisions recorded at each tie crossed, L = 20 against L = 30, all three held-out paths."""
import os, sys, pickle
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ENG = os.path.join(HERE, "..", "engine"); sys.path.insert(0, ENG)
import lib91 as Q
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
INK, MUTED, C1, C2, C4 = "#1f2937", "#6b7280", "#1D4ED8", "#C2185B", "#E07B2E"
WD = os.path.join(ENG, "study91_3x4", "walks")
def load(p, L): return pickle.load(open(os.path.join(WD, f"walk_9102_{p}_L{L}.pkl"), "rb"))
def ties(pk): return [(k + t, nm, pr) for k in range(13) for t, nm, pr in Q.ties_on(pk["Fw"][k], pk["Gw"][k], pk["Fw"][k + 1], pk["Gw"][k + 1])]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 4.0), gridspec_kw=dict(width_ratios=[1.35, 1]))
pk = load(0, 20); T, V = pk["T"], np.asarray(pk["V"]).astype(complex); ta = ties(pk)
lo, hi = 9.50, 9.80; sel = (T >= lo) & (T <= hi); Ts, Vs = T[sel], V[sel]; rl = np.abs(Vs.imag) < 1e-8 * (1 + np.abs(Vs))
for q in range(96):
    y = np.where(rl[:, q], np.arcsinh(Vs[:, q].real), np.nan)
    if np.isfinite(y).any(): ax.plot(Ts, y, lw=0.7, color=plt.cm.tab20(q % 20))
ax.axvspan(9.650, 9.675, color=C4, alpha=0.18, lw=0)
for t, nm, pr in ta:
    if lo <= t <= hi: ax.axvline(t, color=INK, lw=0.9, ls=":")
for e in pk["ev"]:
    if lo <= e["t"] <= hi and e["kind"] == "collision":
        onT = min(abs(e["t"] - t) for t, _, _ in ta) < 1e-6; v = np.arcsinh(np.interp(e["t"], Ts, Vs[:, e["pair"][0]].real))
        ax.plot(e["t"], v, "o" if onT else "x", color=C2 if onT else C4, ms=3.5 if onT else 4, zorder=4)
    if lo <= e["t"] <= hi and e["kind"] == "caustic": ax.axvline(e["t"], color=C1, lw=0.6, alpha=0.6)
ax.set_xlim(lo, hi); ax.set_xlabel("week (path parameter $t$)"); ax.set_ylabel("real critical values, asinh scale")
ax.set_title("held-out path 0, $L=20$: tie $g_1=g_2$ (dotted) with no recorded\ncollision; shaded: the substep with a multiple caustic", fontsize=9)
# right: collisions per tie at L = 20 and L = 30
k = 0; xs, c20, c30, lab = [], [], [], []
for p in range(3):
    a, b = load(p, 20), load(p, 30); tt = ties(a)
    for t, nm, pr in sorted(tt):
        n20 = sum(abs(e["t"] - t) < 1e-6 for e in a["ev"] if e["kind"] == "collision"); n30 = sum(abs(e["t"] - t) < 1e-6 for e in b["ev"] if e["kind"] == "collision")
        xs.append(k); c20.append(n20); c30.append(n30); lab.append(p); k += 1
    k += 1.5
xs = np.array(xs); bx.bar(xs - 0.2, c20, 0.4, color=C1, label="$L=20$"); bx.bar(xs + 0.2, c30, 0.4, color=MUTED, label="$L=30$")
for x, n in zip(xs, c20):
    if n == 0: bx.annotate("0", (x - 0.2, 0.3), ha="center", color=C2, fontsize=9, fontweight="bold")
starts = [xs[[i for i, l in enumerate(lab) if l == p]] for p in range(3)]
bx.set_xticks([s.mean() for s in starts]); bx.set_xticklabels(["path 0", "path 1", "path 2"])
bx.set_ylabel("collisions recorded within $10^{-6}$ of the tie"); bx.legend(frameon=False, fontsize=8, loc="upper right")
bx.set_title("each tie crossed (45): symmetric collisions 981/981;\ntwo ties at $L=20$ with none recorded (QF3, QF4 fail)", fontsize=9)
for s in ("top", "right"): ax.spines[s].set_visible(False); bx.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig91_heldout.png"), dpi=200); plt.close(fig)
