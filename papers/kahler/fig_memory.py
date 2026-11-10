"""Figures for 'The memory at work' (pce/dmledger, pce/poolcond):
  fig_memory_walks    five conditions x the five relations of B and C: the normalised walk of one history (seed 7), the 5-95% envelope of
                      200 honest histories, and the family-wise boundary
  fig_memory_network  the decision maker's relation network at episode 1000 for six conditions (one history each): node colour = lean
                      walk (orange towards X, blue away), grey edges = co-revision walk above 2, arrows = lead-lag walk above 2
  fig_poolcond        the planted-evidence stream: weight of the planting source under Bayesian model averaging, and P(X) under three
                      ways of combining pooling and conditioning, with and without planting
Also prints the worked episode used in the text."""
import os, sys, json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from pce import dmledger as DL, poolcond as PO

E = 1000; SEED = 7
S = json.load(open(os.path.join(HERE, "..", "engine", "dmledger_summary.json"))); Bd = S["bounds"]
INK, MUTED, LINE, BAND, BOUND = "#222222", "#666666", "#1f5f9e", "#d9d9d9", "#b03a2e"
pBC = DL.PAIRS.index((1, 2)); lBC = DL.OPAIRS.index((1, 2))
REL = [("tilt", 1, "lean of B"), ("corev", pBC, "co-revision B-C"), ("order", pBC, "order B before C"), ("leadlag", lBC, "lead-lag B -> C"), ("anchor", 1, "anchor of B")]
CONDS = [("bias", "B leans (alone)"), ("exposure", "uneven news (honest)"), ("cascade", "cascade (honest)"), ("lockstep", "lockstep coalition"), ("staggered", "staggered coalition")]

# honest envelope
H = [DL.walks(DL.increments(DL.ledger("honest", E, sd), E)) for sd in range(40001, 40201)]
env = {f: (np.percentile([W[f][:, j] for W in H], 5, axis=0), np.percentile([W[f][:, j] for W in H], 95, axis=0)) for f, j, _ in REL}

fig, ax = plt.subplots(len(CONDS), len(REL), figsize=(12.5, 9.2), sharex=True)
t = np.arange(1, E + 1)
for r, (sc, title) in enumerate(CONDS):
    W = DL.walks(DL.increments(DL.ledger(sc, E, SEED), E))
    for c, (f, j, lab) in enumerate(REL):
        a = ax[r, c]; lo, hi = env[f]
        a.fill_between(t, lo, hi, color=BAND, lw=0, label="honest 5-95%")
        a.axhline(Bd[f], color=BOUND, ls="--", lw=0.9); a.axhline(-Bd[f], color=BOUND, ls="--", lw=0.9)
        a.plot(t, W[f][:, j], color=LINE, lw=1.3)
        a.set_ylim(-12, 30 if f in ("tilt", "order", "leadlag", "corev") else 12); a.tick_params(labelsize=7)
        a.spines[["top", "right"]].set_visible(False)
        if r == 0: a.set_title(lab, fontsize=9)
        if c == 0: a.set_ylabel(title, fontsize=8.5)
        if r == len(CONDS) - 1: a.set_xlabel("episode", fontsize=8)
fig.text(0.5, 0.005, "Blue: one history (seed 7), walk divided by the square root of the number of increments. Grey: honest 5-95% envelope. Dashed: family-wise boundary.",
         ha="center", fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(os.path.join(HERE, "fig_memory_walks.pdf")); fig.savefig(os.path.join(HERE, "fig_memory_walks.png"), dpi=150); plt.close(fig)

# networks
NET = [("honest", "honest"), ("bias", "B leans"), ("exposure", "uneven news"), ("cascade", "cascade"), ("lockstep", "lockstep coalition"), ("staggered", "staggered coalition")]
fig, ax = plt.subplots(2, 3, figsize=(10.5, 7.2))
ang = np.pi / 2 - 2 * np.pi * np.arange(DL.K) / DL.K; XY = np.c_[np.cos(ang), np.sin(ang)]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("lean", ["#2c7bb6", "#f2f2f2", "#e66101"])
for a, (sc, title) in zip(ax.flat, NET):
    W = DL.walks(DL.increments(DL.ledger(sc, E, SEED), E)); lean = W["tilt"][-1]; anc = W["anchor"][-1]
    for k, (u, v) in enumerate(DL.PAIRS):
        z = W["corev"][-1][k]
        if z > 2: a.plot(*XY[[u, v]].T, color="#9a9a9a", lw=min(1 + z / 3, 7), zorder=1, solid_capstyle="round")
    for k, (u, v) in enumerate(DL.OPAIRS):
        z = W["leadlag"][-1][k]
        if z > 2:
            a.add_patch(FancyArrowPatch(XY[u], XY[v], arrowstyle="-|>", mutation_scale=14, color=INK, lw=1.2 + z / 6, shrinkA=17, shrinkB=17,
                                        connectionstyle="arc3,rad=0.18", zorder=3))
    sc_ = a.scatter(XY[:, 0], XY[:, 1], s=900, c=np.clip(lean, -15, 15), cmap=cmap, vmin=-15, vmax=15, edgecolors=INK, linewidths=1, zorder=4)
    for i in range(DL.K):
        a.text(*XY[i], f"S{i+1}", ha="center", va="center", fontsize=9, color=INK, zorder=5)
        if abs(anc[i]) > 1.645:
            a.text(XY[i, 0], XY[i, 1] - 0.3, ("anchor +" if anc[i] > 0 else "anchor \u2212"), ha="center", va="center", fontsize=7, color=INK,
                   bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#bbbbbb", lw=0.5), zorder=6)
    a.set_title(title, fontsize=10); a.set_xlim(-1.5, 1.5); a.set_ylim(-1.55, 1.4); a.set_aspect("equal"); a.axis("off")
cb = fig.colorbar(sc_, ax=ax.ravel().tolist(), shrink=0.6, pad=0.02); cb.set_label("lean walk (towards X: orange)", fontsize=9); cb.ax.tick_params(labelsize=8)
fig.text(0.02, 0.01, "S2 = B, S3 = C. Grey edge: co-revision walk above 2; arrow: lead-lag walk above 2 (u -> v: v moves towards u after u revises); 'anchor +/-': anchor walk beyond 1.645.",
         fontsize=8, color=MUTED)
fig.savefig(os.path.join(HERE, "fig_memory_network.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_memory_network.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

# pooling stream
st = json.load(open(os.path.join(HERE, "..", "engine", "poolcond_stream.json")))
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
for key, ls in (("planted", "-"), ("honest", ":")):
    o = st[key]; lab = "with planted records" if key == "planted" else "no planting"
    ax[0].plot(o["step"], o["wK"], color=LINE, ls=ls, lw=1.6, label=lab)
ax[0].set_title("(a) weight of the planting source under model averaging", fontsize=9.5, loc="left"); ax[0].set_ylim(0, 1.05); ax[0].set_xlabel("evidence item", fontsize=9)
ax[0].legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
o = st["planted"]
for key, col, lab in (("bma", "#b03a2e", "pool, then condition (model averaging)"), ("geometric", "#c27c0e", "geometric pool"), ("fixed", LINE, "condition, then pool (fixed weights)")):
    ax[1].plot(o["step"], o[key], color=col, lw=1.6, label=lab)
pl = [s for s, p in zip(o["step"], o["planted"]) if p]; ax[1].plot(pl, [-0.04] * len(pl), "|", color=MUTED, ms=6)
ax[1].axhline(0.5, color=MUTED, lw=0.7, ls="--"); ax[1].set_ylim(-0.08, 1.05)
ax[1].set_title("(b) pooled P(X) with planted records (ticks); X is false", fontsize=9.5, loc="left"); ax[1].set_xlabel("evidence item", fontsize=9)
ax[1].legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
for a in ax: a.spines[["top", "right"]].set_visible(False); a.tick_params(labelsize=8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_poolcond.pdf")); fig.savefig(os.path.join(HERE, "fig_poolcond.png"), dpi=150); plt.close(fig)

# the worked episode: staggered coalition, seed 7, first episode where C followed B and an anchor was read
L = DL.ledger("staggered", E, SEED); by = {}
for rec in L: by.setdefault(rec[0], []).append(rec)
for e in range(E):
    recs = sorted(by[e], key=lambda r: r[2]); pos = [r[1] for r in recs]
    if recs[0][5] is not None and pos[2] == pos[1] + 1: break
inc = DL.increments([r for r in L if r[0] == e and True], E)
print("worked episode", e, "anchor", round(recs[0][5], 4))
for r in recs: print(f"  S{r[2]+1}: position {r[1]+1}, initial {r[3]:.4f}, revised {r[4]:.4f}")
row = {f: inc[f][e] for f in DL.FAMILIES}
print("  median revised", round(float(np.median([r[4] for r in recs])), 4))
print("  tilt", row["tilt"].tolist()); print("  corev B-C", round(row["corev"][pBC], 4), " order B-C", row["order"][pBC], " leadlag B->C", row["leadlag"][lBC], " anchor", row["anchor"].tolist())
