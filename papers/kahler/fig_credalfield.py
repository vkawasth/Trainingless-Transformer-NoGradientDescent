"""Figure for 'The conditional credal field': a melatonin-flavoured toy (four pathways, a 120-minute window).
  (a) the field: each pathway's credal band for P(asleep by the deadline | pathway's evidence), over time
  (b) its braid: strands ordered by the central law; crossings certified by the data (filled) or left undecided (open, 'pre-crossings')
  (c) relational data decide a crossing the bands leave open: the field projected on two pathways, without and with a three-way datum
  (d) coordinated poisoning at the deadline: hiding the receptor pathway's lead costs little and raises no contradiction; reversing it
      costs more and an independent assay exposes it
All numbers come from pce/credalfield.py (palette validated: blue, amber, purple, green, adjacent CVD dE >= 14.8)."""
import os, sys, itertools
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from pce import credalfield as CF
COL = ["#1f5f9e", "#c27c0e", "#8e3b8e", "#2e7d32"]; INK, MUTED, GRID = "#222222", "#666666", "#dddddd"
M = CF.MODEL; k, q, p = M["k"], M["q"], M["p"]; T = CF.T
C = np.array([CF.centres(t) for t in T]); W = np.array([CF.halfwidths(t) for t in T])
seq = {(i, j): [CF.order_decided(k, q, p, *CF.data_at(t), i, j) for t in T] for i, j in itertools.combinations(range(k), 2)}
XT = [0, 30, 60, 90, 120]; XL = ["8:00", "8:30", "9:00", "9:30", "10:00 pm"]
fig = plt.figure(figsize=(13, 9)); gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.22)

# (a)
a = fig.add_subplot(gs[0, 0])
for i in range(k):
    a.fill_between(T, C[:, i] - W[:, i], C[:, i] + W[:, i], color=COL[i], alpha=0.22, lw=0)
    a.plot(T, C[:, i], color=COL[i], lw=1.6); a.text(122, C[-1, i], CF.NAMES[i], color=INK, fontsize=8.5, va="center")
a.set_xlim(0, 150); a.set_xticks(XT); a.set_xticklabels(XL, fontsize=8); a.tick_params(labelsize=8)
a.set_ylabel("P(asleep by 10 pm | pathway's evidence)", fontsize=9); a.set_title("(a) the credal field: one band per pathway", fontsize=10, loc="left")

# (b)
b = fig.add_subplot(gs[0, 1])
rank = np.argsort(np.argsort(-C, axis=1), axis=1)                              # rank 0 = highest
Y = 3 - rank
for i in range(k): b.plot(T, Y[:, i], color=COL[i], lw=2.2, solid_capstyle="round")
for i in range(k): b.text(122, Y[-1, i], CF.NAMES[i], fontsize=8.5, va="center", color=INK)
ncert = nund = 0
for (i, j), s in seq.items():
    d = np.sign(C[:, i] - C[:, j]); sw = np.where(d[:-1] != d[1:])[0]
    for t0 in sw:
        before = [v for v in s[:t0 + 1] if v != 0]; after = [v for v in s[t0 + 1:] if v != 0]
        others = [u for u in sw if u != t0 and ((before and u < t0) or (after and u > t0))]
        lb = max([t for t in range(t0 + 1) if s[t] != 0], default=None); la = min([t for t in range(t0 + 1, len(T)) if s[t] != 0], default=None)
        n_between = sum(1 for u in sw if lb is not None and la is not None and lb <= u < la)
        certified = lb is not None and la is not None and s[lb] != s[la] and n_between == 1
        x = T[t0] + (T[t0 + 1] - T[t0]) * abs(d[t0]) / 2; yv = (Y[t0, i] + Y[t0, j] + Y[t0 + 1, i] + Y[t0 + 1, j]) / 4
        if certified: b.plot(x, yv, "o", ms=9, color=INK, zorder=5); ncert += 1
        else: b.plot(x, yv, "o", ms=11, mfc="white", mec=INK, mew=1.4, zorder=5); b.text(x, yv, "?", ha="center", va="center", fontsize=8, zorder=6); nund += 1
b.plot([], [], "o", color=INK, label="crossing certified by the data"); b.plot([], [], "o", mfc="white", mec=INK, label="pre-crossing: order not decided")
b.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=2); b.set_yticks([]); b.set_xlim(0, 150); b.set_xticks(XT); b.set_xticklabels(XL, fontsize=8)
b.set_title("(b) its braid: which pathway leads, and when", fontsize=10, loc="left")

# (c)
c = fig.add_subplot(gs[1, 0]); t = 40.0; aa, ee = CF.data_at(t); i, j = 1, 3
rel = [(1, 3, 1, 0, 1, 0.0, 0.005), (1, 3, 0, 1, 1, 0.02, 1.0)]
P0 = CF.projection(k, q, p, aa, ee, i, j); P1 = CF.projection(k, q, p, aa, ee, i, j, rel=rel)
c.add_patch(Polygon(P0, closed=True, fc=GRID, ec=MUTED, lw=1, label="pathway data only (bands overlap)"))
c.add_patch(Polygon(P1, closed=True, fc=COL[3], alpha=0.35, ec=COL[3], lw=1.2, label="plus one three-way datum"))
lo, hi = P0.min() - 0.03, P0.max() + 0.03; c.plot([lo, hi], [lo, hi], "--", color=INK, lw=1, label="equal (crossing point)")
c.set_xlim(lo, hi); c.set_ylim(lo, hi); c.set_aspect("equal"); c.tick_params(labelsize=8)
c.set_xlabel("light suppression", fontsize=9); c.set_ylabel("arousal", fontsize=9); c.legend(fontsize=8, frameon=False, loc="lower right")
c.set_title("(c) relational data decide an order the bands leave open (8:40 pm)", fontsize=10, loc="left")

# (d)
dax = fig.add_subplot(gs[1, 1]); a0, e0 = CF.data_at(120.0); L = (a0 - e0) / p; U = (a0 + e0) / p; R, S = 0, 2
g = L[R] - U[S]; true_c = CF.centres(120.0)[R]; eta = 0.01
rows = [("honest data", 0, 0, False, "receptor leads: certified"),
        (f"coalition hides it: each source shifts {g / 2:.3f}", g / 2, g / 2, False, "undecided, and no contradiction"),
        ("same, plus an independent assay", g / 2, g / 2, True, "assay re-certifies the lead"),
        ("coalition reverses it: each shifts 0.080", 0.08, 0.08, True, "assay contradicts: exposed")]
for r, (lab, sr, ss, anchor, out) in enumerate(rows):
    y = 3 - r; rl, ru = L[R] - sr, U[R] - sr; sl, su = L[S] + ss, U[S] + ss
    dax.add_patch(Rectangle((rl, y + 0.04), ru - rl, 0.22, color=COL[R], alpha=0.85)); dax.add_patch(Rectangle((sl, y - 0.26), su - sl, 0.22, color=COL[S], alpha=0.85))
    if anchor: dax.add_patch(Rectangle((true_c - eta, y + 0.04), 2 * eta, 0.22, fill=False, ec=INK, lw=1.6, hatch="////"))
    dax.text(0.535, y + 0.42, lab, fontsize=8.5, color=INK); dax.text(0.76, y, out, fontsize=8.5, color=INK, va="center")
dax.set_xlim(0.53, 1.0); dax.set_ylim(-0.6, 3.8); dax.set_yticks([]); dax.tick_params(labelsize=8)
dax.set_xticks([0.55, 0.6, 0.65, 0.7, 0.75]); dax.set_xlabel("P(asleep by 10 pm | pathway's evidence), at 10 pm", fontsize=9)
dax.plot([], [], "s", color=COL[R], label="receptor signal"); dax.plot([], [], "s", color=COL[S], label="sleep pressure")
dax.add_patch(Rectangle((0, 0), 0, 0, fill=False, ec=INK, hatch="////", label="independent assay")); dax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=3)
dax.set_title("(d) coordinated poisoning: hiding is cheap and silent; reversing is exposed", fontsize=10, loc="left")
for x in (a, b, c, dax): x.spines[["top", "right"]].set_visible(False)
b.spines[["left"]].set_visible(False); dax.spines[["left"]].set_visible(False)
fig.savefig(os.path.join(HERE, "fig_credalfield.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_credalfield.png"), dpi=150, bbox_inches="tight")
print("crossings certified", ncert, "undecided", nund)
