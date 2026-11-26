"""Study 85 figures (held-out seed 8502): F1 data support vs thimble events on one path, F2 the full twist as a braid, F3 a collision wall
and its two resolutions, F4 order of update steps. Synthetic data only."""
import os, sys, copy, pickle
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ENG = os.path.join(os.path.dirname(HERE), "engine"); sys.path.insert(0, ENG); os.chdir(ENG)
import lib85 as B, mirror83 as M, l2_83 as W, gen83 as G
INK, MUTED, BLUE, ORANGE, GREEN, PURPLE, GRAY = "#1F2937", "#6B7280", "#1D4ED8", "#C2410C", "#047857", "#7C3AED", "#9CA3AF"
STRAND = [BLUE, GREEN, "#B45309", INK, "#DB2777", "#0E7490"]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
SEED = 8502; SC = {f"{d['params']['cls']}{d['params']['k']}": d for d in G.build(SEED)}; KEY = "D1"; P = SC[KEY]["paths_pop"]
R = pickle.load(open(os.path.join(ENG, "study85_braid", f"check85_{SEED}.pkl"), "rb"))["R"][KEY]
def fg_at(P, s):
    k = min(int(np.floor(s)), len(P) - 2); fr = s - k; f0, g0 = W.fg(P[k]); f1, g1 = W.fg(P[k + 1]); return f0 + fr * (f1 - f0), g0 + fr * (g1 - g0)
def draw_braid(ax, perm0, steps, x0, x1, w):
    """strands = positions in the Seidel order (rank) over x; steps (x, i, j, kind) swap adjacent ranks"""
    rank = {p: q for q, p in enumerate(perm0)}; segs = {p: [(x0, rank[p])] for p in range(6)}
    for x, i, j, kind in steps:
        ri, rj = rank[i], rank[j]
        for p in range(6):
            if p not in (i, j): continue
            segs[p].append((x - w / 2, rank[p])); segs[p].append((x + w / 2, rj if p == i else ri))
        rank[i], rank[j] = rj, ri
        y = (ri + rj) / 2
        if kind == "event": ax.plot(x, y, "o", ms=7, mfc="white", mec=ORANGE, mew=1.8, zorder=5)
        else: ax.plot(x, y, "D", ms=6, mfc=PURPLE, mec="white", mew=0.8, zorder=5)
    for p in range(6):
        segs[p].append((x1, rank[p])); xs, ys = zip(*segs[p]); ax.plot(xs, ys, color=STRAND[p], lw=1.6, zorder=3)
    return rank
# ------------------------------------------------------------------ F1 ------------------------------------------------------------------
rp = B.run_path_fine(P, 20.0); perm0 = list(range(6))
for st in rp["steps"]:
    if st[4] < 0:
        rk = B.rank_of(perm0); perm0[rk[st[1]]], perm0[rk[st[2]]] = perm0[rk[st[2]]], perm0[rk[st[1]]]
steps = [(st[4], st[1], st[2], st[0]) for st in rp["steps"] if st[4] >= 0]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True, gridspec_kw=dict(height_ratios=[1.25, 1]))
rk_end = draw_braid(a1, perm0, steps, 0, 12, 0.22)
a1.set_yticks(range(6)); a1.set_yticklabels([f"{q + 1}" for q in range(6)]); a1.set_ylabel("position in the Seidel order\n(seen from $s_0$)")
for q, p in enumerate(perm0): a1.text(-0.15, q, f"$v_{p + 1}$", color=STRAND[p], ha="right", va="center", fontsize=8.5)
a1.plot([], [], "o", mfc="white", mec=ORANGE, mew=1.8, label="Stokes event: strands cross, one thimble mutates (Hurwitz move)")
a1.plot([], [], "D", mfc=PURPLE, mec="white", label="collision wall: strands touch, orthogonal thimbles swap")
a1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=1, frameon=False, fontsize=8)
a1.set_ylim(-0.6, 5.6); a1.spines[["top", "right"]].set_visible(False)
S = np.linspace(0, 12, 1201)
for h, col, lab in (("h1", BLUE, "$h_1=P(\\mathrm{below}\\mid\\mathrm{unseeded})$"), ("h3", GREEN, "$h_3=x_{12}/(x_{12}+x_{01})$")):
    bd = np.array([B.band(h, *fg_at(P, s))[:2] for s in S]); a2.fill_between(S, bd[:, 0], bd[:, 1], color=col, alpha=0.22, lw=0); a2.plot(S, bd[:, 0], color=col, lw=0.9); a2.plot(S, bd[:, 1], color=col, lw=0.9, label=f"band of {lab} over all tables with these margins")
bw = sorted({round(w["s"], 4) for h in ("h1", "h3") for w in R["bands"][h]})
for x in bw: a2.axvline(x, color=GRAY, lw=0.7, ls="-", zorder=0)
for x in R["mo_walls"]:
    a1.axvline(x, color=PURPLE, lw=0.6, ls=":", zorder=0); a2.axvline(x, color=PURPLE, lw=0.6, ls=":", zorder=0)
for e in R[20.0]["events"]: a2.plot(e["s"], 1.02, "v", color=ORANGE, ms=5, transform=a2.get_xaxis_transform(), clip_on=False)
a2.plot([], [], color=GRAY, lw=0.7, label="credal wall (a band changes formula): on a chamber wall $f_0=\\sum_J g_j$")
a2.plot([], [], color=PURPLE, lw=0.8, ls=":", label="margin-order tie (collision wall)")
a2.plot([], [], "v", color=ORANGE, label="Stokes event at $L=20$ (moves with $L$)")
a2.set_xlabel("week of the season (cumulative tables, held-out scenario D1)"); a2.set_ylabel("supported range"); a2.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=2)
a2.spines[["top", "right"]].set_visible(False); a2.set_ylim(0, 1.0)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig85_F1.pdf")); fig.savefig(os.path.join(HERE, "fig85_F1.png"), dpi=150); plt.close(fig)
# ------------------------------------------------------------------ F2 ------------------------------------------------------------------
T = M.Tracker(); c2, K2, evs, pf = B.rotation(T.vals, [np.array(x) for x in T.cls], list(B.CPRIME), 1)
fig, ax = plt.subplots(figsize=(7.2, 3.0))
draw_braid(ax, M.seidel_perm(T.vals), [(a / (2 * np.pi) * 360, i, j, "event") for a, i, j, d in evs], 0, 360, 4.0)
ax.set_xlim(-12, 372); ax.set_xticks([0, 90, 180, 270, 360]); ax.set_xlabel("rotation of the six critical values (degrees), $W\\mapsto e^{i\\alpha}W$, base point $s_0$ fixed")
ax.set_yticks(range(6)); ax.set_yticklabels([f"{q + 1}" for q in range(6)]); ax.set_ylabel("Seidel position"); ax.spines[["top", "right"]].set_visible(False)
ax.set_title("one full turn of the critical values: 30 crossings, the full twist $\\Delta^2$ in $B_6$", fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig85_F2.pdf")); fig.savefig(os.path.join(HERE, "fig85_F2.png"), dpi=150); plt.close(fig)
# ------------------------------------------------------------------ F3 ------------------------------------------------------------------
sw = R["mo_walls"][1]; DEL = 0.05; T0 = M.Tracker(); pts = [W.fg(x) for x in P]; T0.step_to(pts[0][0], pts[0][1], 20.0)
for q in range(1, int(np.floor(sw - DEL)) + 1): T0.step_to(pts[q][0], pts[q][1], 20.0)
fa, ga = fg_at(P, sw - DEL); T0.step_to(fa, ga, 20.0)
fw, gw = fg_at(P, sw); Tw = copy.deepcopy(T0); Tw.step_to(fw, gw, 20.0); v = Tw.vals; dv = np.abs(v[:, None] - v[None, :]) + 1e9 * np.eye(6); a, b = np.unravel_index(dv.argmin(), dv.shape)
fig, ax = plt.subplots(figsize=(5.6, 4.3)); NS = 80; YM = []
for eps, col, lab in ((0.0, INK, "real margins: the values meet (singular braid)"), (2e-3, BLUE, "B-field $+\\epsilon$: pass on one side ($\\sigma$)"), (-2e-3, ORANGE, "B-field $-\\epsilon$: pass on the other ($\\sigma^{-1}$)")):
    T = copy.deepcopy(T0); B.gstep(T, lambda lam: B.la_of(fa, ga, 20.0, eps * lam)); ds = []
    for q in range(1, NS + 1):
        s0_, s1_ = sw - DEL + 2 * DEL * (q - 1) / NS, sw - DEL + 2 * DEL * q / NS; f0_, g0_ = fg_at(P, s0_); f1_, g1_ = fg_at(P, s1_)
        B.gstep(T, lambda lam: B.la_of(f0_ + lam * (f1_ - f0_), g0_ + lam * (g1_ - g0_), 20.0, eps)); ds.append(T.vals[a] - T.vals[b])
    ds = np.array(ds); ax.plot(ds.real * 1e3, ds.imag * 1e6, color=col, lw=1.6, label=lab); ax.plot(ds[-1].real * 1e3, ds[-1].imag * 1e6, ">", color=col, ms=5); YM.append(np.abs(ds.imag).max() * 1e6)
ax.plot(0, 0, "o", color=PURPLE, ms=6); ax.text(0.0, 0.0, "  0", color=PURPLE, va="bottom")
ax.set_ylim(-1.3 * max(YM), 1.3 * max(YM))
ax.set_xlabel(f"Re$(v_{a + 1}-v_{b + 1})\\times10^3$"); ax.set_ylabel(f"Im$(v_{a + 1}-v_{b + 1})\\times10^6$"); ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
ax.set_title(f"collision wall at week {sw:.2f} (held-out D1): the pair is orthogonal,\nso $\\sigma$ and $\\sigma^{{-1}}$ give the same category", fontsize=8.5)
ax.spines[["top", "right"]].set_visible(False); ax.axhline(0, color=GRAY, lw=0.5, zorder=0)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig85_F3.pdf")); fig.savefig(os.path.join(HERE, "fig85_F3.png"), dpi=150); plt.close(fig)
# ------------------------------------------------------------------ F4 ------------------------------------------------------------------
KEY4 = "A1"; P4 = SC[KEY4]["paths_pop"]; p = P4[5] / P4[5].sum(); fs, gs = W.fg(P4[12])
JR = lambda x: B.J_R(x, fs); JC = lambda x: B.J_C(x, gs); q = lambda x: x[1, 2] / x[1].sum()
x = p.copy()
for _ in range(80): x = JC(JR(x))
qstar = q(x); fig, (b1, b2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
for first, col, lab in (("R", BLUE, "rows first: $J_R, J_C, J_R,\\dots$"), ("C", ORANGE, "columns first: $J_C, J_R, J_C,\\dots$")):
    y = p.copy(); qs = [q(y)]; f0s = [y.sum(1)[0]]
    for n in range(12):
        y = (JR if (n % 2 == 0) == (first == "R") else JC)(y); qs.append(q(y)); f0s.append(y.sum(1)[0])
    b1.plot(range(13), qs, "o-", color=col, ms=3, lw=1.2, label=lab); b2.semilogy(range(13), np.abs(np.array(qs) - qstar) + 1e-17, "o-", color=col, ms=3, lw=1.2)
b1.axhline(qstar, color=GRAY, lw=0.8, ls="--"); b1.text(12, qstar, "I-projection (IPF limit)", ha="right", va="bottom", fontsize=7, color=MUTED)
b1.set_xlabel("number of update steps"); b1.set_ylabel("$P(\\mathrm{below}\\mid\\mathrm{unseeded})$"); b1.legend(fontsize=7, frameon=False)
b2.set_xlabel("number of update steps"); b2.set_ylabel("distance to the limit"); b2.set_title("order matters, then washes out geometrically", fontsize=8)
for ax in (b1, b2): ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig85_F4.pdf")); fig.savefig(os.path.join(HERE, "fig85_F4.png"), dpi=150); plt.close(fig)
print("figures written; F3 pair", a + 1, b + 1, "wall", sw)
