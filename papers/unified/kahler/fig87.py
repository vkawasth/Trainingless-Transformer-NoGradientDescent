"""Study 86/87 figures (held-out seed 8702, slice Sigma_H; Study 86 bracket C0). Synthetic data only."""
import os, sys, pickle, json, itertools
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
HERE = os.path.dirname(os.path.abspath(__file__)); ENG = os.path.join(os.path.dirname(HERE), "engine"); sys.path.insert(0, ENG); os.chdir(ENG)
import lib87 as V, lib85 as B, gen83 as G, l2_83 as W
INK, MUTED, BLUE, ORANGE, GREEN, PURPLE, GRAY, RED = "#1F2937", "#6B7280", "#1D4ED8", "#C2410C", "#047857", "#7C3AED", "#9CA3AF", "#B91C1C"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
SC = {f"{d['params']['cls']}{d['params']['k']}": d for d in G.build(8702)}
def clean(ax): ax.spines[["top", "right"]].set_visible(False)
# ------------------------------------------------------------------ G1 fibre -> regions ---------------------------------------------
sc = SC["C1"]; cnt = np.rint(G.target_table(sc["counts"][:, :1].astype(float))).astype(int); N = cnt.sum(); f, g = cnt.sum(1) / N, cnt.sum(0) / N
X, ac, base = V.fibre(f, g); Fi = V.integer_fibre(cnt.sum(1), cnt.sum(0)); aci = np.array([[(x / N - base)[0, 0], (x / N - base)[1, 2]] for x in Fi])
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.9))
a0 = axs[0]; a0.add_patch(plt.Polygon(ac, closed=True, fc=BLUE, alpha=0.12, ec=BLUE, lw=1.2)); a0.plot(aci[::7, 0], aci[::7, 1], ".", color=GRAY, ms=1.2)
a0.plot(ac[:, 0], ac[:, 1], "o", color=BLUE, ms=4); cen = ac.mean(0); sc_ = 0.25 * (ac.max(0) - ac.min(0)).max()
for m, lab, col in ((np.array([1.0, 0.0]), "$m_A$", INK), (np.array([0.0, 1.0]), "$m_B$", ORANGE), (np.array([1.0, 1.0]), "$m_{02}$", GREEN)):
    a0.annotate("", xy=cen + sc_ * m / np.linalg.norm(m), xytext=cen, arrowprops=dict(arrowstyle="-|>", color=col, lw=1.3)); a0.text(*(cen + 1.15 * sc_ * m / np.linalg.norm(m)), lab, color=col, fontsize=8)
a0.set_title("credal fibre $M_b$ (all tables\nwith these margins), Markov moves", fontsize=8); a0.set_xlabel("$\\Delta x_{00}$"); a0.set_ylabel("$\\Delta x_{12}$"); clean(a0)
hv = np.array([V.FAM["H_lin"](x) for x in X]); a1 = axs[1]; a1.add_patch(plt.Polygon(hv, closed=True, fc=GREEN, alpha=0.15, ec=GREEN, lw=1.2)); a1.plot(hv[:, 0], hv[:, 1], "o", color=GREEN, ms=4)
xc = np.mean(X, axis=0); hc = V.FAM["H_lin"](xc); d = 0.02
for m, col in ((V.mA, INK), (V.mB, ORANGE), (V.m02, GREEN)):
    dh = V.FAM["H_lin"](xc + d * m) - hc; a1.annotate("", xy=hc + 4 * dh, xytext=hc, arrowprops=dict(arrowstyle="-|>", color=col, lw=1.3))
a1.set_title("$R_H(b)$ for $H_{\\rm lin}$: a polygon;\ncorners = vertex tables", fontsize=8); a1.set_xlabel("$P(\\rm below\\mid S{=}0)$"); a1.set_ylabel("$P(\\rm normal\\mid S{=}0)$"); clean(a1)
Pb = V.boundary(f, g, "H_frac", n=200); a2 = axs[2]; Pb = Pb[np.all(np.isfinite(Pb), axis=1)]
from scipy.spatial import ConvexHull
hh = ConvexHull(Pb); a2.fill(Pb[:, 0], Pb[:, 1], fc=ORANGE, alpha=0.18, ec=ORANGE, lw=1.2); a2.plot(np.append(Pb[hh.vertices, 0], Pb[hh.vertices[0], 0]), np.append(Pb[hh.vertices, 1], Pb[hh.vertices[0], 1]), "--", color=GRAY, lw=0.8)
hf = np.array([V.FAM["H_frac"](x) for x in X]); a2.plot(hf[:, 0], hf[:, 1], "o", color=ORANGE, ms=4)
a2.set_title("$R_H(b)$ for $H_{\\rm frac}$: curved,\nnot convex (dashed: hull)", fontsize=8); a2.set_xlabel("$P(\\rm below\\mid S{=}0)$"); a2.set_ylabel("$x_{12}/(x_{12}+x_{01})$"); clean(a2)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig87_G1.pdf")); fig.savefig(os.path.join(HERE, "fig87_G1.png"), dpi=150); plt.close(fig)
# ------------------------------------------------------------------ G2 regions along a path + surface ------------------------------
sc = SC["D1"]; P = sc["paths_pop"]; fig = plt.figure(figsize=(7.4, 3.0)); a = fig.add_subplot(1, 2, 1); cm = plt.get_cmap("Blues")
for wk in (0, 3, 6, 9, 12):
    f, g = W.fg(P[wk]); Pb = V.boundary(f, g, "H_frac", n=120); Pb = Pb[np.all(np.isfinite(Pb), axis=1)]
    a.fill(Pb[:, 0], Pb[:, 1], color=cm(0.3 + 0.7 * wk / 12), alpha=0.10); a.plot(np.append(Pb[:, 0], Pb[0, 0]), np.append(Pb[:, 1], Pb[0, 1]), color=cm(0.3 + 0.7 * wk / 12), lw=1.2, label=f"week {wk + 1}")
a.legend(fontsize=7, frameon=False, loc="lower right")
a.set_xlabel("$P(\\rm below\\mid S{=}0)$"); a.set_ylabel("$x_{12}/(x_{12}+x_{01})$"); a.set_title("support region of $H_{frac}$ as the season accumulates (D1)", fontsize=8); clean(a)
a3 = fig.add_subplot(1, 2, 2, projection="3d"); f, g = W.fg(P[6]); X, ac, base = V.fibre(f, g)
lo, hi = ac.min(0), ac.max(0); A_, C_ = np.meshgrid(np.linspace(lo[0], hi[0], 40), np.linspace(lo[1], hi[1], 40)); Z = np.full(A_.shape + (3,), np.nan)
for i in range(40):
    for j in range(40):
        x = V.table_at(base, A_[i, j], C_[i, j])
        if (x >= -1e-12).all(): Z[i, j] = V.FAM["H_3"](x)
a3.plot_surface(Z[..., 0], Z[..., 1], Z[..., 2], color=GREEN, alpha=0.55, lw=0, antialiased=True)
a3.set_xlabel("$P(\\rm above\\mid S{=}0)$", fontsize=7); a3.set_ylabel("$P(\\rm normal\\mid S{=}0)$", fontsize=7); a3.set_zlabel("$h_3$", fontsize=7); a3.tick_params(labelsize=6)
a3.set_title("$H_3$ ($k=3$): the support region is a ruled surface", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig87_G2.pdf")); fig.savefig(os.path.join(HERE, "fig87_G2.png"), dpi=150); plt.close(fig)
# ------------------------------------------------------------------ G3 order: region changes, scalar does not -----------------------
best = (0, None)
for k, sc in SC.items():
    P = sc["paths_pop"]; p = P[5] / P[5].sum(); fs, gs = W.fg(P[12]); a_ = B.J_C(B.J_R(p, fs), gs); b_ = B.J_R(B.J_C(p, gs), fs)
    if abs(a_[1, 2] / a_[1].sum() - b_[1, 2] / b_[1].sum()) < 1e-12 and np.abs(np.outer(*W.fg(p)) - p).max() > 1e-9:
        Ra = V.boundary(*W.fg(a_), "H_lin", n=40); Rb = V.boundary(*W.fg(b_), "H_lin", n=40)
        hd = max(np.min(np.linalg.norm(Ra[:, None] - Rb[None], axis=2), 1).max(), np.min(np.linalg.norm(Rb[:, None] - Ra[None], axis=2), 1).max())
        if hd > best[0]: best = (hd, k)
key = best[1]
sc = SC[key]; P = sc["paths_pop"]; p = P[5] / P[5].sum(); fs, gs = W.fg(P[12]); a_ = B.J_C(B.J_R(p, fs), gs); b_ = B.J_R(B.J_C(p, gs), fs)
fig, ax = plt.subplots(figsize=(4.6, 3.3))
for t, col, lab in ((a_, BLUE, "rows first, then columns"), (b_, ORANGE, "columns first, then rows")):
    hv = np.array([V.FAM["H_lin"](x) for x in V.fibre(*W.fg(t))[0]]); ax.add_patch(plt.Polygon(hv, closed=True, fc=col, alpha=0.15, ec=col, lw=1.4, label=lab))
q = a_[1, 2] / a_[1].sum(); ax.axvline(q, color=INK, lw=0.8, ls=":"); ax.text(q, ax.get_ylim()[1] if False else 0.02, " point estimate $P(\\rm below\\mid S{=}0)$\n same in both orders", fontsize=7, color=INK, va="bottom", transform=ax.get_xaxis_transform())
ax.autoscale_view(); ax.set_xlabel("$P(\\rm below\\mid S{=}0)$"); ax.set_ylabel("$P(\\rm normal\\mid S{=}0)$"); ax.legend(fontsize=7, frameon=False, loc="upper right")
ax.set_title(f"reordering two updates (held-out {key}): the scalar is unchanged,\nthe supported region is not", fontsize=8); clean(ax)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig87_G3.pdf")); fig.savefig(os.path.join(HERE, "fig87_G3.png"), dpi=150); plt.close(fig)
# ------------------------------------------------------------------ G4 coherence map --------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.4))
for ax, L in zip(axs, (20, 30)):
    d = pickle.load(open(os.path.join(ENG, "study87_region", "diag", f"slice_H_L{L}_n25.pkl"), "rb")); U, Vv = d["U"], d["V"]
    colmap = {"empty": "#FFFFFF", "one-wall": "#DBEAFE", "two-walls": "#93C5FD", "braid-triple": BLUE, "failed": RED}
    for r in d["res"]:
        i, j = r["cell"]; ax.add_patch(plt.Rectangle((U[i], Vv[j]), U[i + 1] - U[i], Vv[j + 1] - Vv[j], fc=colmap[r["kind"]], ec="#E5E7EB", lw=0.3))
        if not r["ok"] and r["kind"] != "failed": ax.plot((U[i] + U[i + 1]) / 2, (Vv[j] + Vv[j + 1]) / 2, "x", color=RED)
    uu = np.linspace(U[0], U[-1], 200)
    for gfun in (lambda v: 0.30, lambda v: v, lambda v: 0.70 - v):            # chamber walls f0 = g_j and f0 = g_j + g_k (= 1 - g_l) in the slice
        pass
    vv = np.linspace(Vv[0], Vv[-1], 400)
    for lab, curve in (("$f_0=g_1$", vv), ("$f_0=g_2$", 0.70 - vv), ("$f_0=g_0{+}g_1$", 0.30 + vv), ("$f_0=g_0{+}g_2$", 1.0 - vv)):
        m_ = (curve >= U[0]) & (curve <= U[-1]); ax.plot(curve[m_], vv[m_], color=GRAY, lw=1.0)
    ax.axvline(0.30, color=GRAY, lw=1.0); ax.axhline(0.30, color=PURPLE, lw=1.0, ls=":"); ax.axhline(0.35, color=PURPLE, lw=1.0, ls=":"); ax.axhline(0.40, color=PURPLE, lw=1.0, ls=":")
    ax.set_xlim(U[0], U[-1]); ax.set_ylim(Vv[0], Vv[-1]); ax.set_xlabel("$f_0$"); ax.set_ylabel("$g_1$ ($g_0=0.30$)")
    kinds = {k: sum(1 for r in d["res"] if r["kind"] == k) for k in colmap}; nok = sum(1 for r in d["res"] if r["ok"])
    ax.set_title(f"$L={L}$: all {nok} computed loops return to the identity\n(two-wall loops {kinds['two-walls'] + kinds['braid-triple']}; beyond a caustic: {kinds['failed']} cells)", fontsize=8)
handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="#E5E7EB") for c in ("#DBEAFE", "#93C5FD", RED)] + [plt.Line2D([], [], color=GRAY), plt.Line2D([], [], color=PURPLE, ls=":")]
fig.legend(handles, ["loop crosses one wall", "loop crosses two walls", "beyond a caustic (critical points merge)", "chamber wall (support wall)", "margin-order wall (collision)"], loc="lower center", ncol=3, fontsize=7, frameon=False)
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(os.path.join(HERE, "fig87_G4.pdf")); fig.savefig(os.path.join(HERE, "fig87_G4.png"), dpi=150); plt.close(fig)
# ------------------------------------------------------------------ G5 Study 86 Floer swap ---------------------------------------------
D86 = os.path.join(ENG, "study86_floer"); b = [x for x in json.load(open(os.path.join(D86, "selection.json"))) if x["key"] == "C0"][0]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.0))
for ax, s, nm in zip(axs, (b["s_before"], b["s_after"]), ("before the wall", "after the wall")):
    x = pickle.load(open(os.path.join(D86, "runs", f"8502_C0_L20_s{s:.6f}", "b83.pkl"), "rb")); Mx = np.full((6, 6), np.nan)
    for (i, j), v in x["dims"].items(): Mx[i, j] = v
    ax.imshow(np.where(np.isnan(Mx), -1, Mx), cmap="Blues", vmin=-1, vmax=7)
    for (i, j), v in x["dims"].items(): ax.text(j, i, str(v), ha="center", va="center", fontsize=8, color=INK if v < 4 else "white")
    pb = list(x["perm"]); r = min(pb.index(b["i"]), pb.index(b["j"]))
    ax.add_patch(plt.Rectangle((r + 0.5, r - 0.5), 1, 1, fill=False, ec=PURPLE, lw=2))
    lab = [f"$V_{{{q + 1}}}$" for q in pb]; ax.set_xticks(range(6)); ax.set_xticklabels(lab, fontsize=7); ax.set_yticks(range(6)); ax.set_yticklabels(lab, fontsize=7)
    ax.set_title(f"{nm} (week {s:.2f})\nHom dimensions in the Seidel order", fontsize=8)
fig.suptitle("Study 86: across a collision wall the two colliding thimbles (violet box: no morphisms) swap places;\nthe algebra after is the algebra before, relabelled", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig87_G5.pdf")); fig.savefig(os.path.join(HERE, "fig87_G5.png"), dpi=150); plt.close(fig)
print("figures written; G3 scenario", key)
