"""Study 83 figures F1-F7 (fig83_F*.pdf), from the held-out seed (8302) outputs. Every projection is named on its figure."""
import os, sys, pickle, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
HERE = os.path.dirname(os.path.abspath(__file__)); E = os.path.join(HERE, "..", "engine"); sys.path.insert(0, E)
import gen83 as G, l1_83 as L1, l2_83 as W, mirror83 as M
SEED, TAG = 8302, "H"; SD = os.path.join(E, "study83_seed")
INK, MUTED, BLUE, ORANGE, GREEN, PURPLE, GRAY = "#1F2937", "#6B7280", "#1D4ED8", "#C2410C", "#047857", "#7C3AED", "#9CA3AF"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
SC = {f"{d['params']['cls']}{d['params']['k']}": d for d in G.build(SEED)}
R1 = {f"{r['cls']}{r['k']}": r for r in pickle.load(open(os.path.join(SD, f"{TAG}_l1.pkl"), "rb"))}
R3 = {f"{r['cls']}{r['k']}": r for r in pickle.load(open(os.path.join(SD, f"{TAG}_l3.pkl"), "rb"))}
def save(fig, name): fig.savefig(os.path.join(HERE, f"fig83_{name}.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, f"fig83_{name}.png"), dpi=120, bbox_inches="tight"); plt.close(fig)
# ------------------------------------------------------------------ F7: the data slice in H^2 (lem:dataslice), drawn in 3D
def f7():
    V = np.array([(1, 1), (-1, 0), (0, -1), (-1, -1), (1, 0), (0, 1)], float); A = np.column_stack([V, np.ones(6)])
    Qf, _ = np.linalg.qr(np.column_stack([A, np.eye(6)])); B3 = Qf[:, 3:6]          # orthonormal basis of the complement of span(M, c1): the slice directions
    X = lambda f, g: np.array([f[i] * g[j] for i, j in W.CELLS]) @ B3
    rng = np.random.default_rng(0); cloud = []
    for _ in range(4000):
        f1 = rng.uniform(0.05, 0.95); g = rng.dirichlet([1.5, 1.5, 1.5]); cloud.append(X([f1, 1 - f1], g))
    cloud = np.array(cloud); fig = plt.figure(figsize=(11, 5.2)); paths = {}
    for k in ("B0", "C0", "D0", "E2"):
        P = [W.fg(x) for x in SC[k]["paths_pop"]]; marks = []
        for w in R3[k]["walls"]:
            s = w["s"]; a_ = int(np.floor(s)); fr = s - a_; f = P[a_][0] + fr * (P[min(a_ + 1, 12)][0] - P[a_][0]); g = P[a_][1] + fr * (P[min(a_ + 1, 12)][1] - P[a_][1]); marks.append((X(f, g), w["type"]))
        paths[k] = (np.array([X(f, g) for f, g in P]), marks)
    allp = np.vstack([p for p, _ in paths.values()]); lo, hi = allp.min(0), allp.max(0); pad = 0.15 * (hi - lo).max()
    for pi, zoom in ((1, False), (2, True)):
        ax = fig.add_subplot(1, 2, pi, projection="3d")
        if zoom: sel = np.all((cloud > lo - pad) & (cloud < hi + pad), axis=1); ax.scatter(*cloud[sel].T, s=2, color=GRAY, alpha=0.35)
        else: ax.scatter(*cloud.T, s=1.5, color=GRAY, alpha=0.3)
        for k, (pts, marks) in paths.items():
            ax.plot(*pts.T, color=BLUE, lw=1.8 if zoom else 1.2)
            for x_, typ in marks: ax.scatter(*x_, s=34 if zoom else 14, color=ORANGE if typ == "leximin" else PURPLE, marker="o" if typ == "leximin" else "s", depthshade=False)
        if zoom: ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(lo[1] - pad, hi[1] + pad); ax.set_zlim(lo[2] - pad, hi[2] + pad)
        ax.set_xlabel("slice coord. 1", fontsize=8); ax.set_ylabel("slice coord. 2", fontsize=8); ax.set_zlabel("slice coord. 3", fontsize=8)
        ax.set_title("the whole data slice" if not zoom else "zoom on the weekly paths", fontsize=9)
        if not zoom:
            ax.scatter([], [], color=GRAY, s=8, label="all $2\\times3$ margins (open in $c_1\\cdot[\\omega]=1$)"); ax.plot([], [], color=BLUE, label="weekly paths (held out: B0, C0, D0, E2)")
            ax.scatter([], [], color=ORANGE, label="leximin wall crossed"); ax.scatter([], [], color=PURPLE, marker="s", label="margin-order wall crossed")
            ax.legend(loc="upper left", fontsize=7, frameon=False)
    fig.suptitle("F7. The data slice of the Kähler space $H^2(\\mathrm{dP}_6)=\\mathbb{R}^4$ (orthonormal coordinates on the hyperplane $c_1\\cdot[\\omega]=1$; nothing is projected away;\n$L$ moves the whole slice along $c_1$, the fourth direction, which no table can reach)", fontsize=9)
    save(fig, "F7_slice")
# ------------------------------------------------------------------ F1: the fibre polygon of one 2x3 table (region vs band)
def f1():
    a, b = "E0", "E1"; tbA = SC[a]["paths_pop"][-1]; tbB = SC[b]["paths_pop"][-1]; f, g = W.fg(tbA)
    # fibre of the 2x3 table: p11, p12 free; p13 = f0 - p11 - p12; p2j = g_j - p1j; all >= 0
    xs = np.linspace(0, f[0], 400); poly = []
    import itertools
    ineq = lambda p11, p12: min(p11, p12, f[0] - p11 - p12, g[0] - p11, g[1] - p12, g[2] - (f[0] - p11 - p12))
    grid = [(x, y) for x in np.linspace(0, f[0], 300) for y in np.linspace(0, f[0], 300) if ineq(x, y) >= 0]; grid = np.array(grid)
    from scipy.spatial import ConvexHull
    hull = ConvexHull(grid); fig, ax = plt.subplots(figsize=(6.2, 5))
    ax.add_patch(Polygon(grid[hull.vertices], closed=True, fc="#DBEAFE", ec=BLUE, lw=1.2, label="fibre: all tables with these margins"))
    ind = np.outer(f, g); ax.plot(ind[0, 0], ind[0, 1], "o", ms=7, mfc="white", mec=INK, mew=1.6, zorder=6, label="independence = maximum entropy (the toric point $x^0$)")
    lv, xl = W.leximin(f, g); xl = np.array(xl).reshape(2, 3); ax.plot(xl[0, 0], xl[0, 1], "D", color=GREEN, label="leximin point")
    for k, mk, col in ((a, "*", ORANGE), (b, "P", PURPLE)):
        tb = SC[k]["paths_pop"][-1]; ax.plot(tb[0, 0], tb[0, 1], mk, ms=12, color=col, label=f"true table, scenario {k}" + (" (no-effect twin)" if k == b else " (effect)"))
    th = 0.6; ax.axvline(th * f[0], color=INK, ls="--", lw=1); ax.text(th * f[0] + 0.002, ax.get_ylim()[1] * 0.02 + 0.07, "policy wall\n$P(\\mathrm{above}\\mid\\mathrm{seeded})=0.6$", fontsize=7.5, color=INK)
    ax.set_xlabel("$p_{11}$ = P(seeded, above normal)"); ax.set_ylabel("$p_{12}$ = P(seeded, normal)")
    ax.set_title("F1. One margin pair, two laws: the fibre of the $2\\times3$ target table\n(linear projection onto $(p_{11},p_{12})$; the other four cells are determined by the margins)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper right"); ax.set_xlim(0, f[0] * 1.02); ax.set_ylim(0, f[0] * 1.02); save(fig, "F1_fibre")
# ------------------------------------------------------------------ F3: critical values along one path, Stokes events
def f3():
    k = "E2"; P = [W.fg(x) for x in SC[k]["paths_pop"]]; T = M.Tracker(); T.step_to(P[0][0], P[0][1], 20.0, label="route"); traj = [T.vals.copy()]
    for q in range(1, 13):
        f0, g0 = T.fgl[0], T.fgl[1]
        for s in np.linspace(0, 1, 9)[1:]:
            T.step_to(f0 + s * (P[q][0] - f0), g0 + s * (P[q][1] - g0), 20.0, label=q); traj.append(T.vals.copy())
    traj = np.array(traj); fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.4)); cols = plt.cm.viridis(np.linspace(0, 0.9, 6))
    for ai, ax in enumerate(axs):
        for p in range(6): ax.plot(traj[:, p].real, traj[:, p].imag, color=cols[p], lw=1.4, label=f"$v_{p + 1}$"); ax.plot(traj[-1, p].real, traj[-1, p].imag, "o", color=cols[p], ms=4)
        for e in R3[k][20.0]["events"]:
            a_ = int(np.floor(e["s"])); idx = min(len(traj) - 1, 1 + a_ * 8 + int(round((e["s"] - a_) * 8)))
            ax.plot(traj[idx, e["i"]].real, traj[idx, e["i"]].imag, "x", color=ORANGE, ms=9, mew=2)
        d = np.exp(0.3j); sc_ = 0.35
        if ai == 1: ax.annotate("", xy=(d.real * sc_, d.imag * sc_), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK)); ax.text(d.real * sc_ * 1.02, d.imag * sc_ * 1.02, "towards $s_0$", fontsize=7.5) if ai == 1 else None
        ax.set_xlabel("Re $v$"); ax.set_ylabel("Im $v$"); ax.grid(alpha=0.15)
    axs[1].set_xlim(-1.0, 0.8); axs[1].set_ylim(-0.5, 0.5); axs[0].set_title("all six critical values", fontsize=9); axs[1].set_title("zoom near the origin", fontsize=9)
    axs[0].plot([], [], "x", color=ORANGE, label="Stokes event (one thimble changes class)"); axs[0].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle("F3. The critical values of $W$ along the weekly path of scenario E2 ($L=20$, held out; no projection: they live in $\\mathbb{C}$)", fontsize=9)
    save(fig, "F3_critical")
# ------------------------------------------------------------------ F2: walls along the weekly paths (timeline)
def f2():
    keys = [k for k in R3 if k[0] in "BCDE"]; fig, ax = plt.subplots(figsize=(8.5, 5.6))
    for r_, k in enumerate(keys):
        y = len(keys) - r_; ax.plot([0, 12], [y, y], color="#E5E7EB", lw=6, solid_capstyle="round")
        for w in R3[k]["walls"]: ax.plot(w["s"], y, "|", ms=14, mew=2, color=ORANGE if w["type"] == "leximin" else PURPLE)
        for e in R3[k][20.0]["events"]: ax.plot(e["s"], y + 0.22, "v", ms=5, color=BLUE if (e["i"], e["j"]) == (5, 4) else GRAY)
        P = SC[k]["P"]; qs = []
        for wk in range(13):
            Pk = P.copy(); Pk[:, wk + 1:] = 0; qs.append(L1.q_of(Pk / Pk.sum()))
        for q in range(12):
            if (qs[q] - 0.6) * (qs[q + 1] - 0.6) < 0: ax.plot(q + 0.5, y - 0.22, "^", ms=6, color=GREEN)
        ax.text(-0.3, y, k, ha="right", va="center", fontsize=8)
    for lab, kw in (("leximin wall (L2, exact from the margins)", dict(marker="|", color=ORANGE, ms=10, mew=2, ls="none")), ("margin-order wall (L2)", dict(marker="|", color=PURPLE, ms=10, mew=2, ls="none")),
                    ("Stokes event of the Study 72 pair (6,5) (L3)", dict(marker="v", color=BLUE, ls="none")), ("other Stokes event (L3)", dict(marker="v", color=GRAY, ls="none")),
                    ("decision change, $\\theta=0.6$ (L1/L5, needs the full table)", dict(marker="^", color=GREEN, ls="none"))): ax.plot([], [], label=lab, **kw)
    ax.set_xlabel("week (cumulative data, June to August)"); ax.set_yticks([]); ax.set_xlim(-1, 12.2)
    ax.set_title("F2. One data stream, several kinds of wall (held-out seed, classes B-E, $L=20$)", fontsize=9); ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    save(fig, "F2_walls")
# ------------------------------------------------------------------ F6: information, band, action per refinement
def f6():
    keys = ["A1", "B0", "D1", "E0"]; fig, axs = plt.subplots(1, 3, figsize=(10, 3.4))
    for k in keys:
        r = R1[k]; H = [r["levels"][l]["H_maxent"] for l in range(5)]; wd = [r["levels"][l]["band_pop"][1] - r["levels"][l]["band_pop"][0] for l in range(5)]
        act = [{"seed": 1, "undetermined": 0, "do not seed": -1}[r["levels"][l]["decisions_pop"][0.6]] for l in range(5)]
        axs[0].plot(range(5), H, "o-", label=k); axs[1].plot(range(5), wd, "o-", label=k); axs[2].plot(range(5), act, "o-", label=k)
    axs[0].set_title("information: entropy of the max-ent law (nats)", fontsize=8.5); axs[1].set_title("band width of $q$ (identification)", fontsize=8.5); axs[2].set_title("robust action at $\\theta=0.6$", fontsize=8.5)
    axs[2].set_yticks([-1, 0, 1]); axs[2].set_yticklabels(["do not seed", "undetermined", "seed"])
    for a in axs: a.set_xticks(range(5)); a.set_xticklabels([f"I{l}" for l in range(5)]); a.grid(alpha=0.15)
    axs[0].legend(fontsize=7, frameon=False); fig.suptitle("F6. Information gained, band width and action can move independently (held-out seed)", fontsize=9); fig.tight_layout(); save(fig, "F6_info")
if __name__ == "__main__":
    for fn in (f7, f1, f3, f2, f6): fn(); print("done", fn.__name__)
# ------------------------------------------------------------------ F4: Floer dimensions before / predicted / after at each leximin bracket
def b83(key, s):
    p = os.path.join(SD, "runs", f"{SEED}_{key}_L20_s{s:.4f}", "b83.pkl")
    if not os.path.exists(p): return None
    b = pickle.load(open(p, "rb")); ws = os.path.join(os.path.dirname(p), "wall_seconds.txt"); b["wall"] = float(open(ws).read()) if os.path.exists(ws) else np.nan; return b
def f4():
    import mut83 as U
    SEL = json.load(open(os.path.join(SD, f"{TAG}_selection.json"))); br = SEL["p2_brackets"]
    fig, axs = plt.subplots(len(br), 2, figsize=(8.4, 3.0 * len(br)), squeeze=False)
    for r_, (k, sb, sa, pb, pa, ch, ps) in enumerate(br):
        bb, ba = b83(k, sb), b83(k, sa)
        if bb is None or ba is None: continue
        side, pred = U.predict(bb["CB"], bb["dims"], pb, pa, ch, ps); cmp_ = U.compare(pred, bb["dims"], pb, ba["dims"], pa, ch)
        for c_, (b, perm, title) in enumerate(((bb, pb, f"{k}: before the event (s = {sb})"), (ba, pa, f"after (s = {sa}); {side} mutation, thimble {ch + 1} passes {ps + 1}"))):
            ax = axs[r_, c_]; Mx = np.full((6, 6), np.nan)
            for (i, j), v in b["dims"].items(): Mx[i, j] = v
            ax.imshow(np.where(np.isnan(Mx), 0, Mx), cmap="Blues", vmin=0, vmax=8)
            for i in range(6):
                for j in range(6):
                    if i < j:
                        posA, posB = perm[i], perm[j]; txt = f"{int(Mx[i, j])}"
                        if c_ == 1:
                            key2 = (posA, posB) if (posA, posB) in cmp_ else (posB, posA); pv = cmp_.get(key2, (None,))[0]
                            if ch in (posA, posB): txt += f"\n(pred {pv})"
                        ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=INK)
            ax.set_xticks(range(6)); ax.set_yticks(range(6)); ax.set_xticklabels([f"$V_{p + 1}$" for p in perm], fontsize=7); ax.set_yticklabels([f"$V_{p + 1}$" for p in perm], fontsize=7)
            ax.set_title(title, fontsize=8)
    fig.suptitle("F4. The mutation predicted from the algebra before the Stokes event, against the Floer dimensions after it\n(held-out brackets; rows and columns in Seidel order; 'pred' = dimension predicted by the mutation)", fontsize=9)
    fig.tight_layout(); save(fig, "F4_mutation")
# ------------------------------------------------------------------ F5: compute ledger on the chamber-reuse path
def f5():
    SEL = json.load(open(os.path.join(SD, f"{TAG}_selection.json"))); k, reps, audits, wrong = SEL["p5"][0]
    ev = sorted(e["s"] for e in R3[k][20.0]["events"] if e["s"] < 8.0); fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 3.2), gridspec_kw=dict(width_ratios=[2.2, 1]))
    bounds = [0.0] + ev + [8.0]
    for c_ in range(len(bounds) - 1): ax.axvspan(bounds[c_], bounds[c_ + 1], color=["#EFF6FF", "#FEF3C7"][c_ % 2])
    for e in ev: ax.axvline(e, color=BLUE, lw=1)
    for w in R3[k]["walls"]:
        if w["type"] == "margin-order" and w["s"] < 8: ax.axvline(w["s"], color=PURPLE, lw=1.4, ls="--")
    ax.plot([], [], color=PURPLE, ls="--", label="margin-order wall: two critical values coincide (missed by the event detector)")
    import iso83 as I
    for s_, c_ in audits:
        ba = b83(k, s_); rb = b83(k, reps[str(c_)]); ok = ba is not None and rb is not None and I.iso(rb["CB"], rb["dims"], ba["CB"], ba["dims"])[0] is not None
        ax.text(s_, 0.28, "match" if ok else "no match", ha="center", fontsize=7.5, color=GREEN if ok else ORANGE)
    ax.plot(range(9), [0] * 9, "|", ms=18, color=GRAY, mew=2, label="baseline: a Floer computation at every week point")
    ax.plot(list(reps.values()), [0] * len(reps), "o", ms=10, color=GREEN, label="performed: one representative per chamber")
    ax.plot([s for s, _ in audits], [0] * len(audits), "D", ms=8, mfc="white", mec=ORANGE, mew=2, label="audit (random, must match its chamber)")
    ax.set_yticks([]); ax.set_xlabel("week"); ax.set_xlim(-0.2, 8.2); ax.set_ylim(-1, 1); ax.legend(fontsize=6.8, frameon=False, loc="upper center", ncol=1, bbox_to_anchor=(0.5, -0.33))
    ax.set_title(f"chambers of scenario {k} between Stokes events (blue lines), held out", fontsize=8.5)
    walls = [b83(k, s)["wall"] for s in reps.values() if b83(k, s)]; mean = np.mean(walls) if walls else np.nan; l3 = R3[k][20.0]["seconds"]
    bx.barh([2, 1, 0], [9 * mean / 60, len(reps) * mean / 60, l3 / 60], color=[GRAY, GREEN, BLUE])
    bx.set_yticks([2, 1, 0]); bx.set_yticklabels(["baseline (9 Floer runs)", f"performed ({len(reps)} Floer runs)", "geometry: mirror layer L3"], fontsize=7.5); bx.set_xlabel("minutes")
    bx.set_title("compute ledger", fontsize=8.5); fig.suptitle("F5. What knowing the chambers saves: the category is computed once per chamber, not at every point", fontsize=9); fig.tight_layout(); save(fig, "F5_ledger")
# ------------------------------------------------------------------ F8: forest plot of the decision quantity per scenario
def f8():
    keys = [f"{c}{k}" for c in "ABCDE" for k in range(4)]; fig, ax = plt.subplots(figsize=(8.2, 7.4))
    for r_, k in enumerate(keys):
        y = len(keys) - r_; R = R1[k]; L3, L4 = R["levels"][3], R["levels"][4]
        ax.plot(L3["band_pop"], [y + 0.18] * 2, color="#93C5FD", lw=6, solid_capstyle="butt")
        if L4["band_fin"]: ax.plot(L4["band_fin"], [y - 0.05] * 2, color=BLUE, lw=2.2)
        c3 = L3["conv"]; ax.plot(c3[1], [y - 0.25] * 2, color=ORANGE, lw=2.2); ax.plot(c3[0], y - 0.25, "|", color=ORANGE, ms=8, mew=2)
        ax.plot(R["q_true"], y + 0.18, "*", color=INK, ms=11, zorder=5)
        ax.text(-0.02, y, k + (" (twin)" if R["params"]["twin"] else ""), ha="right", va="center", fontsize=7.5)
    for th in (0.5, 0.6, 0.7): ax.axvline(th, color=GRAY, ls="--", lw=1)
    for c_, y in zip("ABCDE", [18.5, 14.5, 10.5, 6.5, 2.5]): ax.text(1.02, y, {"A": "A: confounded,\nno effect", "B": "B: weak effect", "C": "C: redistribution", "D": "D: detectable\neffect", "E": "E: effect and\nno-effect twin"}[c_], fontsize=7.5, va="center", color=MUTED)
    for y in (4.5, 8.5, 12.5, 16.5): ax.axhline(y, color="#E5E7EB", lw=0.8)
    ax.plot([], [], "*", color=INK, ms=10, label="true $q$ (known only to the scorer)"); ax.plot([], [], color="#93C5FD", lw=6, label="geometric band at $I_3$ (population: what the contexts can identify)")
    ax.plot([], [], color=BLUE, lw=2.2, label="geometric band at $I_4$ (finite sample, 95% box)"); ax.plot([], [], color=ORANGE, lw=2.2, label="conventional 95% interval at $I_3$ (forecast ignored)")
    ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_xlabel("$q$ = P(above-normal rain | seeded, wet forecast)"); ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2)
    ax.set_title("F8. What each method says about the decision quantity, scenario by scenario (held-out seed)\ndashed lines: the three registered policy thresholds $\\theta$", fontsize=9); save(fig, "F8_forest")
# ------------------------------------------------------------------ F9: decision map (actions by scenario and refinement) vs truth
def f9(th=0.6):
    keys = [f"{c}{k}" for c in "ABCDE" for k in range(4)]; code = {"seed": 2, "undetermined": 1, "do not seed": 0, None: np.nan, "empty": np.nan}
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#FCA5A5", "#FDE68A", "#6EE7B7"]); fig, axs = plt.subplots(1, 4, figsize=(12, 6.2), gridspec_kw=dict(width_ratios=[5, 5, 5, 1.3]))
    G0 = np.array([[code[R1[k]["levels"][l]["decisions_pop"][th]] for l in range(5)] for k in keys], float)
    G1 = np.array([[code[R1[k]["levels"][l]["decisions_fin"][th]] for l in range(5)] for k in keys], float)
    G2 = np.array([[code[R1[k]["levels"][l]["decisions_conv"][th]] if l >= 1 else np.nan for l in range(5)] for k in keys], float)
    T = np.array([[2 if R1[k]["q_true"] >= th else 0] for k in keys], float)
    for ax, M_, title in ((axs[0], G0, "geometric, population\n(what the data could identify)"), (axs[1], G1, "geometric, finite sample\n(about 800 events per context)"), (axs[2], G2, "conventional\n(point estimate + 95% CI)"), (axs[3], T, "truth")):
        ax.imshow(M_, cmap=cmap, vmin=0, vmax=2, aspect="auto")
        ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys if ax is axs[0] else [], fontsize=7)
        ax.set_xticks(range(M_.shape[1])); ax.set_xticklabels([f"$I_{l}$" for l in range(M_.shape[1])] if M_.shape[1] > 1 else [""], fontsize=8); ax.set_title(title, fontsize=8.5)
        if M_.shape[1] > 1:
            for i in range(M_.shape[0]):
                for j in range(M_.shape[1]):
                    if np.isnan(M_[i, j]): ax.text(j, i, "n/a", ha="center", va="center", fontsize=6, color=MUTED); continue
                    if M_[i, j] != 1 and M_[i, j] != T[i, 0]: ax.text(j, i, "wrong", ha="center", va="center", fontsize=6.5, color=INK, fontweight="bold")
    from matplotlib.patches import Patch
    axs[2].legend(handles=[Patch(color="#6EE7B7", label="seed"), Patch(color="#FDE68A", label="undetermined: collect more information"), Patch(color="#FCA5A5", label="do not seed")],
                  fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.0, -0.06), ncol=3)
    fig.suptitle(f"F9. The robust action at $\\theta={th}$ as the refinements arrive ($I_0$ monthly totals ... $I_4$ forecast and mass), against the truth (held-out seed)\n'wrong' = a committed action on the wrong side of $\\theta$: the geometric layer never commits wrongly; it commits correctly once the data identify $q$ ($I_4$, population)", fontsize=9)
    fig.subplots_adjust(top=0.84); save(fig, "F9_decisionmap")
# ------------------------------------------------------------------ F10: the decision geometry of a band
def f10(th=0.6):
    fig, ax = plt.subplots(figsize=(6.6, 6.2)); ax.add_patch(Polygon([[0, 0], [th, th], [0, th]], closed=True, fc="#FCA5A5", alpha=0.7))
    ax.add_patch(Polygon([[th, th], [1, 1], [th, 1]], closed=True, fc="#6EE7B7", alpha=0.7)); ax.add_patch(Polygon([[0, th], [th, th], [th, 1], [0, 1]], closed=True, fc="#FDE68A", alpha=0.7))
    ax.plot([0, 1], [0, 1], color=INK, lw=1); ax.text(0.03, 0.47, "do not seed\n(upper end below $\\theta$)", fontsize=8); ax.text(0.68, 0.9, "seed\n(lower end\nabove $\\theta$)", fontsize=8)
    ax.text(0.08, 0.78, "undetermined:\ncollect more information", fontsize=8.5)
    for k, col in (("A0", PURPLE), ("B1", BLUE), ("D1", GREEN), ("E2", ORANGE), ("E3", INK)):
        R = R1[k]; pts = [R["levels"][l]["band_pop"] for l in range(5)]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=col, ms=4, lw=1.3, label=k + (" (twin of E2)" if k == "E3" else ""))
        ax.annotate("$I_3$", (pts[3][0], pts[3][1]), textcoords="offset points", xytext=(4, 3), fontsize=7, color=col)
        ax.plot(R["q_true"], R["q_true"], "*", color=col, ms=12, mec="white")
    ax.annotate("$I_0, I_1, I_2$: band $[0,1]$", (0, 1), textcoords="offset points", xytext=(8, -14), fontsize=7.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.0); ax.set_aspect("equal"); ax.set_xlabel("lower end of the band of $q$"); ax.set_ylabel("upper end of the band of $q$")
    ax.set_title(f"F10. The decision geometry of a band ($\\theta={th}$): each refinement moves the band;\nat $I_4$ it reaches the diagonal (a point, star = truth) and the action is committed", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right"); save(fig, "F10_decisiongeom")
