"""Figures for Section 17: the worked example of decisions on a time-varying sheaf.

Figure A (fig_timesheaf_a): one column per time.
  top    the context complex: variables as vertices, windows (faces) as edges carrying their tables; the triangle A-B-C is shaded when the
         three pair windows form a cycle, red if they fail to glue; edges of the robust-satisficing cover are solid, others dashed
  bottom the decision plane (expected utility of the triple against the pair): the explanation polytopes E_alpha projected (alpha = lam*
         and alpha = 0.6), the part of E_lam* that survives the expert hypothesis "triple harm <= 0.60" (hatched), the regions where each
         act is best, the requirement u0, and the robust values (star)
Figure B (fig_timesheaf_b): lam* of every cover over time with the support of the hypothesis; the cover frontier at t1 and t3; the
         robustness of each act over time and the decisions taken.
"""
import json, os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from timesheaf_example import steps, ACTS, U0, any_harm
from pce.timesheaf import lam, interval, polygon, polygon_under, lam_under, robust_value, robustness, covers, frontier
from pce.ladder import level1

S, labels = steps(); T = len(S)
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
GOOD, CRIT = "#0ca30c", "#d03b3b"; INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"
ACT_COL = {"triple A+B+C": BLUE, "pair A+B": ORANGE, "nothing": "#8c8c8c"}
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.facecolor": SURF, "figure.facecolor": "white"})
H_BOUND = 0.60
u_tri, u_pair = ACTS["triple A+B+C"], ACTS["pair A+B"]

# ---------- computations ----------
info = []
for s in S:
    l = lam(s); allc = covers(s)
    best = max(((c, a, robustness(s, u, U0, c)) for c in allc for a, u in ACTS.items()), key=lambda r: r[2])
    rung = level1(s.domains, s.windows)["rung"]
    info.append({"lam": l, "best": best, "rung": rung, "hyp": lam_under(s, any_harm, H_BOUND),
                 "poly_l": polygon(s, u_tri, u_pair, l - 1e-7, None, 160), "poly_6": polygon(s, u_tri, u_pair, 0.75, None, 160)})
    al_h = 0.75
    info[-1]["poly_h"] = polygon_under(s, u_tri, u_pair, al_h, any_harm, H_BOUND, None, 160); info[-1]["al_h"] = al_h
    info[-1]["V"] = {a: robust_value(s, u, l - 1e-7) for a, u in ACTS.items()}

# ---------- Figure A ----------
POS = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (0.5, 0.86), "D": (1.55, 0.55)}
fig, axes = plt.subplots(2, T, figsize=(13.5, 6.2), gridspec_kw={"height_ratios": [1.0, 1.15]})
allpts = np.vstack([np.vstack([d["poly_6"], d["poly_l"], [[d["V"]["triple A+B+C"], d["V"]["pair A+B"]]]]) for d in info])
xlo, ylo = allpts.min(axis=0) - 0.04; xhi, yhi = allpts.max(axis=0) + 0.04
for k, (s, d) in enumerate(zip(S, info)):
    ax = axes[0, k]; ax.set_xlim(-0.35, 1.95); ax.set_ylim(-0.42, 1.2); ax.set_aspect("equal"); ax.axis("off")
    names = [w.vars for w in s.windows]; chosen = set(d["best"][0])
    if {("A", "B"), ("A", "C"), ("B", "C")} <= set(names):
        tri = Polygon([POS["A"], POS["B"], POS["C"]], closed=True, facecolor=CRIT if d["lam"] < 1 - 1e-6 else GOOD, alpha=0.12, edgecolor="none")
        ax.add_patch(tri); ax.text(0.5, 0.3, f"cycle\nlam* = {d['lam']:.3f}", ha="center", va="center", fontsize=7, color=INK)
    for i, w in enumerate(s.windows):
        p, q = POS[w.vars[0]], POS[w.vars[1]]; on = i in chosen
        ax.plot([p[0], q[0]], [p[1], q[1]], color=INK if on else "#9b9a95", lw=2.2 if on else 1.2, ls="-" if on else (0, (3, 2)), zorder=2)
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2; nx, ny = q[1] - p[1], p[0] - q[0]; nrm = np.hypot(nx, ny); nx, ny = nx / nrm, ny / nrm
        cx, cy = 0.5, 0.33; sgn = 1 if (mx - cx) * nx + (my - cy) * ny > 0 else -1
        ox, oy = mx + sgn * nx * 0.26, my + sgn * ny * 0.26; Tw = np.asarray(w.table)
        sz = 0.26; ia = ax.inset_axes([ox - sz * Tw.shape[1] / 4, oy - sz / 2, sz * Tw.shape[1] / 2, sz], transform=ax.transData)
        ia.imshow(Tw, cmap="Blues", vmin=0, vmax=0.6); ia.set_xticks([]); ia.set_yticks([])
        for sp in ia.spines.values(): sp.set_edgecolor(INK2); sp.set_linewidth(0.5)
        ia.set_title(f"{w.vars[0]}{w.vars[1]}", fontsize=6, pad=1, color=INK2)
    used = sorted({v for w in names for v in w})
    for v in used:
        ax.scatter(*POS[v], s=170, color="white", edgecolor=INK, lw=1.2, zorder=3)
        ax.text(*POS[v], v + ("'" if (v == "C" and s.domains["C"] == 3) else ""), ha="center", va="center", fontsize=8, zorder=4, color=INK)
    rung = {"R>=0": "glues", "R": "contextual (signed section)", "none": "direct influence"}[d["rung"]]
    ax.set_title(labels[k] + f"\nlam* = {d['lam']:.3f}; {rung}", fontsize=8, color=INK)
    # decision plane
    ax = axes[1, k]
    gx, gy = np.meshgrid(np.linspace(xlo, xhi, 300), np.linspace(ylo, yhi, 300)); best = np.where(gx > np.maximum(gy, 0), 0, np.where(gy > np.maximum(gx, 0), 1, 2))
    tints = matplotlib.colors.ListedColormap(["#dbe8f8", "#fbe1d6", "#ececea"])
    ax.imshow(best, origin="lower", extent=[xlo, xhi, ylo, yhi], cmap=tints, vmin=0, vmax=2, aspect="auto", zorder=0)
    ax.add_patch(Polygon(d["poly_6"], closed=True, facecolor=VIOLET, alpha=0.10, edgecolor=VIOLET, lw=0.8, zorder=1, label="E_0.75"))
    ax.add_patch(Polygon(d["poly_l"], closed=True, facecolor=VIOLET, alpha=0.35, edgecolor=VIOLET, lw=1.2, zorder=2, label="E_lam*"))
    if d["poly_h"] is not None:
        ax.add_patch(Polygon(d["poly_h"], closed=True, facecolor="none", edgecolor=INK, hatch="////", lw=0.8, zorder=3, label="cut by hypothesis"))
    ax.axvline(U0, color=BLUE, ls=":", lw=1); ax.axhline(U0, color=ORANGE, ls=":", lw=1)
    vt, vp = d["V"]["triple A+B+C"], d["V"]["pair A+B"]; ax.scatter([vt], [vp], marker="*", s=90, color=INK, zorder=5)
    a_best = d["best"][1]; ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_xlabel("E[utility of triple]"); ax.set_ylabel("E[utility of pair]" if k == 0 else "")
    ax.set_title(f"decision: {a_best.split()[0]} on " + "+".join(s.names[i] for i in d["best"][0]) + f"\nrobustness {d['best'][2]:.2f}; hypothesis support {d['hyp']:.3f}",
                 fontsize=7.5, color=ACT_COL[a_best])
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
axes[1, 0].text(xlo + 0.01, yhi - 0.02, "pair best", color=ORANGE, fontsize=7, va="top")
axes[1, 0].text(xhi - 0.01, ylo + 0.01, "triple best", color=BLUE, fontsize=7, ha="right")
axes[1, 0].text(xlo + 0.01, ylo + 0.01, "nothing best", color=INK2, fontsize=7)
h = [Polygon([[0, 0]], facecolor=VIOLET, alpha=0.35, edgecolor=VIOLET), Polygon([[0, 0]], facecolor=VIOLET, alpha=0.10, edgecolor=VIOLET),
     Polygon([[0, 0]], facecolor="none", edgecolor=INK, hatch="////"),
     plt.Line2D([], [], marker="*", color=INK, ls="none", ms=9), plt.Line2D([], [], color=INK2, ls=":")]
fig.legend(h, ["explanations E_lam* (as much evidence as glues)", "E_0.75 (less evidence insisted on)", "E_0.75 cut by the hypothesis 'triple harm <= 0.60'",
               "robust values: worst explained case, unexplained part adversarial (hence outside)", "requirement u0 = 0.05 ('good enough')"],
           loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(os.path.join(HERE, "fig_timesheaf_a.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_timesheaf_a.png"), dpi=150, bbox_inches="tight")

# ---------- Figure B ----------
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0)); t = np.arange(T)
names_all = sorted({"+".join(sorted(S[k].names[i] for i in c)) for k in range(T) for c in covers(S[k])})
for nm in names_all:
    ys = []
    for s in S:
        idx = [i for i, n in enumerate(s.names)]; parts = nm.split("+")
        ys.append(lam(s, tuple(s.names.index(p) for p in parts)) if all(p in s.names for p in parts) else np.nan)
    full = nm == "+".join(sorted(S[-1].names))
    ax[0].plot(t, ys, color="#b9b8b3", lw=0.8, zorder=1)
ax[0].plot(t, [d["lam"] for d in info], color=BLUE, lw=2, marker="o", ms=4, label="all windows observed so far", zorder=3)
ax[0].plot(t, [d["hyp"] for d in info], color=ORANGE, lw=1.6, ls="--", marker="s", ms=4, label="support of 'triple harm <= 0.60'", zorder=3)
ax[0].plot([], [], color="#b9b8b3", lw=0.8, label="every sub-cover")
ax[0].set_xticks(t); ax[0].set_xticklabels([l.split(":")[0] for l in labels]); ax[0].set_ylabel("glueable fraction lam*"); ax[0].set_ylim(0.8, 1.01)
for k, ev in ((1, "conflicting\ntrial"), (2, "harm\nrefined"), (3, "drift"), (4, "pendant")):
    ax[0].annotate(ev, (k, 0.805), ha="center", fontsize=6.5, color=INK2)
ax[0].legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3); ax[0].set_title("(a) gluability over time, by cover", fontsize=9)
for k, col, mk in ((1, BLUE, "o"), (3, ORANGE, "s")):
    s = S[k]; pts = [(c, lam(s, c), np.diff(interval(s, any_harm, lam(s, c) - 1e-9, c))[0]) for c in covers(s)]
    ax[1].scatter([p[1] for p in pts], [p[2] for p in pts], color=col, s=18, marker=mk, alpha=0.5)
    fr = sorted(frontier(pts), key=lambda p: p[1]); ax[1].plot([p[1] for p in fr], [p[2] for p in fr], color=col, lw=1.4, label=f"frontier at {labels[k].split(':')[0]}")
    for c, l_, w_ in fr: ax[1].annotate("+".join(s.names[i] for i in c), (l_, w_), fontsize=6, color=INK2, xytext=(3, 3), textcoords="offset points")
ax[1].set_xlabel("glueable fraction of the cover lam*(C)"); ax[1].set_ylabel("width of the triple-harm interval"); ax[1].legend(fontsize=7, frameon=False)
ax[1].set_title("(b) choosing what to glue: covers and their frontier", fontsize=9)
for a, u in ACTS.items():
    if a == "nothing": continue
    hb = [max(robustness(s, u, U0, c) for c in covers(s)) for s in S]; hf = [robustness(s, u, U0) for s in S]
    ax[2].plot(t, [x if x > -np.inf else np.nan for x in hb], color=ACT_COL[a], lw=2, marker="o", ms=4, label=f"{a}: best cover")
    ax[2].plot(t, [x if x > -np.inf else np.nan for x in hf], color=ACT_COL[a], lw=1, ls="--", label=f"{a}: all windows")
ax[2].set_xticks(t); ax[2].set_xticklabels([l.split(":")[0] for l in labels]); ax[2].set_ylabel("robustness (contamination tolerated)")
for k, d in enumerate(info): ax[2].annotate(d["best"][1].split()[0], (k, d["best"][2] + 0.008), ha="center", fontsize=7, color=ACT_COL[d["best"][1]])
ax[2].axvspan(1.5, 2.5, color=CRIT, alpha=0.06); ax[2].annotate("decision expires:\nnew distinction (severe harm)\nenters the utility", (2, 0.17), ha="center", fontsize=6.5, color=CRIT)
ax[2].legend(fontsize=6.5, frameon=False, loc="upper right"); ax[2].set_title("(c) robust satisficing over time", fontsize=9); ax[2].set_ylim(0, 0.26)
for a in ax:
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_timesheaf_b.pdf")); fig.savefig(os.path.join(HERE, "fig_timesheaf_b.png"), dpi=150)
json.dump([{"label": lb, "lam": d["lam"], "rung": d["rung"], "hyp": d["hyp"], "best": ["+".join(S[i].names[j] for j in d["best"][0]), d["best"][1], d["best"][2]],
            "V": d["V"]} for i, (lb, d) in enumerate(zip(labels, info))], open(os.path.join(HERE, "timesheaf_fig.json"), "w"), indent=1, default=float)
print("done")
