"""Figure for Section 14: how perturbations move the level-one polytope and the level-two fibre."""
import json, os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from pce import Window
from pce.ball import CopySystem, profile, agreement_polygon, min_contextuality
from pce.cbd import cbd
R = json.load(open(os.path.join(HERE, "ball_results.json")))
SQ = json.load(open(os.path.join(HERE, "..", "engine", "study8_snowqueen", "data_study8.json"))); KEYS = ["11", "12", "21", "22"]
W3 = [Window((f"A{k[0]}", f"B{k[1]}"), np.asarray(SQ["D3"][k], float) / np.sum(SQ["D3"][k])) for k in KEYS]
S3 = CopySystem(W3); c3 = cbd(W3)["c"]; s = S3.s()
BLUE, ORANGE, PURPLE, GREY, GREEN = "#3a6ea5", "#c0622f", "#6b2d5c", "#8c8c8c", "#4c8c4a"
fig, ax = plt.subplots(2, 3, figsize=(12, 7.2)); ax = ax.ravel()

# (a) Snow Queen: support against the connexion budget, for three data radii
rs = np.linspace(0, 0.2, 41)
for rho, col in ((0.0, BLUE), (0.008, ORANGE), (0.016, PURPLE)):
    cm = min_contextuality(S3, rho); ax[0].plot(rs, np.minimum(1, 1 - cm + rs), color=col, label=f"data radius {rho}")
ax[0].axvline(c3, color=GREY, ls=":", lw=1); ax[0].text(c3 + 0.003, 0.965, f"r = c = {c3:.3f}", fontsize=7, color=GREY)
ax[0].set_xlabel("connexion budget r"); ax[0].set_ylabel("supported fraction"); ax[0].legend(fontsize=7, frameon=False, loc="lower right")
ax[0].set_title("(a) Snow Queen: support vs budget", fontsize=9)

# (b) non-cyclic system: each connexion relaxed alone
nc = R["noncyclic"]; r_ = np.array(nc["r"])
for k, (lab, v) in enumerate(nc["single"].items()):
    is_b3 = lab.startswith("B3"); ax[1].plot(r_, v, color=ORANGE if is_b3 else GREY, lw=2 if is_b3 else 0.8, alpha=1 if is_b3 else 0.7,
                                            label="B3 connexion alone" if is_b3 else ("any other connexion alone" if k == 0 else None))
ax[1].plot(r_, nc["total"], color=BLUE, ls="--", lw=1.2, label="free allocation")
ax[1].set_xlabel("budget r"); ax[1].set_ylabel("supported fraction"); ax[1].legend(fontsize=7, frameon=False, loc="lower right")
ax[1].set_title("(b) non-cyclic system: grading by connexion", fontsize=9)

# (c) the polytope of couplings, projected on agreements; the CbD point and its gap c
i = 2; others = [0, 1, 3]
P = agreement_polygon(S3, i, others, data_radius=0.0, n_dir=120)
ax[2].add_patch(Polygon(P, closed=True, facecolor=BLUE, alpha=0.3, edgecolor=BLUE, lw=1, label="couplings of the copies (tables exact)"))
cb = (1 - s[i], float(np.sum(1 - s[others]))); ax[2].plot(*cb, marker="*", color="k", ms=10, ls="none", label="CbD point (all connexions maximal)")
m = float(np.max(P.sum(axis=1))); xs = np.linspace(0.5, 0.75, 2); ax[2].plot(xs, m - xs, color=ORANGE, lw=1, label="best total agreement")
ax[2].annotate("", xy=(cb[0], m - cb[0]), xytext=cb, arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1))
ax[2].text(cb[0] + 0.004, (cb[1] + m - cb[0]) / 2, f"c = {cb[0] + cb[1] - m:.3f}", fontsize=7, color=ORANGE)
ax[2].set_xlabel("P(Kind/Evil copies agree)"); ax[2].set_ylabel("sum of the other three agreements")
ax[2].set_xlim(0.5, 0.72); ax[2].set_ylim(2.5, 2.9); ax[2].legend(fontsize=6.5, frameon=False, loc="lower left")
ax[2].set_title("(c) couplings polytope; CbD point outside by c", fontsize=9)

# (d) heatmap: supported fraction over (data radius, budget), with the sampling scale
rho_g = np.linspace(0, 0.05, 51); r_g = np.linspace(0, 0.16, 33)
cm = np.array([min_contextuality(S3, x) for x in rho_g]); Z = np.minimum(1, 1 - cm[None, :] + r_g[:, None])
im = ax[3].imshow(Z, origin="lower", aspect="auto", extent=[rho_g[0], rho_g[-1], r_g[0], r_g[-1]], cmap="viridis", vmin=0.85, vmax=1)
rng = np.random.default_rng(0); tv = []
for k in KEYS:
    p = np.asarray(SQ["D3"][k], float).ravel(); p = p / p.sum(); n = SQ["n"][k]
    tv += [0.5 * np.abs(rng.multinomial(n, p) / n - p).sum() for _ in range(4000)]
med, q95 = np.median(tv), np.quantile(tv, 0.95)
for x, ls, lab in ((med, "-", "median sampling move"), (q95, "--", "95% sampling move")):
    ax[3].axvline(x, color="k", ls=ls, lw=1); ax[3].text(x + 0.0008, 0.15, lab, rotation=90, va="top", fontsize=6.5, color="k")
rho_star = rho_g[np.argmax(cm <= 1e-9)]; ax[3].plot(rho_g, np.maximum(0, cm), color="w", lw=0.6, ls=":")
ax[3].set_xlabel("data radius (TV per window)"); ax[3].set_ylabel("connexion budget r"); fig.colorbar(im, ax=ax[3], fraction=0.046)
ax[3].text(0.002, 0.004, "contextual\nremainder", fontsize=6.5, color="w"); ax[3].set_title("(d) support over data ball x budget", fontsize=9)

# (e) level two: the fibre along a path out of the model
rows = R["path_C"]["rows"]; t = np.array([r["t"] for r in rows]); code = np.array([r["code"] for r in rows])
w1 = np.array([r["w1"] for r in rows]); wi = np.array([r["w_imag"] for r in rows]); neg = np.array([r["neg2"] for r in rows])
cols = {0: "#dfe9f5", 1: "#f6e3d6", 2: "#eadcf0", 3: "#e6e6e6", 4: "#ffffff"}
names = {0: "R>=0 (mixture)", 1: "R>=0 off box", 2: "R (signed)", 3: "C (complex pair)", 4: "none"}
start = 0
for k in range(1, len(t) + 1):
    if k == len(t) or code[k] != code[start]:
        ax[4].axvspan(t[start], t[min(k, len(t) - 1)], color=cols[code[start]], label=names[code[start]]); start = k
ax[4].plot(t, w1, color=BLUE, label="Re w1"); ax[4].plot(t, wi, color=PURPLE, label="|Im w|")
for wl in R["path_C"]["walls"]: ax[4].axvline(wl, color=GREY, lw=0.6)
h, l = ax[4].get_legend_handles_labels(); seen = {}; [seen.setdefault(b, a) for a, b in zip(h, l)]
ax[4].legend(seen.values(), seen.keys(), fontsize=6.5, frameon=False, loc="upper left"); ax[4].set_xlabel("path parameter t (mixture -> outside)")
ax[4].set_ylim(-1.5, 2.5); ax[4].set_ylabel("weight (clipped; diverges at the walls)"); ax[4].set_title("(e) level-two fibre along a path", fontsize=9)

# (f) level two: rung shares in the data ball
pr = R["ball"]["profile"]; rr = np.array(sorted(float(k) for k in pr)); keys = ["R>=0", "R>=0 off box", "R", "C"]
Y = np.array([[pr[str(r) if str(r) in pr else repr(r)][k] for r in rr] for k in keys])
ax[5].stackplot(rr, Y, colors=[BLUE, ORANGE, PURPLE, GREY], labels=["R>=0 (mixture)", "R>=0 off box", "R (signed)", "C (complex)"])
ax[5].axvline(R["ball"]["wall_distance"], color="k", ls=":", lw=1); ax[5].text(R["ball"]["wall_distance"] + 0.002, 0.04, "wall distance", fontsize=7, color="w")
ax[5].set_xlabel("data radius (TV)"); ax[5].set_ylabel("share of tables in the ball"); ax[5].set_ylim(0, 1)
ax[5].legend(fontsize=6.5, frameon=True, facecolor="white", framealpha=0.9, loc="center right"); ax[5].set_title("(f) level-two robustness of a hypothesis", fontsize=9)
for a in ax: a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_ball.pdf")); fig.savefig(os.path.join(HERE, "fig_ball.png"), dpi=140)
json.dump({"rho_star": float(rho_star), "sampling_tv_median": float(med), "sampling_tv_q95": float(q95)}, open(os.path.join(HERE, "ball_fig_stats.json"), "w"))
print("rho*", rho_star, "median TV", med, "q95", q95)
