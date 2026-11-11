"""The atlas: Study 26 seen at five levels, from the toric (Kahler) layer to braids, with labelled maps between levels.
One representative signal is followed through every level: the foreign {M,R,Y} table that was refitted and later retracted (purple).
Other signals appear where they live: the coalition (amber), the twist (black, dotted), the decision threshold (blue, dashed).
Object types are kept apart: observed tables (grey squares), inferred laws (hollow circles), selected states (filled markers),
constructed encodings (the braid). Every panel is computed from Study 26 except where marked; the braid is built by an explicit encoding
rule (ours) from the study's actual refit order, and it records order only."""
import os, sys, itertools, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ENG = os.path.join(HERE, "..", "engine"); sys.path.insert(0, ENG); sys.path.insert(0, os.path.join(ENG, "study26_e2e"))
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyBboxPatch, FancyArrowPatch, Rectangle
from scipy.optimize import linprog
import world26 as W
from pce import refitcommute as RC
R = json.load(open(os.path.join(ENG, "study26_e2e", "results_study26.json"))); KTS = json.load(open(os.path.join(ENG, "kerneltransport_summary.json")))
BLUE, AMBER, PURPLE, GREEN = "#1f5f9e", "#c27c0e", "#8e3b8e", "#2e7d32"; INK, MUTED, GRID, OBS = "#222222", "#666666", "#d9d9d9", "#9a9a9a"
plt.rcParams.update({"font.size": 7.4, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 6.6, "ytick.labelsize": 6.6})
S, V = W.STATES, W.VARS; NS = 32; Pstar = W.truth(); dev, hold = W.generate()
def ctxmat(ctx, states=S, names=V):
    idx = [names.index(v) for v in ctx]; A = np.zeros((2 ** len(ctx), len(states)))
    for i, s in enumerate(states): A[int("".join(str(s[j]) for j in idx), 2), i] = 1
    return A
def ev(pred): return np.array([1.0 if pred(dict(zip(V, s))) else 0.0 for s in S])
def labels(ctx): return np.array([int("".join(str(s[V.index(v)]) for v in ctx), 2) for s in S])

# ------------------------------------------------------------------ computations
DEVCTX = [c for c, _, _ in W.CORE_ALLOC]; A0 = np.vstack([ctxmat(c) for c in DEVCTX]); m0 = A0 @ Pstar
def refit(Q, ctxs, target=Pstar):
    B = RC.span_labels([labels(c) for c in ctxs], NS); return RC.iproj(Q, B, B.T @ target)
def inj_law():
    th = dict(W.THETA_TRIPLE); th[("M", "R", "Y")] += 1.5; logp = np.zeros(NS)
    for i, s in enumerate(S):
        v = dict(zip(V, s)); logp[i] = sum(t * v[x] for x, t in W.THETA_MAIN.items()) + sum(t * v[x] * v[y] for (x, y), t in W.THETA_PAIR.items()) \
            + sum(t * v[x] * v[y] * v[z] for (x, y, z), t in th.items())
    p = np.exp(logp); return p / p.sum()
rng = np.random.default_rng(26); o1 = rng.permutation(len(DEVCTX)); U0 = np.ones(NS) / NS
P = U0.copy()
for step, i in enumerate(o1):
    P = refit(P, [DEVCTX[i]])
    if step == 2: P = refit(P, [("M", "R", "Y")], inj_law())
RB = refit(P, DEVCTX); R0 = refit(U0, DEVCTX)
M1, M0, Y1 = ev(lambda v: v["M"] == 1), ev(lambda v: v["M"] == 0), ev(lambda v: v["Y"] == 1)
kk = ev(lambda v: v["C"] == 0 and v["R"] == 1 and v["H"] == 0); dvec = Y1 * M1 * kk / ((M1 * kk) @ Pstar) - Y1 * M0 * kk / ((M0 * kk) @ Pstar); tau = 0.10
Nsp = np.linalg.svd(A0)[2][np.linalg.matrix_rank(A0):].T
d1 = RB - R0; d1 /= np.linalg.norm(d1); dv = Nsp @ (Nsp.T @ dvec); d2 = dv - (dv @ d1) * d1; d2 /= np.linalg.norm(d2)
def section_polygon(base, e1, e2, n=720):
    pts = []
    for th in np.linspace(0, 2 * np.pi, n + 1)[:-1]:
        c = np.array([np.cos(th), np.sin(th)]); r = linprog(-c, A_ub=-np.column_stack([e1, e2]), b_ub=base, bounds=[(None, None)] * 2, method="highs"); pts.append(r.x)
    return np.array(pts)
poly = section_polygon(R0, d1, d2)
coord = lambda X: np.linalg.lstsq(np.column_stack([d1, d2]), X - R0, rcond=None)[0]
t1 = np.log(RB) - np.log(R0); t2 = ev(lambda v: v["M"] and v["H"] and v["Y"])
def chart(s, t):
    Q = R0 * np.exp(s * t1 + t * t2); Q /= Q.sum()
    try: return coord(refit(Q, DEVCTX))
    except RuntimeError: return np.array([np.nan, np.nan])
gs_ = np.linspace(-1.5, 3.0, 31); gt_ = np.linspace(-2.5, 2.5, 31)
lines_s = [np.array([chart(s, t) for s in gs_]) for t in np.linspace(-2, 2, 7)]
lines_t = [np.array([chart(s, t) for t in gt_]) for s in np.linspace(-1, 2.5, 8)]
# leximin of the exact-data fibre (sequential LP over the 32 cells)
def lexmaxmin(Aeq, beq, n):
    fixed, free = {}, set(range(n))
    while free:
        Ae = np.vstack([Aeq] + [np.eye(n)[k][None] for k in fixed]); be = np.concatenate([beq] + [[fixed[k]] for k in fixed])
        r = linprog(np.r_[np.zeros(n), -1.0], A_ub=np.array([np.r_[-np.eye(n)[k], 1.0] for k in sorted(free)]), b_ub=np.zeros(len(free)),
                    A_eq=np.hstack([Ae, np.zeros((len(be), 1))]), b_eq=be, bounds=[(0, None)] * n + [(None, None)], method="highs"); t = -r.fun; Sx = []
        for k in sorted(free):
            r2 = linprog(-np.eye(n)[k], A_ub=np.array([-np.eye(n)[j] for j in sorted(free)]), b_ub=-np.full(len(free), t) + 1e-12, A_eq=Ae, b_eq=be, bounds=[(0, None)] * n, method="highs")
            if r2.status == 0 and -r2.fun <= t + 1e-7: Sx.append(k)
        for k in (Sx or sorted(free)): fixed[k] = t; free.discard(k)
    return np.array([fixed[k] for k in range(n)])
LEX = lexmaxmin(A0, m0, NS)
# cycle correlations
cyc = [s for s in hold if s["family"].startswith("cycle")]
def corr(cond):
    out = {}
    for e in W.CYCLE_EDGES:
        cnt = sum(s["counts"] for s in cyc if s["ctx"] == e and s["family"] == "cycle-" + cond); p = cnt / cnt.sum(); out[e] = p[0] - p[1] - p[2] + p[3]
    return out
cc, ct = corr("consistent"), corr("twisted"); pat = [g for g in itertools.product((1, -1), repeat=4) if np.prod(g) == -1]
def proj_poly(nc):
    pts = []
    for th in np.linspace(0, 2 * np.pi, 361)[:-1]:
        c = np.array([np.cos(th), np.sin(th)]); r = linprog(-np.array([c[0], c[0], c[0], c[1]]), A_ub=np.array(pat, float) if nc else None, b_ub=np.full(len(pat), 2.0) if nc else None,
                                                            bounds=[(-1, 1)] * 4, method="highs"); x = r.x; pts.append([x[0] + x[1] + x[2], x[3]])
    return np.array(pts)
ncpoly, locpoly = proj_poly(True), proj_poly(False)
# the braid: encoding rule E (ours). Strands = contexts; a refit moves its context's strand to the top, passing over the strands above it.
names = ["".join(c) for c in DEVCTX]; order = list(range(len(DEVCTX))); steps = []
foreign_born, foreign_end = None, None; cols_order = [order[:]]; events = []
pos_f = None
for step, i in enumerate(o1):
    j = order.index(i); order = [i] + order[:j] + order[j + 1:]; events.append(("refit", i)); cols_order.append(order[:])
    if step == 2: order = ["F"] + order; events.append(("born", "F")); cols_order.append(order[:])
order = [x for x in order if x != "F"]; events.append(("retract", "F")); cols_order.append(order[:])

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(8.3, 13.2))
fig.text(0.04, 0.988, "Study 26 at five levels: what each level keeps, and what each map loses", fontsize=11, fontweight="bold", color=INK, va="top")
fig.text(0.04, 0.971, "The purple signal is followed through every level: a table on the foreign context {M,R,Y}, refitted at step 3 and later retracted.",
         fontsize=7.6, color=MUTED, va="top")
leg = [("observed tables", "s", OBS, "none"), ("inferred laws", "o", "white", INK), ("selected states", "*", INK, INK)]
x0 = 0.04
for lab, mk, fc, ec in leg:
    fig.add_artist(plt.Line2D([x0 + 0.006], [0.956], marker=mk, color=fc, mec=ec if ec != "none" else fc, ms=6, lw=0, transform=fig.transFigure)); fig.text(x0 + 0.016, 0.956, lab, fontsize=7.2, va="center"); x0 += 0.13
for lab, c, ls_ in (("the {M,R,Y} history", PURPLE, "-"), ("coalition", AMBER, "-"), ("decision threshold", BLUE, "--"), ("twist", INK, ":")):
    fig.add_artist(plt.Line2D([x0, x0 + 0.022], [0.956, 0.956], color=c, ls=ls_, lw=1.8, transform=fig.transFigure)); fig.text(x0 + 0.026, 0.956, lab, fontsize=7.2, va="center"); x0 += {"the {M,R,Y} history": 0.14, "coalition": 0.10, "decision threshold": 0.15}.get(lab, 0.1)
rows = [0.762, 0.574, 0.386, 0.198, 0.012]; H = 0.152
def mapbox(y, title, op, keeps, loses):
    ax = fig.add_axes([0.70, y, 0.28, H]); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.0,rounding_size=0.04", fc="#f7f7f7", ec=MUTED, lw=0.6, transform=ax.transAxes))
    ax.text(0.05, 0.94, title, fontsize=7.6, fontweight="bold", color=INK, va="top", transform=ax.transAxes)
    ax.text(0.05, 0.80, op, fontsize=6.8, color=INK, va="top", transform=ax.transAxes, wrap=True)
    ax.text(0.05, 0.52, "keeps: " + keeps, fontsize=6.6, color=GREEN, va="top", transform=ax.transAxes)
    ax.text(0.05, 0.25, "loses: " + loses, fontsize=6.6, color=PURPLE, va="top", transform=ax.transAxes)
def level_title(y, n, title, sub):
    fig.text(0.04, y + H + 0.024, f"{n}. {title}", fontsize=9, fontweight="bold", color=INK); fig.text(0.04, y + H + 0.010, sub, fontsize=6.8, color=MUTED)
def arrow_down(y):
    fig.add_artist(FancyArrowPatch((0.84, y - 0.002), (0.84, y - 0.026), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=10, color=MUTED, lw=1.0))

# ---- 1. toric layer
y = rows[0]; level_title(y, 1, "Toric (Kahler) layer: the fibre in its log-linear chart", "computed; a 2-D section of the 9-dimensional fibre, chart lines projected onto it")
ax = fig.add_axes([0.06, y, 0.30, H]); ax.add_patch(Polygon(poly, closed=True, fc="#f3f6fb", ec=BLUE, lw=0.9))
for L_ in lines_t: ax.plot(L_[:, 0], L_[:, 1], color=BLUE, lw=0.45, alpha=0.6)
for L_ in lines_s: ax.plot(L_[:, 0], L_[:, 1], color=BLUE, lw=0.45, alpha=0.6)
mid = lines_s[3]; ax.plot(mid[:, 0], mid[:, 1], color=PURPLE, lw=1.6)
ax.plot(0, 0, "o", color=GREEN, ms=6, zorder=5); bx = coord(RB); ax.plot(*bx, "o", mfc="white", mec=PURPLE, mew=1.5, ms=6, zorder=5)
lx = coord(LEX); ax.plot(*lx, "*", color=INK, ms=9, zorder=6)
ax.set_aspect("equal", adjustable="datalim"); ax.set_xticks([]); ax.set_yticks([]); ax.text(0.98, 0.03, "across: trace direction; up: decision gradient", fontsize=5.9, color=MUTED, ha="right", transform=ax.transAxes)
ax2 = fig.add_axes([0.38, y, 0.30, H]); ax2.axis("off")
ax2.text(0, 0.98, "Each interior law has a log-linear (trace) coordinate,\na bijection onto the fibre's positive laws (Thm 146(ii)).", fontsize=6.9, va="top")
ax2.text(0, 0.74, "green: maximum entropy, trace 0, the Birch point of\nthe log-linear model (Prop 1(3)); a selected state", fontsize=6.9, va="top", color=GREEN)
ax2.text(0, 0.50, "purple line: the chart line along the {M,R,Y} trace;\nhollow circle: history B's end law, TV 0.089 away", fontsize=6.9, va="top", color=PURPLE)
ax2.text(0, 0.26, "star: the leximin law, a different selection: the limit of\nthe soft-min family (Prop 149(ii), Thm 27), not a chart point", fontsize=6.9, va="top")
ax2.text(0, 0.02, "torus phases sit over each interior point; not drawn", fontsize=6.6, va="bottom", color=MUTED)
mapbox(y, "map 1 to 2: expectation", "Take expectations E_P[u] of decision functionals.", "decision values, identification widths\n(support function of the fibre)", "the chart coordinate: a value does\nnot say which history reached it")
arrow_down(y)

# ---- 2. probability / decision
y = rows[1]; level_title(y, 2, "Probability and decision layer", "computed; same section in probability coordinates, with the decision hyperplane; identification widths")
ax = fig.add_axes([0.06, y, 0.30, H]); ax.add_patch(Polygon(poly, closed=True, fc="#f3f6fb", ec=BLUE, lw=0.9))
a0 = dvec @ R0; ca, cb = dvec @ d1, dvec @ d2; xl = np.array([poly[:, 0].min(), poly[:, 0].max()]); ax.plot(xl, (tau - a0 - ca * xl) / cb, color=BLUE, ls="--", lw=1.3)
ax.plot(0, 0, "o", color=GREEN, ms=6, zorder=5); ax.plot(*bx, "o", mfc="white", mec=PURPLE, mew=1.5, ms=6, zorder=5)
ps = coord(Pstar); ax.plot(*ps, "o", mfc="white", mec=INK, ms=5, zorder=5)
ax.set_aspect("equal", adjustable="datalim"); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(*ax.get_xlim())
ax.text(0.02, 0.97, f"honest {a0:+.3f}", color=GREEN, fontsize=6.6, transform=ax.transAxes, va="top")
ax.text(0.02, 0.86, f"history B {dvec @ RB:+.3f}", color=PURPLE, fontsize=6.6, transform=ax.transAxes, va="top")
ax.text(0.02, 0.75, "hollow black: truth (projected)", color=INK, fontsize=6.6, transform=ax.transAxes, va="top")
ax.text(0.98, 0.03, "Delta = 0.10", color=BLUE, fontsize=6.6, transform=ax.transAxes, ha="right")
ax2 = fig.add_axes([0.40, y + 0.012, 0.27, H - 0.03])
lad = R["ladder"]; nm = ["none", "C=0", "C=0,R=1", "C=0,R=1,H=0"]
for i, r in enumerate(lad):
    yy = 4 - i; ax2.plot(r["delta_oracle"], [yy, yy], color=BLUE, lw=4, solid_capstyle="butt")
    if r["delta_oracle"][1] - r["delta_oracle"][0] < 1e-6: ax2.plot(r["delta_oracle"][0], yy, "|", color=BLUE, ms=9, mew=2)
P_ = R["protection"]
ax2.plot([P_["gstar"]], [-0.2], "o", mfc="white", mec=INK, ms=4); ax2.plot([P_["g_pooled"]], [-0.2], "s", color=AMBER, ms=4.5); ax2.plot([P_["tau2"]] * 2, [-0.55, 0.15], color=AMBER, ls="--", lw=0.9)
ax2.axvline(tau, color=BLUE, ls="--", lw=0.9, ymin=0.3)
ax2.set_yticks([4, 3, 2, 1, -0.2]); ax2.set_yticklabels(nm + ["coalition: g"], fontsize=6.4); ax2.set_xlim(-0.45, 1.02); ax2.set_ylim(-0.8, 4.6)
ax2.tick_params(axis="x", labelsize=6.2); ax2.grid(axis="x", color=GRID, lw=0.5)
ax2.text(1.0, 0.25, f"truth {P_['gstar']:.3f}, pooled {P_['g_pooled']:.3f}", fontsize=6.2, color=AMBER, ha="right")
ax2.set_title("effect ranges as conditions are added", fontsize=7, color=INK, loc="left", pad=2)
mapbox(y, "map 2 to 3: restriction", "Restrict to contexts: P -> (pi_C P) for each table.", "local tables and their supports;\ncompatibility (is there any law?)", "the joint law: which law, and so the\ndecision; the coalition moves a point\nthat restriction still finds compatible")
arrow_down(y)

# ---- 3. contextuality / sheaf
y = rows[2]; level_title(y, 3, "Contextuality and sheaf layer", "left: observed covers (drawn from the design); right: the cycle's correlations (computed)")
ax = fig.add_axes([0.04, y, 0.31, H]); ax.set_xlim(-1.4, 3.6); ax.set_ylim(-1.45, 1.35); ax.set_aspect("equal"); ax.axis("off")
pos = {v: (np.cos(t), np.sin(t)) for v, t in zip("MCRHY", np.pi / 2 + np.arange(5) * 2 * np.pi / 5)}
for ctx, _, _ in W.CORE_ALLOC:
    pts = np.array([pos[v] for v in ctx])
    if len(ctx) == 2: ax.plot(*pts.T, color=OBS, lw=0.9)
    else:
        c = pts.mean(0); hull = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]; hull = c + 1.15 * (hull - c)
        ax.add_patch(Polygon(hull, closed=True, fc=AMBER if ctx == ("M", "C", "Y") else OBS, alpha=0.30 if ctx == ("M", "C", "Y") else 0.16, ec="none"))
pts = np.array([pos[v] for v in "MRY"]); c = pts.mean(0); ax.add_patch(Polygon(c + 1.3 * (pts - c), closed=True, fill=False, ec=PURPLE, ls="--", lw=1.1))
for v, (x, y_) in pos.items(): ax.add_patch(Circle((x, y_), 0.17, fc="white", ec=INK, lw=0.7, zorder=3)); ax.text(x, y_, v, ha="center", va="center", fontsize=7, zorder=4)
cp = {"A": (2.2, 0.8), "B": (3.3, 0.8), "D": (3.3, -0.3), "E": (2.2, -0.3)}
for (a, b) in W.CYCLE_EDGES: ax.plot([cp[a][0], cp[b][0]], [cp[a][1], cp[b][1]], color=INK, lw=1.2, ls=":" if (a, b) == ("E", "A") else "-")
for v, (x, y_) in cp.items(): ax.add_patch(Circle((x, y_), 0.15, fc="white", ec=INK, lw=0.7, zorder=3)); ax.text(x, y_, v, ha="center", va="center", fontsize=6.8, zorder=4)
ax.text(0, -1.42, "dashed purple: {M,R,Y}, seen, then\nretracted; absent from the final cover", fontsize=6.3, color=PURPLE, ha="center", va="bottom")
ax.text(2.75, -1.42, "cycle: (E,A) coded\ndifferently (the twist)", fontsize=6.3, ha="center", va="bottom")
ax = fig.add_axes([0.40, y + 0.022, 0.27, H - 0.022]); ax.add_patch(Polygon(locpoly, closed=True, fc="#f4f4f4", ec=MUTED, lw=0.7)); ax.add_patch(Polygon(ncpoly, closed=True, fc="#eaf3ea", ec=GREEN, lw=0.9))
xs_ = np.linspace(-3, 3, 10); ax.plot(xs_, xs_ - 2, color=INK, ls=":", lw=1.2)
for cdict, mk, col in ((cc, "o", GREEN), (ct, "X", INK)):
    ax.plot(cdict[("A", "B")] + cdict[("B", "D")] + cdict[("D", "E")], cdict[("E", "A")], mk, color=col, ms=6)
ax.set_xlim(-3.2, 3.2); ax.set_ylim(-1.3, 1.3); ax.set_xlabel("<AB>+<BD>+<DE>", fontsize=6.6, labelpad=1); ax.set_ylabel("<EA>", fontsize=6.6, labelpad=1); ax.grid(color=GRID, lw=0.4)
cy = R["cycle"]
ax.text(-3.1, 1.18, f"support level: {cy['consistent']['global_sections']} and {cy['twisted']['global_sections']} global sections (blind)", fontsize=5.9, va="top")
ax.text(-3.1, 0.95, f"noncontextual fraction {cy['consistent']['ncf']:.3f} vs {cy['twisted']['ncf']:.3f}", fontsize=5.9, va="top")
ax.text(3.1, -1.2, "facet (1,1,1,-1)", fontsize=5.9, ha="right")
mapbox(y, "map 3 to 4: add time", "Record the sequence of tables seen, and\nthe updater's state after each.", "the order of revisions; which contexts\nwere ever seen", "nothing about the final tables, but the\nfinal tables alone lose the history\n(two histories, same tables, different laws)")
arrow_down(y)

# ---- 4. history / transport
y = rows[3]; level_title(y, 4, "History and transport layer", "computed (Study 26), except the variable-changing row, which is the example of Prop 142")
ax = fig.add_axes([0.06, y, 0.30, H]); ax.axis("off")
rowsh = [("reorder the same tables", GREEN, "TV 2e-16: path independent (Thm 146(iii))"),
         ("retract a foreign table {M,R,Y}", PURPLE, f"TV {R['history']['tvB']:.3f}; trace in H_D + F_MRY (6e-15),\nnot in H_D + F_MHY ({R['history']['resid_MHY']:.1f})"),
         ("change the variables (merge, split)", MUTED, f"honest loss H = {KTS['H']:.2f}; outcome-dependent split\nlaunders +-{KTS['laundered'][0]:.2f} (Prop 142(iv))")]
for k, (lab, col, txt) in enumerate(rowsh):
    yy = 0.88 - k * 0.33; ax.add_patch(Rectangle((0.0, yy - 0.06), 0.05, 0.12, fc=col, ec="none", transform=ax.transAxes))
    ax.text(0.08, yy + 0.04, lab, fontsize=7, fontweight="bold", color=INK, transform=ax.transAxes, va="center")
    ax.text(0.08, yy - 0.08, txt, fontsize=6.4, color=INK, transform=ax.transAxes, va="top")
ax = fig.add_axes([0.40, y + 0.03, 0.27, H - 0.06]); Hh = R["history"]
ax.plot(Hh["fibre"], [1, 1], color=BLUE, lw=5, solid_capstyle="butt"); ax.plot(Hh["slice"], [0, 0], color=PURPLE, lw=5, solid_capstyle="butt")
ax.plot(Hh["honest"], 0.5, "o", color=GREEN, ms=5); ax.plot(Hh["histB"], 0.5, "o", mfc="white", mec=PURPLE, mew=1.4, ms=5)
ax.axvline(tau, color=BLUE, ls="--", lw=1); ax.set_yticks([1, 0.5, 0]); ax.set_yticklabels(["whole fibre", "end laws", "{M,R,Y} edits"], fontsize=6.4)
ax.set_ylim(-0.5, 1.5); ax.set_xlim(-0.45, 1.02); ax.grid(axis="x", color=GRID, lw=0.5); ax.tick_params(axis="x", labelsize=6.2)
ax.set_title("reachable values of Delta: the slice crosses 0.10", fontsize=7, color=INK, loc="left", pad=2)
mapbox(y, "map 4 to 5: encoding E (ours)", "Contexts become strands; a refit moves its\nstrand to the top, over the strands it passes;\na table seen then retracted is a strand\nborn and capped.", "the order of interactions (a braid word)", "all probability values and the size of\nthe trace: a braid invariant does not\ndetermine the law")
arrow_down(y)

# ---- 5. braid
y = rows[4]; level_title(y, 5, "Braid layer: an encoding of history B's refit order", "constructed by the rule E from the study's actual refit order; records order only")
ax = fig.add_axes([0.06, y, 0.61, H]); ax.axis("off"); ncol = len(cols_order); X = np.linspace(0.12, 0.98, ncol)
allnames = list(range(len(DEVCTX))) + ["F"]; colof = {k: (PURPLE if k == "F" else (AMBER if k == DEVCTX.index(("M", "C", "Y")) else GREEN)) for k in allnames}
nslots = 9; ypos = lambda p: 0.92 - p * (0.84 / (nslots - 1))
for j in range(ncol - 1):
    a, b = cols_order[j], cols_order[j + 1]; ev_ = events[j]
    mover = ev_[1] if ev_[0] == "refit" else None
    for k in a:
        if k not in b: continue
        ya, yb = ypos(a.index(k)), ypos(b.index(k)); xx = np.linspace(X[j], X[j + 1], 30); yy = ya + (yb - ya) * (3 * ((xx - X[j]) / (X[j + 1] - X[j])) ** 2 - 2 * ((xx - X[j]) / (X[j + 1] - X[j])) ** 3)
        top = (k == mover)
        if top: ax.plot(xx, yy, color="white", lw=4.2, transform=ax.transAxes, zorder=4)
        ax.plot(xx, yy, color=colof[k], lw=1.5 if k != "F" else 1.9, transform=ax.transAxes, zorder=5 if top else 2)
    for k in b:
        if k not in a:   # born
            ax.plot([X[j + 1] - 0.012, X[j + 1]], [ypos(b.index(k)) + 0.05, ypos(b.index(k))], color=colof[k], lw=1.9, transform=ax.transAxes)
            ax.plot(X[j + 1] - 0.012, ypos(b.index(k)) + 0.05, "o", color=PURPLE, ms=3.5, transform=ax.transAxes)
    for k in a:
        if k not in b: ax.plot(X[j] + 0.012, ypos(a.index(k)) + 0.05, "o", color=PURPLE, ms=3.5, transform=ax.transAxes); ax.plot([X[j], X[j] + 0.012], [ypos(a.index(k)), ypos(a.index(k)) + 0.05], color=PURPLE, lw=1.9, transform=ax.transAxes)
for p, k in enumerate(cols_order[0]): ax.text(X[0] - 0.01, ypos(p), "{" + ",".join(DEVCTX[k]) + "}", fontsize=6.1, ha="right", va="center", color=colof[k], transform=ax.transAxes)
for j, e in enumerate(events):
    lab = ("refit " + "".join(DEVCTX[e[1]])) if e[0] == "refit" else ("see {M,R,Y}" if e[0] == "born" else "retract")
    ax.text((X[j] + X[j + 1]) / 2, 0.995, lab, fontsize=5.6, ha="center", va="bottom", rotation=0, color=PURPLE if e[1] == "F" else MUTED, transform=ax.transAxes)
ax2 = fig.add_axes([0.70, y, 0.28, H]); ax2.axis("off")
ax2.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.0,rounding_size=0.04", fc="#f7f7f7", ec=MUTED, lw=0.6, transform=ax2.transAxes))
ax2.text(0.05, 0.94, "what the braid keeps", fontsize=7.6, fontweight="bold", color=INK, va="top", transform=ax2.transAxes)
ax2.text(0.05, 0.80, "History A is the same picture without\nthe purple strand. The difference is a\nstrand born and capped: the encoding\nof a newcomer (cf. Def. 133), whose\neffect on the law the braid does not record.\n\nCo-revision of two sources would be a\nfull twist (Cor. 130); the coalition sits\ninside one context, so this encoding\ndoes not show it.", fontsize=6.5, color=INK, va="top", transform=ax2.transAxes)
fig.savefig(os.path.join(HERE, "fig_atlas.pdf")); fig.savefig(os.path.join(HERE, "fig_atlas.png"), dpi=160)
print("ok", [len(c) for c in cols_order], o1)
