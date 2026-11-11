"""The tower (overview): Study 26 seen at seven levels, signals keeping their colour throughout. Companion of fig_atlas.py.
The atlas, from the toric (Kahler) layer to braids, with labelled maps between levels.
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


# ------------------------------------------------------------------ extra computations for the tower
from scipy.stats import chi2
from pce import selectorlink as SLK
from matplotlib.patches import Ellipse, PathPatch
from matplotlib.path import Path
core = [s for s in dev if s["ctx"] != ("L3",)]; by = {}
for s in core: by.setdefault(s["ctx"], []).append(s)
Fl = set()
for ctx, ss in by.items():
    for s in ss:
        o = sum(x["counts"] for x in ss if x is not s); e = s["n"] * o / o.sum(); m = e > 1e-9
        if chi2.sf(float((((s["counts"] - e) ** 2)[m] / e[m]).sum()), m.sum() - 1) < 0.001: Fl.add(s["id"])
pool = {}
for s in core:
    if s["id"] not in Fl: pool[s["ctx"]] = pool.get(s["ctx"], 0) + s["counts"]
def pM1(ctx, cnt):
    k = ctx.index("M"); tot = cnt.sum(); return sum(c for i, c in enumerate(cnt) if format(i, f"0{len(ctx)}b")[k] == "1") / tot, tot
f = np.array(R["selection"]["f"]); g = np.array(R["selection"]["g"]); lexc = np.array(R["selection"]["lex"])
Vc, lamc = SLK.cells(tuple(f), tuple(g))
cpoly = section_polygon(-lamc, Vc[:, 0], Vc[:, 1])
def ipf2(a, b):
    Q = np.outer(f, g) * np.exp(np.array([[a, b, 0.0], [0.0, 0.0, 0.0]]))
    for _ in range(3000): Q *= (f / Q.sum(1))[:, None]; Q *= (g / Q.sum(0))[None, :]
    return Q[0, 0], Q[0, 1]
grid_a = [np.array([ipf2(a, b) for b in np.linspace(-6, 6, 60)]) for a in np.linspace(-3, 3, 9)]
grid_b = [np.array([ipf2(a, b) for a in np.linspace(-6, 6, 60)]) for b in np.linspace(-3, 3, 9)]
toric_path = np.array([SLK.toric(tuple(f), tuple(g), L) for L in np.logspace(-1, 3, 60)])
me = np.array([f[0] * g[0], f[0] * g[1]]); lsq = SLK.leastsq(tuple(f), tuple(g))
Lc = 12.0; xs = np.linspace(cpoly[:, 0].min() - 0.06, cpoly[:, 0].max() + 0.06, 220); ys = np.linspace(cpoly[:, 1].min() - 0.06, cpoly[:, 1].max() + 0.06, 220)
XX, YY = np.meshgrid(xs, ys); cellv = np.stack([Vc[k, 0] * XX + Vc[k, 1] * YY - lamc[k] for k in range(6)])
Fpot = -np.log(np.exp(-Lc * cellv).sum(0)) / Lc
crit = np.array([SLK.toric(tuple(f), tuple(g), L) for L in (3, 6, 12, 24, 48, 96, 400)])
geo = []
for sgm in np.linspace(-1.5, 3.0, 46):
    Q = R0 * np.exp(sgm * t1); Q /= Q.sum()
    try: geo.append(coord(refit(Q, DEVCTX)))
    except RuntimeError: pass
geo = np.array(geo)

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(8.3, 11.8))
gsp = fig.add_gridspec(4, 2, height_ratios=[1, 1.08, 1.08, 1.0], hspace=0.48, wspace=0.30, left=0.09, right=0.98, top=0.905, bottom=0.045)
fig.text(0.04, 0.985, "One collection of tables, seen at every level of the tower", fontsize=11, fontweight="bold", color=INK, va="top")
fig.text(0.04, 0.965, "Study 26's data. Each signal keeps its colour at every level. Panels 1 and 7 are drawn; the others are computed.", fontsize=7.8, color=MUTED, va="top")
x0 = 0.04
for lab, c, ls_ in (("honest", GREEN, "-"), ("coalition", AMBER, "-"), ("history ({M,R,Y} table)", PURPLE, "-"), ("decision threshold", BLUE, "--"), ("twist", INK, ":")):
    fig.add_artist(plt.Line2D([x0, x0 + 0.025], [0.943, 0.943], color=c, ls=ls_, lw=2, transform=fig.transFigure)); fig.text(x0 + 0.03, 0.943, lab, fontsize=7.6, color=INK, va="center")
    x0 += {"honest": 0.10, "coalition": 0.11, "history ({M,R,Y} table)": 0.21, "decision threshold": 0.17}.get(lab, 0.1)
def level(ax, n, name, sub):
    ax.set_title(f"{n}  {name}", loc="left", color=INK, pad=12, fontsize=8.6, fontweight="bold"); ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=6.7, color=MUTED, va="bottom")

# 1 tables and sheaf (drawn from the design)
ax = fig.add_subplot(gsp[0, 0]); ax.set_xlim(-1.35, 3.5); ax.set_ylim(-1.45, 1.3); ax.set_aspect("equal"); ax.axis("off")
level(ax, "1", "tables and their sheaf", "contexts as hyperedges; supports form the sheaf (drawn from the design)")
pos = {v: (np.cos(t), np.sin(t)) for v, t in zip("MCRHY", np.pi / 2 + np.arange(5) * 2 * np.pi / 5)}
for ctx, _, _ in W.CORE_ALLOC:
    pts = np.array([pos[v] for v in ctx])
    if len(ctx) == 2: ax.plot(*pts.T, color=MUTED, lw=1.0, alpha=0.8)
    else:
        c = pts.mean(0); hull = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]; hull = c + 1.12 * (hull - c)
        ax.add_patch(Polygon(hull, closed=True, fc=(AMBER if ctx == ("M", "C", "Y") else BLUE), alpha=0.28 if ctx == ("M", "C", "Y") else 0.13, ec="none"))
for v, (x, y_) in pos.items(): ax.add_patch(Circle((x, y_), 0.17, fc="white", ec=INK, lw=0.8, zorder=3)); ax.text(x, y_, v, ha="center", va="center", fontsize=8, zorder=4)
ax.text(0, -1.42, "{M,C,Y} carries the coalition", ha="center", fontsize=6.8, color=AMBER)
cpos = {"A": (2.1, 0.8), "B": (3.1, 0.8), "D": (3.1, -0.2), "E": (2.1, -0.2)}
for (a, b) in W.CYCLE_EDGES: ax.plot([cpos[a][0], cpos[b][0]], [cpos[a][1], cpos[b][1]], color=INK, lw=1.3, ls=":" if (a, b) == ("E", "A") else "-")
for v, (x, y_) in cpos.items(): ax.add_patch(Circle((x, y_), 0.15, fc="white", ec=INK, lw=0.8, zorder=3)); ax.text(x, y_, v, ha="center", va="center", fontsize=7.5, zorder=4)
ax.text(2.6, -0.62, "holdout cycle; (E,A) coded\ndifferently: the twist", ha="center", va="top", fontsize=6.6, color=INK)
ax.text(2.6, -1.42, "supports: all 16 cells positive", ha="center", fontsize=6.6, color=MUTED)

# 2 data space
ax = fig.add_subplot(gsp[0, 1]); level(ax, "2", "data: the polytope of margins", "each table's reading of P(M = 1); coherent data lie on the diagonal")
ref, _ = pM1(("M", "C", "R", "H"), pool[("M", "C", "R", "H")]); offs = {("M", "C"): 0.004, ("M", "R"): 0.004, ("M", "H"): 0.014, ("M", "C", "Y"): 0.004}
for ctx, cnt in pool.items():
    if "M" not in ctx or ctx == ("M", "C", "R", "H"): continue
    yv, n = pM1(ctx, cnt); se = np.sqrt(yv * (1 - yv) / n); col = BLUE if ctx == ("M", "C") else (AMBER if ctx == ("M", "C", "Y") else GREEN)
    ax.errorbar(ref, yv, yerr=3 * se, fmt="o", color=col, ms=4.5, capsize=2, lw=0.9)
    ax.text(ref + offs.get(ctx, 0.004), yv + (0.004 if ctx == ("M", "R") else (-0.004 if ctx == ("M", "H") else 0)), "{" + ",".join(ctx) + "}", fontsize=6.4, color=col, va="center")
lo, hi = 0.64, 0.75; ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.9); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ym, _ = pM1(("M", "C"), pool[("M", "C")]); ax.annotate("", xy=(ym, ym), xytext=(ref + 0.013, ym), arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
ax.text(0.645, 0.745, "pooled {M,C} is off the diagonal (the\nother population); the elastic step moved\n{M,C,Y} instead: attribution", fontsize=6.3, color=INK, va="top")
ax.set_xlabel("P(M = 1) read from {M,C,R,H}", fontsize=6.8); ax.set_ylabel("P(M = 1) from each other table", fontsize=6.8); ax.grid(color=GRID, lw=0.5)

# 3 fibre section
ax = fig.add_subplot(gsp[1, 0]); level(ax, "3", "the fibre: laws matching the tables", "2-D section of the 9-dim. polytope through the honest law; in the Fisher kernel")
ax.add_patch(Polygon(poly, closed=True, fc="#eef3f9", ec=BLUE, lw=1.0))
a0 = dvec @ R0; ca, cb = dvec @ d1, dvec @ d2; xl = np.array([poly[:, 0].min(), poly[:, 0].max()]); ax.plot(xl, (tau - a0 - ca * xl) / cb, color=BLUE, ls="--", lw=1.3)
ax.plot(geo[:, 0], geo[:, 1], color=PURPLE, lw=1.4)
ax.plot(0, 0, "o", color=GREEN, ms=7, zorder=5); ax.annotate(f"honest {a0:+.3f}", (0, 0), (-0.045, -0.035), fontsize=6.8, color=GREEN, arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.6))
bx = coord(RB); ax.plot(*bx, "s", color=PURPLE, ms=6, zorder=5); ax.annotate(f"history B {dvec @ RB:+.3f}", tuple(bx), (bx[0] + 0.01, 0.06), fontsize=6.8, color=PURPLE, arrowprops=dict(arrowstyle="-", color=PURPLE, lw=0.6))
ps = coord(Pstar); ax.plot(*ps, "o", mfc="white", mec=INK, ms=5, zorder=5); ax.annotate("truth (projected)", tuple(ps), (-0.07, 0.07), fontsize=6.6, color=INK, arrowprops=dict(arrowstyle="-", color=INK, lw=0.5))
ax.set_xlabel("along the history trace (purple: the trace's chart line, projected)", fontsize=6.6); ax.set_ylabel("along the decision gradient", fontsize=6.8)
ax.set_aspect("equal", adjustable="datalim"); ax.grid(color=GRID, lw=0.5); ax.tick_params(labelsize=6)

# 4 cycle
ax = fig.add_subplot(gsp[1, 1]); level(ax, "4", "contextuality: the noncontextual polytope", "cycle correlations, two coordinates; grey: local data, green: global law")
ax.add_patch(Polygon(locpoly, closed=True, fc="#f4f4f4", ec=MUTED, lw=0.8)); ax.add_patch(Polygon(ncpoly, closed=True, fc="#eaf3ea", ec=GREEN, lw=1.0))
xs_ = np.linspace(-3, 3, 10); ax.plot(xs_, xs_ - 2, color=INK, ls=":", lw=1.3); ax.text(2.95, -1.35, "facet (1,1,1,-1)", fontsize=6.4, ha="right")
for cd, mk, col, lab, dy in ((cc, "o", GREEN, "consistent coding", 0.17), (ct, "X", INK, "coding changed", -0.22)):
    x = cd[("A", "B")] + cd[("B", "D")] + cd[("D", "E")]; y_ = cd[("E", "A")]; ax.plot(x, y_, mk, color=col, ms=7); ax.text(x - 0.15, y_ + dy, lab, fontsize=6.8, color=col, ha="right")
cy = R["cycle"]; ax.text(-3.05, 1.3, f"supports: {cy['twisted']['global_sections']} global sections either way (blind);\nnoncontextual fraction {cy['consistent']['ncf']:.3f} vs {cy['twisted']['ncf']:.3f}", fontsize=6.3, va="top")
ax.set_xlim(-3.2, 3.2); ax.set_ylim(-1.5, 1.45); ax.set_xlabel("<AB> + <BD> + <DE>", fontsize=6.8); ax.set_ylabel("<EA>", fontsize=6.8); ax.grid(color=GRID, lw=0.5)

# 5 toric chart of the coupling
ax = fig.add_subplot(gsp[2, 0]); level(ax, "5", "toric layer: the coupling M x light", "moment polygon, log-linear grid (the torus chart), maximum entropy")
ax.add_patch(Polygon(cpoly, closed=True, fc="#f7f7fb", ec=INK, lw=0.9))
for line in grid_a + grid_b: ax.plot(line[:, 0], line[:, 1], color=BLUE, lw=0.45, alpha=0.55)
for (x, y_) in ((0.07, 0.24), (0.21, 0.04)): ax.add_patch(Ellipse((x, y_), 0.03, 0.012, fill=False, ec=MUTED, lw=0.6))
ax.plot(*me, "o", color=GREEN, ms=6, zorder=5); ax.annotate("maximum entropy\n(product of margins)", tuple(me), (0.24, 0.2), fontsize=6.6, color=GREEN, arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.6))
f1, g3 = f[0], g[2]; xl = np.array([-0.05, 0.36]); ax.plot(xl, 0.10 - g3 + f1 - xl, color=BLUE, ls="--", lw=1.2)
ax.set_xlabel("P(M = 0, dim)   (dashed: P(M=1, bright) = 0.10; ellipses: torus fibres)", fontsize=6.4); ax.set_ylabel("P(M = 0, moderate)", fontsize=6.8)
ax.set_xlim(-0.06, 0.40); ax.set_ylim(-0.08, 0.36); ax.set_aspect("equal", adjustable="box"); ax.grid(color=GRID, lw=0.5); ax.tick_params(labelsize=6)

# 6 tropical / potential
ax = fig.add_subplot(gsp[2, 1]); level(ax, "6", "tropical layer: the potential and its limit", "soft-min of cells, L = 12; critical points as T -> 0 (Prop 149(ii))")
ax.contourf(XX, YY, Fpot, levels=14, cmap="Blues", alpha=0.55); ax.add_patch(Polygon(cpoly, closed=True, fill=False, ec=INK, lw=0.9))
ax.plot(toric_path[:, 0], toric_path[:, 1], color=INK, lw=1.0); ax.plot(crit[:, 0], crit[:, 1], "o", color=INK, ms=2.6)
ax.plot(lexc[0], lexc[1], "*", ms=11, mfc="white", mec=INK, zorder=6); ax.annotate("leximin (the limit)", (lexc[0], lexc[1]), (-0.03, 0.30), fontsize=6.6, color=INK, arrowprops=dict(arrowstyle="-", color=INK, lw=0.6), bbox=dict(fc="white", ec="none", pad=0.5))
ax.plot(*lsq, "^", color=INK, ms=4.5); ax.annotate("least squares (T -> 1)", tuple(lsq), (0.16, 0.27), fontsize=6.6, color=INK, arrowprops=dict(arrowstyle="-", color=INK, lw=0.6), bbox=dict(fc="white", ec="none", pad=0.5))
ax.plot(xl, 0.10 - g3 + f1 - xl, color=BLUE, ls="--", lw=1.2)
ax.set_xlabel(f"P(M = 0, dim)   (leximin sits {abs(lexc[5] - 0.10):.4f} from the threshold)", fontsize=6.4); ax.set_ylabel("P(M = 0, moderate)", fontsize=6.8)
ax.set_xlim(-0.06, 0.40); ax.set_ylim(-0.08, 0.36); ax.set_aspect("equal", adjustable="box"); ax.tick_params(labelsize=6)

# 7 braid (schematic)
ax = fig.add_subplot(gsp[3, :]); level(ax, "7", "braids: the revision history", "schematic: strands are reporting groups over revision time (not computed in Study 26; numbers are the study's)")
ax.set_xlim(-0.2, 10.4); ax.set_ylim(-1.75, 6.6); ax.axis("off")
names = ["exposure", "circadian", "history", "outcome (honest)", "coalition", "overlap"]; cols = [GREEN, GREEN, GREEN, GREEN, AMBER, GREEN]; ys0 = [5, 4, 3, 2, 1, 0]
def strand(xa, ya, xb, yb, col, lw=1.6, under=False):
    verts = [(xa, ya), ((xa + xb) / 2, ya), ((xa + xb) / 2, yb), (xb, yb)]
    if under: ax.add_patch(PathPatch(Path(verts, [1, 4, 4, 4]), fc="none", ec="white", lw=lw + 3.2, zorder=1))
    ax.add_patch(PathPatch(Path(verts, [1, 4, 4, 4]), fc="none", ec=col, lw=lw, zorder=3 if under else 2))
X = [0.9, 2.4, 3.9, 5.4, 6.9, 8.4]
for n, c, y_ in zip(names, cols, ys0): ax.text(0.8, y_, n, ha="right", va="center", fontsize=7, color=c)
for j in range(len(X) - 1):
    for i, y_ in enumerate(ys0):
        if i in (3, 4) and j in (1, 2): continue
        strand(X[j], y_, X[j + 1], y_, cols[i])
for j in (1, 2): strand(X[j], 2, X[j + 1], 1, GREEN, under=(j == 2)); strand(X[j], 1, X[j + 1], 2, AMBER, under=(j == 1))
verts = [(4.9, 6.3), (5.2, 5.6), (5.5, 3.6), (6.0, 3.5), (6.6, 3.5), (7.1, 3.6), (7.5, 5.6), (7.9, 6.3)]
ax.add_patch(PathPatch(Path(verts, [1, 4, 4, 4, 2, 4, 4, 4]), fc="none", ec=PURPLE, lw=1.8, zorder=4))
ax.text(0.9, 6.5, "the {M,R,Y} table enters and is retracted:\nhistory trace TV 0.089, attributed to {M,R,Y}", fontsize=6.6, color=PURPLE, va="top")
ax.text(4.65, -0.75, "outcome and coalition co-revise: a full twist, the square of a band generator (Cor. 130)", ha="center", fontsize=6.6, color=AMBER)
ax.add_patch(FancyBboxPatch((0.845, -0.12), 0.152, 1.0, clip_on=False, boxstyle="round,pad=0.0,rounding_size=0.02", fc="#f6f6f6", ec=MUTED, lw=0.6, transform=ax.transAxes))
ax.text(0.921, 0.85, "what each signal\nbecomes up the tower", ha="center", va="top", fontsize=6.6, fontweight="bold", color=INK, transform=ax.transAxes)
ax.text(0.853, 0.70, "coalition: a coherent\nmove inside the data\npolytope (2); a full\ntwist (7)\n\nhistory: a point off\nthe honest law (3);\na strand that enters\nand leaves (7)\n\ntwist: data across\na facet (4); invisible\nto supports (1)", va="top", fontsize=5.9, color=INK, linespacing=1.15, transform=ax.transAxes)
for x in X: ax.plot([x, x], [-0.3, 5.3], color=GRID, lw=0.5, zorder=0)
fig.savefig(os.path.join(HERE, "fig_tower.pdf")); fig.savefig(os.path.join(HERE, "fig_tower.png"), dpi=160)
print("ok")
