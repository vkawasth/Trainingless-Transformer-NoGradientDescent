"""Figures for the companion paper "A tangent structure for injected data". Computes from engine/pce/tangent.py; descriptive only.

fig_cones    (a) feasible cones C_x of Delta_3 at a vertex, an edge point and an interior point; (b) the universality counterexample on
             the edge: u enters the face, w leaves it, and the family x + t u + s t w stays in Delta; (c) fibre of T(p) over 0 against
             the image of the vertical lift, by face type of Delta_3 and Delta_4; (d) |T rhat| against the distance to a coincidence
fig_jets     (a) the value function along chi: a kink at the wall, where DV is not additive; (b) the obstruction [H(A, A)] around a circle
             of first-order stealthy directions at the three truths; (c) tetrad movement of a bent injection against eps
"""
import itertools, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
HERE = os.path.dirname(os.path.abspath(__file__)); ENG = os.path.join(HERE, "..", "engine"); sys.path.insert(0, ENG)
from pce import tangent as TG
from pce.tangent import HD
from pce.lab import law
from pce import injection as IJ
from pce import lifting as LF
from scipy.linalg import null_space
BLUE, ORANGE, VIOLET, AQUA, GREY, CRIT = "#2a78d6", "#eb6834", "#6250d6", "#1baf7a", "#8c8c8c", "#d03b3b"; INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
def save(fig, name): fig.savefig(os.path.join(HERE, name + ".pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, name + ".png"), dpi=150, bbox_inches="tight"); plt.close(fig)
V3 = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
bary = lambda p: np.asarray(p) @ V3
rng = np.random.default_rng(7)

fig, ax = plt.subplots(1, 4, figsize=(15.5, 3.7))
# (a)
a = ax[0]; a.add_patch(Polygon(V3, closed=True, fc="#f4f4f4", ec=INK2, lw=1)); a.set_aspect("equal"); a.axis("off")
def cone(x, dirs, col, r=0.22):
    c = bary(x); ang = sorted(np.arctan2(*(bary(np.asarray(x) + d) - c)[::-1]) for d in dirs)
    if len(dirs) == 0:
        a.add_patch(plt.Circle(c, r, fc=col, alpha=0.25, ec=col)); return
    th = np.linspace(ang[0], ang[1], 40); a.add_patch(Polygon(np.vstack([c, c + r * np.c_[np.cos(th), np.sin(th)]]), fc=col, alpha=0.3, ec=col))
cone([1, 0, 0], [np.array([-1, 1, 0]), np.array([-1, 0, 1])], ORANGE)
cone([0.5, 0.5, 0], [np.array([1, -1, 0]), np.array([-1, 1, 0])], VIOLET)
c = bary([0.5, 0.5, 0]); a.add_patch(Polygon(np.vstack([c + [-.22, 0], c + .22 * np.c_[np.cos(np.linspace(np.pi, 0, 40)), np.sin(np.linspace(np.pi, 0, 40))], c + [.22, 0]]), fc=VIOLET, alpha=0.3, ec=VIOLET))
cone([0.3, 0.35, 0.35], [], AQUA, r=0.16)
for x, t, col, off in (([1, 0, 0], "vertex: $C_x$ a cone,\nno negatives", ORANGE, (-0.33, 0.02)), ([0.5, 0.5, 0], "edge: a half-plane", VIOLET, (-0.2, -0.13)), ([0.3, 0.35, 0.35], "interior: all of $V$\n(a group)", AQUA, (-0.12, 0.17))):
    p = bary(x); a.scatter(*p, color=col, zorder=5, s=18); a.text(p[0] + off[0], p[1] + off[1], t, fontsize=7.5, color=col)
a.set_title("(a) feasible cones $C_x$ on $\\Delta_3$", fontsize=9)
# (b)
a = ax[1]; xe = np.array([0.5, 0.5, 0.0]); ue = np.array([-1.0, 0.0, 1.0]); we = np.array([0.0, 1.0, -1.0]); a.set_aspect("equal")
a.add_patch(Polygon(V3, closed=True, fc="#f4f4f4", ec=INK2, lw=1)); p0 = bary(xe)
for s, col in ((0.0, BLUE), (0.4, VIOLET), (0.8, ORANGE)):
    ts = np.linspace(0, 0.25, 30); pts = np.array([bary(xe + t * ue + s * t * we) for t in ts]); a.plot(*pts.T, color=col, lw=1.8, label=f"$x+t\\,u+s\\,t\\,w$, $s={s}$")
a.add_patch(FancyArrowPatch(p0, bary(xe + 0.25 * ue), arrowstyle="-|>", mutation_scale=12, color=BLUE, lw=1.6))
a.add_patch(FancyArrowPatch(p0, bary(xe + 0.18 * we), arrowstyle="-|>", mutation_scale=12, color=CRIT, lw=1.6))
a.text(*(bary(xe + 0.18 * we) + [0.0, -0.09]), "$w\\notin C_x$", color=CRIT, fontsize=8, ha="center")
a.text(*(bary(xe + 0.25 * ue) + [-0.07, 0.0]), "$u$", color=BLUE, fontsize=9)
a.set_xlim(0.05, 0.95); a.set_ylim(-0.2, 0.75); a.axis("off"); a.legend(frameon=False, fontsize=6.8, loc="upper left", bbox_to_anchor=(-0.05, 1.02))
a.set_title("(b) $(x,u,0,w)\\in T^2\\Delta$ but $w\\notin C_x$:\nuniversality fails on the edge", fontsize=9)
# (c)
a = ax[2]; T = json.load(open(os.path.join(ENG, "..", "lab", "tangent.json")))["universality"]
rows = []
for n in (3, 4):
    for k in range(1, n + 1):
        key = f"Delta{n} supp {tuple(range(k))}"; v = T[key]; rows.append((f"$\\Delta_{n}$, dim {k-1}", v["fibre"], v["image"], k == n))
yy = np.arange(len(rows))[::-1]; a.barh(yy + 0.18, [r[1] for r in rows], height=0.34, color=ORANGE, label="fibre of $T(p)$ over $0$")
a.barh(yy - 0.18, [r[2] for r in rows], height=0.34, color=BLUE, label="image of the vertical lift")
a.set_yticks(yy); a.set_yticklabels([r[0] + (" (interior)" if r[3] else "") for r in rows], fontsize=7.5); a.legend(frameon=False, fontsize=7, loc="center right")
for y_, r in zip(yy, rows): a.text(max(r[1], r[2]) + 3, y_, f"{r[1]}/{r[2]}", va="center", fontsize=7, color=INK2)
a.set_xlabel("grid points $(u,w)\\in\\{-1,0,1\\}^n$, zero sum"); a.set_title("(c) equal exactly on the open faces' own interiors\n(top-dimensional face = the interior)", fontsize=9)
# (d)
a = ax[3]; a1 = np.array([.5, .1, .4]); a2 = np.array([.05, .5, .1]); vv, gg = [], []
for tv in np.geomspace(0.5, 0.003, 14):
    a2c = a2.copy(); a2c[2] = a1[2] - tv * (a1[2] - a2[2]); L = law(0.6, a1, a2c); v2 = np.sqrt(0.24) * (a1[2] - a2c[2]); Lc = IJ.as_vec(L)
    vv.append(v2); gg.append(max(abs(TG.rhat_hd(HD.var(Lc, [e - e.mean()])).c[1]) for e in np.eye(8)))
sl = np.polyfit(np.log(vv), np.log(gg), 1)[0]
a.loglog(vv, gg, "o-", color=VIOLET, ms=3.5); a.set_xlabel("$|v_2|$ (distance to the coincidence $v_2=0$)"); a.set_ylabel("$\\max_x|T\\hat r(P,\\delta_x-\\bar\\delta)|$")
a.set_title(f"(d) $\\hat r$ is a morphism only off the strata:\n$|T\\hat r|\\propto|v_2|^{{{sl:.2f}}}$", fontsize=9)
plt.tight_layout(); save(fig, "fig_cones")

# ------------------------------------------------------------------ fig_jets
W = LF.we(); P = IJ.as_vec(W["P"]); CHI = IJ.chi(3, (0, 1, 2))
ua = np.array([1.0 if x == (0, 0, 0) else 0.0 for x in TG.X3]); ub = np.array([0.872 - float(x[0] or x[1]) for x in TG.X3])
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))
a = ax[0]; lo, hi = IJ.stealth_range(W["P"], (0, 1, 2)); ss = np.linspace(lo, hi, 400)
UA = [ua @ (P + s * CHI) for s in ss]; UB = [ub @ (P + s * CHI) for s in ss]; sw = 0.128 - P[TG.X3.index((0, 0, 1))]
a.plot(ss, UA, color=BLUE, lw=1, ls="--", label="$U_T$"); a.plot(ss, UB, color=ORANGE, lw=1, ls="--", label="$U_P$")
a.plot(ss, np.maximum(UA, UB), color=INK, lw=2, label="$V=\\max(U_T,U_P)$"); a.axvline(sw, color=CRIT, lw=0.8); a.axvline(0, color=GREY, lw=0.6, ls=":")
a.text(sw + 0.0006, min(UA) + 0.001, f"wall $s={sw:.3f}$:\n$DV(\\chi)+DV(-\\chi)=|\\ell(\\chi)|=1$", color=CRIT, fontsize=7.5)
a.set_xlabel("$s$ on the pair line $P+s\\chi$ (stealth range)"); a.set_ylabel("expected utility"); a.legend(frameon=False, fontsize=7, loc="center right")
a.set_title("(a) $V$ is a morphism on each chamber;\non the wall $+$ is not preserved", fontsize=9)
truths = {"generic": ([.8, .3, .7, .25], [.2, .7, .15, .7], BLUE), "one coincidence": ([.8, .3, .7, .5], [.2, .7, .15, .5], AQUA), "two coincidences": ([.8, .3, .4, .5], [.2, .7, .4, .5], CRIT)}
B16 = null_space(np.ones((1, 16))); th = np.linspace(0, np.pi, 181)
for name, (b1, b2, col) in truths.items():
    L4 = law(.35, np.array(b1), np.array(b2)); pp = IJ.as_vec(L4); J, H = TG.tetrad_jets(pp); K = B16 @ null_space(J @ B16, rcond=1e-9)
    g = np.random.default_rng(11); e = K @ g.normal(size=K.shape[1]); f = K @ g.normal(size=K.shape[1]); e /= np.linalg.norm(e); f -= (f @ e) * e; f /= np.linalg.norm(f)
    ob = [TG.lift_pair(J, H, np.cos(t) * e + np.sin(t) * f, np.cos(t) * e + np.sin(t) * f, B16)[0] for t in th]
    ax[1].plot(th, ob, color=col, lw=1.8, label=name)
    A = e; o, d2 = TG.lift_pair(J, H, A, A, B16); Bb = 0.5 * (d2 if d2 is not None else B16 @ np.linalg.lstsq(J @ B16, -np.einsum("kij,i,j->k", H, A, A), rcond=None)[0])
    es = np.geomspace(1e-3, 3e-2, 12); mv = [np.abs(IJ.tetrads(IJ.as_law(pp + t * A + t * t * Bb, 4))).max() for t in es]
    sl = np.polyfit(np.log(es), np.log(mv), 1)[0]; ax[2].loglog(es, mv, "o-", ms=3, color=col, label=f"{name}: slope {sl:.2f}")
ax[1].set_xlabel("direction $A=\\cos\\theta\\,e+\\sin\\theta\\,f$ in $\\ker J$"); ax[1].set_ylabel("$\\|[H(A,A)]\\|$ in $\\operatorname{coker}J$")
ax[1].legend(frameon=False, fontsize=7.5); ax[1].set_title("(b) $(A,A)\\in T_2V$ completes to $T^2V$ iff the class\nvanishes: only on the quadric cone, at two coincidences", fontsize=9)
ax[2].set_xlabel("$\\varepsilon$"); ax[2].set_ylabel("max |tetrad| at $P+\\varepsilon A+\\varepsilon^2B$"); ax[2].legend(frameon=False, fontsize=7.5)
ax[2].set_title("(c) the best bent injection (a 2-jet $(P,A,A,2B)$):\n$\\varepsilon^3$ where it completes, $\\varepsilon^2$ where obstructed", fontsize=9)
plt.tight_layout(); save(fig, "fig_jets")
print("ok")
