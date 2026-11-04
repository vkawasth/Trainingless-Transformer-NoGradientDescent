"""Figures for section 12 of the companion note: objects and morphisms, Stokes walls, and the Clifford shadow of selection.
  fig_morph_objects.pdf   the objects (critical points with their local systems) drawn over their torus fibres, with End = Cl_2 and no Hom
                          between distinct objects; F_1 on both sides of its wall, and the 2x3 fibre before and after one count
  fig_morph_stokes.pdf    critical values along the F_1 path, and the Stokes walls converging to the outcome wall alpha = 1/3
  fig_morph_clifford.pdf  the Clifford form Q/W of the positive object: its small eigenvalue vanishes mod T exactly above the wall"""
import os, sys, math
from fractions import Fraction as Fr
import numpy as np
import mpmath as mp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, Arc
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from pce import floermorph as FM, kahlerbridge as KB, couplingfibre as CF
BLUE, ORANGE, AQUA, VIOLET, ROSE, INK, INK2, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#c2417a", "#22211f", "#52514e", "#9a9a9a"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
TYPECOL = {"positive": BLUE, "real-signed": ROSE, "complex": GREY}; TYPEMK = {"positive": "o", "real-signed": "s", "complex": "D"}
def f1_obj(a, T, dps=150):
    v, lam = KB.F1(a); return FM.objects(v, [mp.mpf(Fr(x).limit_denominator(1000).numerator) / Fr(x).limit_denominator(1000).denominator for x in lam], T, FM.f1_critical(a, T, dps))
def cpl_obj(f, g, tau):
    pts, D = CF.critical_points(f, g, tau, dps=300); ff = [float(x) for x in f]; gf = [float(x) for x in g]; F = CF.facet_cells(ff, gf)
    V, L = CF.cell_functions(Fr(f[0]), Fr(g[0]), Fr(g[1])); T = (mp.mpf(Fr(tau).numerator) / Fr(tau).denominator) ** D; mp.mp.dps = 300
    return FM.objects([V[c] for c in F], [mp.mpf(Fr(L[c]).numerator) / Fr(L[c]).denominator for c in F], T, pts)
def hol_label(h):
    out = []
    for z in h:
        if abs(z.imag) < 1e-3: out.append(f"{z.real:+.0f}")
        elif abs(z.real) < 1e-3: out.append(("+" if z.imag > 0 else "-") + "i")
        else:
            ang = round(np.angle(z) / (2 * np.pi / 3)); out.append("ζ" if ang == 1 else "ζ²")
    return "(" + ",".join(out) + ")"
def draw_fibres(ax, poly, fibres, title, place, span):
    ax.add_patch(Polygon(poly, closed=True, fc="#f5f5f5", ec=INK2, lw=1.1))
    dx, h = 0.1 * span, 0.09 * span
    for (u, obs), (bx, by) in zip(fibres, place):
        ax.plot(*u, "x", color=INK, ms=7, mew=1.6, zorder=5)
        w = dx * (len(obs) + 0.3)
        ax.add_patch(FancyBboxPatch((bx - w / 2, by - h / 2), w, h, boxstyle=f"round,pad={0.01 * span}", fc="white", ec=INK2, lw=0.9, zorder=6))
        ax.annotate("", (u[0], u[1]), (bx, by - h / 2 - 0.01 * span if by > u[1] else by + h / 2 + 0.01 * span), arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7), zorder=4)
        for k, o in enumerate(obs):
            x = bx - w / 2 + dx * (k + 0.65); y = by + 0.012 * span
            ax.plot(x, y, TYPEMK[o["type"]], color=TYPECOL[o["type"]], ms=8, zorder=8)
            ax.add_patch(Arc((x, y + 0.022 * span), 0.035 * span, 0.035 * span, theta1=-50, theta2=230, color=TYPECOL[o["type"]], lw=1.0, zorder=8))
            ax.text(x, y - 0.022 * span, hol_label(o["hol"]), fontsize=6.6, ha="center", va="top", zorder=9, color=INK)
    ax.set_title(title, fontsize=9.5); ax.set_aspect("equal")

# ---------------------------------------------------------------- objects and morphisms
T = mp.mpf(10) ** -24
fig, axs = plt.subplots(2, 2, figsize=(9.4, 9.6)); ax = axs.ravel()
for a_, axx, place in ((0.2, ax[0], [(0.62, 0.72), (0.66, 0.42)]), (0.6, ax[1], [(0.62, 0.62)])):
    ob = f1_obj(a_, T); fib = []
    for o in ob:
        for fb in fib:
            if np.abs(fb[0] - o["u"]).max() < 2e-2: fb[1].append(o); break
        else: fib.append([o["u"], [o]])
    fib.sort(key=lambda z: -z[0][1]); poly = [(0, 0), (1, 0), (a_, 1 - a_), (0, 1 - a_)]
    draw_fibres(axx, poly, [(tuple(np.round(fb[0], 3)), sorted(fb[1], key=lambda o: o["type"] != "positive")) for fb in fib],
                f"(%s) $F_1$, $\\alpha={a_:g}$, {'below' if a_ < 1 / 3 else 'above'} the wall: {len(fib)} fibre{'s' if len(fib) > 1 else ''}, 4 objects" % ("a" if a_ < 0.5 else "b"), place, 1.0)
    axx.set_xlim(-0.05, 1.05); axx.set_ylim(-0.05, 0.9); axx.set_xlabel("$u_1$"); axx.set_ylabel("$u_2$")
for (g_, ttl, fibs, place), axx in zip(((CF.EX_G, "(c) the 2x3 example: 3 fibres, 5 objects", [(2 / 15, 2 / 15), (0.13, 0.13), (0.2, 0.1)], [(0.1, 0.36), (0.035, 0.02), (0.26, 0.27)]),
                                        ((Fr(30, 100), Fr(44, 100), Fr(26, 100)), "(d) one count later: 2 fibres, 5 objects", [(0.135, 0.135), (0.2, 0.1)], [(0.14, 0.36), (0.26, 0.27)])), ax[2:]):
    ob = cpl_obj(CF.EX_F, g_, "1/100"); G = [[] for _ in fibs]
    for o in ob: G[int(np.argmin([np.abs(o["u"] - np.array(f_)).max() for f_ in fibs]))].append(o)
    P, _ = CF.polygon(CF.EX_F, g_); PF = [(float(x), float(y)) for x, y in P]
    draw_fibres(axx, PF, [(f_, sorted(gg, key=lambda o: o["type"] != "positive")) for f_, gg in zip(fibs, G)], ttl, place, 0.42)
    axx.set_xlim(-0.04, 0.38); axx.set_ylim(-0.04, 0.44); axx.set_xlabel("$x=\\alpha_{11}$"); axx.set_ylabel("$y=\\alpha_{12}$")
for t, c in TYPECOL.items(): ax[2].plot([], [], TYPEMK[t], color=c, label={"positive": "positive object (the probabilistic one)", "real-signed": "real object with a negative coordinate", "complex": "complex object"}[t])
ax[2].plot([], [], "x", color=INK, label="its torus fibre $L(u)$"); ax[2].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(1.1, -0.12), ncol=4)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_morph_objects.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_morph_objects.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
if os.environ.get("ONLY_OBJECTS"): sys.exit(0)

# ---------------------------------------------------------------- Stokes walls
fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.3))
al = np.linspace(0.05, 0.95, 181)
def solv(a_):
    ob = f1_obj(a_, T); return [o["y"] for o in ob], [o["W"] for o in ob]
P, Wt = FM.track(al, solv); types0 = [o["type"] for o in f1_obj(0.05, T)]
ob0 = f1_obj(0.05, T); P0 = np.array([[complex(x) for x in o["y"]] for o in ob0])
from scipy.optimize import linear_sum_assignment
_, col = linear_sum_assignment(sum(np.abs(P[0][None, :, i] - P0[:, None, i]) / np.abs(P0[:, None, i]) for i in range(2))); lab = {col[r]: types0[r] for r in range(4)}
a = ax[0]; seen = set()
for k in range(4):
    t = lab[k]; a.plot(al, np.log10(np.abs(Wt[:, k])), color=TYPECOL[t], lw=2, ls="-" if t != "complex" else "--", label=t if t not in seen else None); seen.add(t)
a.axvline(1 / 3, color=INK, lw=0.8, ls=":"); a.text(1 / 3 + 0.01, -0.6, "outcome wall", fontsize=8)
a.set_xlabel("$\\alpha$"); a.set_ylabel("$\\log_{10}|W(\\mathrm{crit})|$ at $T=10^{-24}$"); a.legend(frameon=False, fontsize=7.5, loc="lower right")
a.set_title("(a) Critical values: the signed object's value dominates below the wall,\nthe barycentre's above; their sizes cross at $\\alpha=1/3$", fontsize=9.5)
b = ax[1]
for th, c_ in ((0.3, BLUE), (1.0, ORANGE), (math.pi / 2, AQUA)):
    xs, ys = [], []
    for kk in (12, 24, 48):
        Tk = mp.mpf(10) ** -kk; Lk = kk * math.log(10); al2 = np.linspace(0.2, 0.5, 401)
        def sv(a_):
            ob = f1_obj(a_, Tk); return [o["y"] for o in ob], [o["W"] for o in ob]
        _, W2 = FM.track(al2, sv)
        for x, i, j in FM.stokes_events(W2, al2, th): xs.append(1 / Lk); ys.append(x)
    b.plot(xs, ys, "o", color=c_, ms=5, label=f"Stokes walls, phase $\\theta={th:.2g}$")
b.axhline(1 / 3, color=INK, lw=1, ls="--"); b.text(0.001, 1 / 3 + 0.003, "outcome wall $\\alpha=1/3$", fontsize=8)
b.set_xlim(0, 0.04); b.set_xlabel("$1/\\log(1/T)$"); b.set_ylabel("$\\alpha$ at which two critical values align"); b.legend(frameon=False, fontsize=7.5, loc="lower right")
b.set_title("(b) Mutations happen at Stokes walls; for every phase they\nconverge to the outcome wall, linearly in $1/\\log(1/T)$", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_morph_stokes.pdf")); fig.savefig(os.path.join(HERE, "fig_morph_stokes.png"), dpi=150); plt.close(fig)

# ---------------------------------------------------------------- the Clifford shadow
fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.3)); a = ax[0]; al = np.linspace(0.05, 0.95, 91); ramp = ["#b7d3f2", "#2a78d6", "#123f7a"]
for c_, kk in zip(ramp, (12, 24, 48)):
    Tk = mp.mpf(10) ** -kk; E = []
    for a_ in al:
        p = [o for o in f1_obj(a_, Tk, dps=160) if o["type"] == "positive"][0]; E.append(np.sort(np.linalg.eigvalsh(p["Qn"].real)))
    E = np.array(E); a.semilogy(al, E[:, 0], color=c_, lw=2, label=f"small eigenvalue, $T=10^{{-{kk}}}$"); a.semilogy(al, E[:, 1], color=c_, lw=1, ls=":")
a.axvline(1 / 3, color=INK, lw=0.8, ls="--"); a.text(0.345, 2e-9, "outcome wall", fontsize=8)
a.set_xlabel("$\\alpha$"); a.set_ylabel("eigenvalues of $Q/W$ (positive object)"); a.legend(frameon=False, fontsize=7.5, loc="lower left")
a.set_title("(a) $F_1$: above the wall the small eigenvalue is $\\sim T^{(3\\alpha-1)/4}$,\nso the Clifford form reduced mod $T$ drops from rank 2 to rank 1", fontsize=9.5)
b = ax[1]
for g_, c_, lab_ in ((CF.EX_G, BLUE, "the example (row 1 binds)"), ((Fr(30, 100), Fr(44, 100), Fr(26, 100)), ROSE, "one count later (column 3 binds)")):
    xs, ys, ys2 = [], [], []
    for tau in ("1/10", "1/100", "1/1000", "1/10000", "1/100000"):
        pts, D = CF.critical_points(CF.EX_F, g_, tau, dps=300); ff = [0.4, 0.6]; gf = [float(x) for x in g_]; F = CF.facet_cells(ff, gf)
        V, L = CF.cell_functions(Fr(2, 5), Fr(g_[0]), Fr(g_[1])); Tq = (mp.mpf(Fr(tau).numerator) / Fr(tau).denominator) ** D; mp.mp.dps = 300
        ob = FM.objects([V[c] for c in F], [mp.mpf(Fr(L[c]).numerator) / Fr(L[c]).denominator for c in F], Tq, pts); p = [o for o in ob if o["type"] == "positive"][0]
        ev = np.sort(np.linalg.eigvalsh(p["Qn"].real)); xs.append(float(-mp.log10(Tq))); ys.append(ev[0]); ys2.append(ev[1])
    b.semilogy(xs, ys, "o-", color=c_, lw=2, label=lab_ + ": small eigenvalue"); b.semilogy(xs, ys2, "o:", color=c_, lw=1)
b.set_xscale("log"); b.set_xlabel("$\\log_{10}(1/T)$"); b.set_ylabel("eigenvalues of $Q/W$"); b.legend(frameon=False, fontsize=7.5, loc="lower left")
b.set_title("(b) The $2\\times3$ fibre: one count turns a nondegenerate\nreduced form into a degenerate one (valuation $1/200$)", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_morph_clifford.pdf")); fig.savefig(os.path.join(HERE, "fig_morph_clifford.png"), dpi=150); plt.close(fig)
print("ok")
