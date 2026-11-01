"""Figures for "Probability, Context and Differentiation: The Layers", Step 1.
fig_architecture: the layers and which step builds each; fig_bayes: the contextual Bayes decomposition on the worked example;
fig_wall: conditioning near the boundary of its domain; fig_forget: forgetting provenance on the worked example.
Step 2: fig_face: the source face, injection and relabelling directions; fig_sens: sensitivity of the worked example to the source share; fig_jumps: strata and jumps. Step 3: fig_orders: orders of transitions. Step 4a-4b: fig_fan: several parameters, the order fan. Step 4c: fig_para: the decision fan and two study designs. Step 4d: fig_stat: the statistical fan; fig_graded: the Poisson band and the graded classes. Step 4f: fig_degen: recursive refinement. Step 5: fig_duality: logs and the cotangent cone."""
import os, sys, itertools
from fractions import Fraction as F
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from pce import context as CT
BLUE, ORANGE, VIOLET, AQUA, GREY, INK2 = "#2a78d6", "#eb6834", "#6250d6", "#1baf7a", "#8c8c8c", "#52514e"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})

# ---- architecture
fig, ax = plt.subplots(figsize=(8.6, 2.9)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0.35, 3.55)
boxes = [(0.2, "probability\nmonad $\\mathcal{D}$", "laws, kernels,\nmixtures ($\\eta,\\mu$)", BLUE, "Step 1"),
         (2.9, "provenance\ncomonad $W=L\\times(-)$", "source labels\n($\\epsilon,\\delta$)", ORANGE, "Step 1"),
         (5.6, "tangent structure $T$", "feasible directions\nat the boundary", VIOLET, "Step 2"),
         (8.3, "orders of transitions\n$\\mathrm{ord}$: germs $\\to(\\min,+)$", "at which order a\ntransition is seen", AQUA, "Step 3")]
for x, t, sub, c, st in boxes:
    ax.add_patch(FancyBboxPatch((x, 1.5), 2.4, 1.6, boxstyle="round,pad=0.03,rounding_size=0.12", fc="white", ec=c, lw=1.8))
    ax.text(x + 1.2, 2.62, t, ha="center", va="center", fontsize=8.6, color=c, weight="bold"); ax.text(x + 1.2, 1.95, sub, ha="center", va="center", fontsize=7.8, color=INK2)
    ax.text(x + 1.2, 3.3, st, ha="center", fontsize=8.5, color=INK2, style="italic")
for x0, x1, lab in ((2.6, 2.9, "$\\lambda$"), (5.3, 5.6, ""), (8.0, 8.3, "")):
    ax.annotate("", (x1, 2.3), (x0, 2.3), arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.2)); ax.text((x0 + x1) / 2, 2.45, lab, ha="center", fontsize=10)
ax.text(1.3, 1.05, "marginalisation: total, affine", ha="center", fontsize=8.2, color=BLUE)
ax.text(4.9, 1.05, "forgetting provenance = marginalising over $L$", ha="center", fontsize=8.2, color=ORANGE)
ax.text(2.75, 0.55, "conditioning: partial, nonlinear; contextual Bayes reweights the sources", ha="center", fontsize=8.5, color=INK2)
ax.text(9.5, 1.05, "Steps 4-5: fans, decisions, Theorem J,\ngraded classes, duality and logs", ha="center", fontsize=8.2, color=INK2, style="italic")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_architecture.pdf")); fig.savefig(os.path.join(HERE, "fig_architecture.png"), dpi=150); plt.close(fig)

# ---- contextual Bayes on the worked example
p1 = {(1, 1): F(3, 10), (1, 0): F(1, 2), (0, 1): F(1, 10), (0, 0): F(1, 10)}; p2 = {(1, 1): F(1, 5), (1, 0): F(3, 10), (0, 1): F(3, 10), (0, 0): F(1, 5)}
pis = [F(1, 4), F(3, 4)]; A = lambda x: x[0] == 1; piA = CT.bayes_weights(pis, [p1, p2], A)
pooled = CT.pool(pis, [p1, p2]); right = CT.prob(CT.cond(pooled, A), lambda x: x[1] == 1)
naive = CT.prob(CT.pool(pis, [CT.cond(p1, A), CT.cond(p2, A)]), lambda x: x[1] == 1)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw=dict(width_ratios=[1.1, 1]))
a = ax[0]; xs = np.arange(2); w = 0.36
a.bar(xs - w / 2, [float(v) for v in pis], w, color=GREY, label="before: $\\pi_l$")
a.bar(xs + w / 2, [float(v) for v in piA], w, color=ORANGE, label="after observing $R=1$: $\\pi^A_l$")
for i in range(2):
    a.text(i - w / 2, float(pis[i]) + 0.02, f"{pis[i]}", ha="center", fontsize=9); a.text(i + w / 2, float(piA[i]) + 0.02, f"{piA[i]}", ha="center", fontsize=9)
a.set_xticks(xs); a.set_xticklabels(["source 1 (small trials)\n$p_1(R{=}1)=4/5$", "source 2 (registry)\n$p_2(R{=}1)=1/2$"]); a.set_ylim(0, 0.95); a.set_ylabel("source weight")
a.legend(frameon=False, fontsize=8.5, loc="upper left"); a.set_title("(a) conditioning on $R=1$ moves weight to the source\nwhere $R=1$ is more likely", fontsize=9.5)
b = ax[1]; vals = [float(right), float(naive)]
b.barh([1, 0], vals, color=[BLUE, GREY], height=0.5)
b.text(vals[0] + 0.005, 1, f"{right} = {float(right):.4f}", va="center", fontsize=9); b.text(vals[1] + 0.005, 0, f"{naive} = {float(naive):.4f}", va="center", fontsize=9)
b.set_yticks([1, 0]); b.set_yticklabels(["Bayes weights $\\pi^A$\n(correct)", "old weights $\\pi$\n(wrong)"]); b.set_xlim(0.36, 0.43); b.set_xlabel("$P(H=1\\mid R=1)$ in the pooled population")
b.set_title("(b) pooling the conditioned sources needs\nthe updated weights", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_bayes.pdf")); fig.savefig(os.path.join(HERE, "fig_bayes.png"), dpi=150); plt.close(fig)

# ---- conditioning near the wall
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw=dict(width_ratios=[1.1, 1]))
a = ax[0]; V = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])          # vertices a, b, c
n = 120; pts = []; cols = []
for i in range(n + 1):
    for j in range(n + 1 - i):
        k = n - i - j; p = np.array([i, j, k]) / n
        if p[0] + p[1] == 0: continue
        pts.append(p @ V); cols.append(p[0] / (p[0] + p[1]))
pts = np.array(pts); sc = a.scatter(pts[:, 0], pts[:, 1], c=cols, cmap="coolwarm", s=6, vmin=0, vmax=1)
a.add_patch(Polygon(V, closed=True, fill=False, ec=INK2, lw=1))
for al, c in ((0.2, BLUE), (0.5, GREY), (0.8, ORANGE)):
    ts = np.linspace(0.02, 1, 50); path = np.array([[t * al, t * (1 - al), 1 - t] for t in ts]) @ V; a.plot(path[:, 0], path[:, 1], color=c, lw=2)
    a.text(*(np.array([al, 1 - al, 0]) @ V + np.array([0, -0.07])), f"$C_A=({al},{1 - al:.1f})$", ha="center", fontsize=8, color=c)
a.plot(*V[2], "o", color="black", ms=7); a.text(V[2][0] + 0.04, V[2][1], "wall $p(A)=0$:\nevery limit occurs", fontsize=8.5, va="center")
a.text(V[0][0] - 0.03, V[0][1] - 0.03, "$a$", ha="right"); a.text(V[1][0] + 0.03, V[1][1] - 0.03, "$b$"); a.text(V[2][0] - 0.05, V[2][1] + 0.02, "$c$", ha="right")
a.set_aspect("equal"); a.axis("off"); plt.colorbar(sc, ax=a, fraction=0.04, pad=0.02, label="$C_A(p)(a)$, $A=\\{a,b\\}$")
a.set_title("(a) $C_A$ is constant on rays into $c$, so it has\nno continuous extension at the wall", fontsize=9.5)
b = ax[1]; rng = np.random.default_rng(1); h = np.array([0.7, -0.2, -0.5])
qs = np.logspace(-4, -0.3, 40); norms = []
for q in qs:
    p = np.array([0.6 * q, 0.4 * q, 1 - q]); N = np.array([1, 1, 0.0]); norms.append(np.abs(N * h / q - N * p * (N * h).sum() / q ** 2).max())
b.loglog(qs, norms, color=VIOLET, lw=2, label="$\\|DC_A(p)[h]\\|$"); b.loglog(qs, norms[-1] * qs[-1] / qs, color=GREY, ls="--", lw=1, label="slope $-1$")
b.set_xlabel("$p(A)$"); b.set_ylabel("size of the derivative"); b.legend(frameon=False, fontsize=8.5)
b.set_title("(b) approaching the wall, the derivative\ngrows like $1/p(A)$", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_wall.pdf")); fig.savefig(os.path.join(HERE, "fig_wall.png"), dpi=150); plt.close(fig)

# ---- forgetting provenance
P = {(x, y): F(1, 4) for x in (0, 1) for y in (0, 1)}; Q = {(x, y): F(1, 8) * (1 if x == y else -1) for x in (0, 1) for y in (0, 1)}
pa = {k: P[k] + Q[k] for k in P}; pb = {k: P[k] - Q[k] for k in P}
fig, ax = plt.subplots(1, 4, figsize=(11, 3.2))
def table(axx, T, title, c):
    M = np.array([[float(T[(x, y)]) for y in (0, 1)] for x in (0, 1)]); axx.imshow(M, cmap="Greys", vmin=0, vmax=0.6)
    for x in (0, 1):
        for y in (0, 1): axx.text(y, x, f"{T[(x, y)]}", ha="center", va="center", fontsize=11, color="white" if M[x, y] > 0.3 else "black")
    axx.set_xticks([0, 1]); axx.set_yticks([0, 1]); axx.set_xlabel("$X_2$"); axx.set_ylabel("$X_1$"); axx.set_title(title, fontsize=9.5, color=c)
    for s in axx.spines.values(): s.set_visible(False)
table(ax[0], P, "family A: both sources\n$P$ (no contrast)", GREY)
table(ax[1], pa, "family B, source 1:\n$P+Q$", ORANGE); table(ax[2], pb, "family B, source 2:\n$P-Q$", ORANGE)
table(ax[3], CT.pool([F(1, 2), F(1, 2)], [pa, pb]), "pooled, A and B alike:\n$\\frac{1}{2}(P+Q)+\\frac{1}{2}(P-Q)=P$", BLUE)
fig.suptitle("Forgetting provenance is not injective: family B has a source contrast in its pair window that pooling erases", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_forget.pdf")); fig.savefig(os.path.join(HERE, "fig_forget.png"), dpi=150); plt.close(fig)
print("ok")

# ======================================================================== Step 2
# ---- the source face and the injection direction
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
a = ax[0]; a.axis("off"); a.set_aspect("equal")
V = {"1a": np.array([0.0, 0.0]), "1b": np.array([4.0, 0.0]), "2a": np.array([1.0, 2.6]), "2b": np.array([5.0, 2.4])}
for u, v in itertools.combinations(V, 2):
    face = {u, v} in ({"1a", "1b"}, {"2a", "2b"}); P, Q = V[u], V[v]
    a.plot([P[0], Q[0]], [P[1], Q[1]], color=(BLUE if {u, v} == {"1a", "1b"} else ORANGE) if face else GREY, lw=3 if face else 0.8,
           ls="-" if face else ":", zorder=1)
for k, P in V.items():
    a.text(P[0] + (-0.25 if k.endswith("a") else 0.2), P[1] + (-0.3 if k[0] == "1" else 0.18), f"$({k[0]},{k[1]})$", fontsize=9, color=INK2)
p, Qd = 0.6, 0.25                                  # p = (0.6, 0.4), Q = (0.25, 0.75) on X = {a, b}
q = p * V["1a"] + (1 - p) * V["1b"]; tQ1 = Qd * V["1a"] + (1 - Qd) * V["1b"]; tQ2 = Qd * V["2a"] + (1 - Qd) * V["2b"]; tp2 = p * V["2a"] + (1 - p) * V["2b"]
s = 0.55
off = np.array([0.0, 0.13]); a.annotate("", q + off + s * (tQ1 - q), q + off, arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2, shrinkA=0)); a.annotate("", q + s * (tQ2 - q), q, arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2))
a.annotate("", q + s * (tp2 - q), q, arrowprops=dict(arrowstyle="-|>", color=VIOLET, lw=1.6, ls="--"))
a.plot(*q, "o", color="black", ms=6, zorder=5); a.text(q[0] - 1.15, q[1] - 0.42, "$q=\\delta_1\\otimes p$", fontsize=9.5)
a.text(2.1, -0.42, "face of source 1", color=BLUE, fontsize=8.5); a.text(2.3, 2.75, "face of source 2", color=ORANGE, fontsize=8.5)
a.text(q[0] + s * (tQ1 - q)[0] + 0.1, 0.32, "internal shift $\\delta_1\\otimes(Q-p)$", color=BLUE, fontsize=8.2, ha="left")
a.text(q[0] + s * (tQ2 - q)[0] + 0.15, q[1] + s * (tQ2 - q)[1] - 0.05, "injection under label 2\n$\\delta_2\\otimes Q-q$", color=ORANGE, fontsize=8.2)
a.text(q[0] + s * (tp2 - q)[0] - 1.35, q[1] + s * (tp2 - q)[1] + 0.1, "relabelling\n$\\delta_2\\otimes p-q$", color=VIOLET, fontsize=8.2)
a.set_xlim(-0.6, 6.6); a.set_ylim(-0.9, 3.1); a.set_title("(a) laws on $L\\times X$, $L=\\{1,2\\}$, $X=\\{a,b\\}$: the tetrahedron $\\Delta_{L\\times X}$", fontsize=9.5)
b = ax[1]; b.axis("off"); b.set_xlim(-0.3, 4.9); b.set_ylim(-1.4, 1.6)
b.plot([0, 4], [0, 0], color=INK2, lw=3); b.text(-0.2, -0.35, "$a$", fontsize=10); b.text(4.05, -0.35, "$b$", fontsize=10)
pp = (1 - p) * 4; QQ = (1 - Qd) * 4
b.plot(pp, 0, "o", color="black", ms=6); b.text(pp - 0.15, -0.4, "$p$", fontsize=10); b.plot(QQ, 0, "s", color=GREY, ms=5); b.text(QQ - 0.1, -0.4, "$Q$", fontsize=10)
b.annotate("", (pp + s * (QQ - pp), 0.35), (pp, 0.35), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2))
b.annotate("", (pp + s * (QQ - pp), 0.62), (pp, 0.62), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2))
b.text(pp + s * (QQ - pp) + 0.1, 0.62, "image of the injection", va="center", fontsize=8.2, color=ORANGE); b.text(pp + s * (QQ - pp) + 0.1, 0.35, "image of the internal shift", va="center", fontsize=8.2, color=BLUE)
b.plot(pp, 0.9, marker="o", mfc="none", mec=VIOLET, ms=8); b.text(pp + 0.15, 0.85, "relabelling $\\mapsto 0$", color=VIOLET, fontsize=8.2)
b.text(2.0, 1.3, "after forgetting the label ($D\\epsilon$)", ha="center", fontsize=9, color=INK2)
b.text(2.0, -1.05, "both arrows become $Q-p$:\nan injected source and a drifting source\nlook the same once provenance is forgotten", ha="center", fontsize=8.5, color=INK2)
b.set_title("(b) $\\Delta_X$, the pooled view", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_face.pdf")); fig.savefig(os.path.join(HERE, "fig_face.png"), dpi=150); plt.close(fig)

# ---- sensitivity of the worked example to the source share
P1 = [F(3, 10), F(1, 2)]; P2 = [F(1, 5), F(3, 10)]
fx = lambda t: (t * P1[0] + (1 - t) * P2[0]) / (t * (P1[0] + P1[1]) + (1 - t) * (P2[0] + P2[1]))
pa = lambda t: t * (P1[0] + P1[1]) / (t * (P1[0] + P1[1]) + (1 - t) * (P2[0] + P2[1]))
t0 = F(1, 4); f1 = F(-16, 529); d1 = F(640, 529)
ts = np.linspace(0, 1, 201)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.7))
a = ax[0]; a.plot(ts, [float(fx(F(t).limit_denominator(10 ** 6))) for t in ts], color=BLUE, lw=2, label="$f(\\pi_1)=P(H{=}1\\mid R{=}1)$")
a.plot(ts, [float(fx(t0) + f1 * (F(t).limit_denominator(10 ** 6) - t0)) for t in ts], color=ORANGE, lw=1.4, ls="--", label="tangent at $\\pi_1=1/4$, slope $-16/529$")
a.plot(float(t0), float(fx(t0)), "o", color="black", ms=6); a.annotate("$9/23$", (0.25, float(fx(t0))), (0.33, 0.396), fontsize=9, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
a.axhline(0.4, color=GREY, lw=0.8, ls=":"); a.axhline(0.375, color=GREY, lw=0.8, ls=":")
a.text(0.02, 0.4008, "source 2 alone: $2/5$", fontsize=8, color=INK2); a.text(0.3, 0.3757, "source 1 alone: $3/8$", fontsize=8, color=INK2)
a.set_xlabel("small-trial share $\\pi_1$"); a.set_ylabel("high-risk share among the recovered"); a.set_ylim(0.372, 0.403); a.legend(frameon=False, fontsize=8, loc="upper right")
a.set_title("(a) the answer moves slowly and stays between\nthe two sources' answers", fontsize=9.5)
b = ax[1]; b.plot(ts, [float(pa(F(t).limit_denominator(10 ** 6))) for t in ts], color=ORANGE, lw=2, label="Bayes weight $\\pi^A_1$")
b.plot(ts, ts, color=GREY, lw=1, ls=":", label="old weight $\\pi_1$")
b.plot(ts, [float(pa(t0) + d1 * (F(t).limit_denominator(10 ** 6) - t0)) for t in ts], color=VIOLET, lw=1.2, ls="--", label="tangent, slope $640/529$")
b.plot(0.25, float(pa(t0)), "o", color="black", ms=6); b.annotate("$8/23$", (0.25, float(pa(t0))), (0.34, 0.25), fontsize=9, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
b.set_ylim(0, 1); b.set_xlabel("small-trial share $\\pi_1$"); b.legend(frameon=False, fontsize=8, loc="lower right")
b.set_title("(b) the slope of (a) is the reweighting term:\n$f'=\\frac{640}{529}\\,(\\frac{3}{8}-\\frac{2}{5})=-\\frac{16}{529}$", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_sens.pdf")); fig.savefig(os.path.join(HERE, "fig_sens.png"), dpi=150); plt.close(fig)

# ---- strata and jumps
fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.6))
ts = np.linspace(-0.25, 0.25, 1001); cols = {0.1: GREY, 0.025: BLUE, 0.01: VIOLET}
a = ax[0]
for eta, c in cols.items(): a.plot(ts, 2 * eta + np.abs(ts), color=c, lw=1.8, label=f"gap $\\eta={eta:g}$")
a.plot(ts, np.abs(ts), color=ORANGE, lw=1.8, label="$\\eta=0$ (touches the wall)")
a.set_xlabel("time $t$"); a.set_ylabel("$P_t(A)$"); a.set_ylim(0, 0.5); a.legend(frameon=False, fontsize=7.8, loc="upper center")
a.set_title("(a) the event's probability changes gradually", fontsize=9.5)
b = ax[1]
for eta, c in cols.items():
    w = (eta + np.maximum(-ts, 0)) / (2 * eta + np.abs(ts)); b.plot(ts, w, color=c, lw=1.8)
    b.plot([-2 * eta, 2 * eta], [0.75, 0.25], "|", color=c, ms=8)
b.plot(ts[ts < 0], np.ones((ts < 0).sum()), color=ORANGE, lw=1.8); b.plot(ts[ts > 0], np.zeros((ts > 0).sum()), color=ORANGE, lw=1.8)
b.plot([0], [1], "o", mfc="white", mec=ORANGE, ms=6); b.plot([0], [0], "o", mfc="white", mec=ORANGE, ms=6)
b.set_xlabel("time $t$"); b.set_ylabel("posterior weight of source 1, $\\pi^A_1$"); b.set_ylim(-0.05, 1.05)
b.text(0.03, 0.8, "width $4\\eta$,\nslope $-1/(4\\eta)$", fontsize=8.2, color=INK2)
b.set_title("(b) which source explains $A$: gradual,\nthen sudden; a jump at $\\eta=0$", fontsize=9.5)
c = ax[2]; tt = np.geomspace(0.3, 2e-3, 3000)
wgt = (tt ** 2) / (tt ** 2 + 0.25); own = (1 + np.sin(1 / tt)) / 2
c.plot(tt, own, color=GREY, lw=0.7, label="source 1's own $C_A(p_1)(a)$")
c.plot(tt, wgt, color=ORANGE, lw=2, label="its posterior weight $\\pi^A_1$")
c.set_xscale("log"); c.invert_xaxis(); c.set_xlabel("$t\\to0$ (log scale)"); c.set_ylim(-0.05, 1.5); c.set_yticks([0, 0.25, 0.5, 0.75, 1]); c.legend(frameon=True, framealpha=1, edgecolor="white", fontsize=7.8, loc="upper center")
c.set_title("(c) source extinction: the weight fades\ncontinuously, the source's own conditional has no limit", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_jumps.pdf")); fig.savefig(os.path.join(HERE, "fig_jumps.png"), dpi=150); plt.close(fig)

# ======================================================================== Step 3
fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.7))
a = ax[0]; s = np.geomspace(1e-4, 0.5, 300)
# four sources on A with Lambda(l, A) = c_l s^{m_l}: m_l = 1, 1, 3/2, 2 (m = 1)
cl = [2.0, 1.0, 1.0, 3.0]; ml = [1.0, 1.0, 1.5, 2.0]; cols4 = [BLUE, AQUA, VIOLET, ORANGE]
tot = sum(c * s ** e for c, e in zip(cl, ml))
for c, e, col in zip(cl, ml, cols4):
    a.plot(s, c * s ** e / tot, color=col, lw=2, label=f"$m_l={e:g}$" + (", survives" if e == 1 else f", extinct at order {e - 1:g}"))
a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(1e-5, 1.2); a.invert_xaxis(); a.set_xlabel("$s=t-t^*\\to0$"); a.set_ylabel("posterior weight $\\pi^A_l(s)$")
a.legend(frameon=False, fontsize=7.6, loc="lower left"); a.set_title("(a) sources reaching the wall at orders $m_l$:\nthe lowest order survives (here $2/3$, $1/3$)", fontsize=9.5)
b = ax[1]; tt = np.linspace(-0.5, 0.5, 2001); eta = 0.01
for m, col in zip((1, 2, 3), (BLUE, VIOLET, ORANGE)):
    w = (eta + np.maximum(-tt, 0) ** m) / (2 * eta + np.abs(tt) ** m); b.plot(tt, w, color=col, lw=2, label=f"order $m={m}$")
    r = (2 * eta) ** (1 / m); b.plot([-r, r], [0.75, 0.25], "|", color=col, ms=9)
b.set_xlabel("time $t$"); b.set_ylabel("$\\pi^A_1$"); b.legend(frameon=False, fontsize=8); b.set_ylim(-0.03, 1.03)
b.set_title("(b) same gap $\\eta=0.01$: a higher-order approach\nswitches more gradually (width $2(2\\eta)^{1/m}$)", fontsize=9.5)
c = ax[2]; ss = np.geomspace(2e-2, 2e-5, 4000)
hardy = 1 / (1 + np.log(1 / ss)); osc = (1 + np.sin(1 / ss)) / (2 + np.sin(1 / ss))
c.plot(ss, osc, color=GREY, lw=0.6, label="$e^{-1/s}(1+\\sin\\frac{1}{s})$ against $e^{-1/s}$: no limit")
c.plot(ss, hardy, color=BLUE, lw=2, label="$e^{-1/s}/\\log\\frac{1}{s}$ against $e^{-1/s}$: limit $0$")
c.set_xscale("log"); c.invert_xaxis(); c.set_ylim(-0.03, 1.25); c.set_xlabel("$s\\to0$ (every derivative vanishes)"); c.set_ylabel("share of the first cell")
c.legend(frameon=True, framealpha=1, edgecolor="white", fontsize=7.4, loc="upper center")
c.set_title("(c) flat approaches: comparable germs\nhave a limit, oscillating ones need not", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_orders.pdf")); fig.savefig(os.path.join(HERE, "fig_orders.png"), dpi=150); plt.close(fig)

# ======================================================================== Step 4a-4b
from matplotlib.patches import Polygon as Poly
fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.6))
a = ax[0, 0]
a.add_patch(Poly([(1, 1), (2, 0), (3.6, 0), (3.6, 3.6), (1, 3.6)], closed=True, fc="#e8eefc", ec="none"))
a.plot([1, 2], [1, 0], color=INK2, lw=2); a.plot([1, 1], [1, 3.6], color=INK2, lw=1); a.plot([2, 3.6], [0, 0], color=INK2, lw=1)
a.plot(1, 1, "o", color=BLUE, ms=9); a.plot(2, 0, "o", color=ORANGE, ms=9); a.plot(2, 1, "o", color=GREY, ms=6)
a.text(0.55, 1.15, "$ts$\n(source 1)", color=BLUE, fontsize=9, ha="center"); a.text(2.05, 0.2, "$s^2$ (source 2)", color=ORANGE, fontsize=9)
a.text(2.1, 1.05, "$-ts^2$", color=GREY, fontsize=8.5)
a.annotate("", (1.5 + 0.45, 0.5 + 0.45), (1.5, 0.5), arrowprops=dict(arrowstyle="-|>", color=VIOLET, lw=1.6)); a.text(1.95, 0.75, "normal $(1,1)$", color=VIOLET, fontsize=8.5)
a.text(2.6, 2.6, "Newton polyhedron $N$", fontsize=9, color=INK2, ha="center")
a.set_xlim(0, 3.6); a.set_ylim(-0.2, 3.6); a.set_xlabel("exponent of $s$"); a.set_ylabel("exponent of $t$")
a.set_title("(a) the race: $\\Lambda(1,A)=ts$, $\\Lambda(2,A)=(1-t)s^2$", fontsize=9.5)
b = ax[0, 1]; b.set_aspect("equal")
b.fill([0, 3, 3], [0, 0, 3], color=BLUE, alpha=0.15); b.fill([0, 0, 3], [0, 3, 3], color=ORANGE, alpha=0.15); b.plot([0, 3], [0, 3], color=VIOLET, lw=2)
b.text(2.1, 0.7, "source 1 explains $A$\n$\\pi^A_1\\to1$", color=BLUE, fontsize=9, ha="center"); b.text(0.8, 1.75, "source 2 explains $A$\n$\\pi^A_1\\to0$", color=ORANGE, fontsize=9, ha="center")
b.text(2.95, 1.62, "ray $t=cs$:\n$\\pi^A_1\\to c/(1+c)$", color=VIOLET, fontsize=9, ha="right")
b.plot(2.95, 0.05, "s", color=BLUE, ms=8, clip_on=False); b.plot(0.05, 2.95, "s", color=ORANGE, ms=8, clip_on=False)
b.text(2.85, 0.15, "corner $w_s\\gg w_t$:\n$s\\to0$ first, limit 1", fontsize=8, ha="right", color=BLUE)
b.text(0.15, 2.72, "corner $w_t\\gg w_s$:\n$t\\to0$ first, limit 0", fontsize=8, ha="left", color=ORANGE)
b.set_xlim(0, 3); b.set_ylim(0, 3); b.set_xlabel("weight $w_s$ (how fast $s\\to0$)"); b.set_ylabel("weight $w_t$")
b.set_title("(b) the order fan: two cones and a ray;\niterated limits sit at the corners", fontsize=9.5)
c = ax[1, 0]; s = np.geomspace(0.3, 1e-6, 400)
for w, cc, col, lab in ((0.5, 1, BLUE, "$t=s^{1/2}$: $\\to1$"), (1, 3, AQUA, "$t=3s$: $\\to3/4$"), (1, 1, VIOLET, "$t=s$: $\\to1/2$"), (1.5, 1, ORANGE, "$t=s^{3/2}$: $\\to0$")):
    t = cc * s ** w; p1 = t * s / (t * s + (1 - t) * s ** 2); c.plot(s, p1, color=col, lw=2, label=lab)
c.set_xscale("log"); c.invert_xaxis(); c.set_ylim(-0.03, 1.03); c.set_xlabel("$s\\to0$ along each arc"); c.set_ylabel("$\\pi^A_1$"); c.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 0.08))
c.set_title("(c) the race along four arcs: the limit set\nis all of $[0,1]$", fontsize=9.5)
d = ax[1, 1]; x = np.geomspace(0.3, 1e-6, 400)
for v, col in ((1, GREY), (5, GREY)):
    y = v * x; d.plot(x, x ** 2 / (x ** 2 + y), color=col, lw=1.4, ls="--")
for cc, col in ((1, BLUE), (3, ORANGE)):
    y = cc * x ** 2; d.plot(x, x ** 2 / (x ** 2 + y), color=col, lw=2, label=f"parabola $y={cc}x^2$: $\\to1/{1 + cc}$")
d.plot([], [], color=GREY, ls="--", label="lines $y=vx$: $\\to0$"); d.set_xscale("log"); d.invert_xaxis(); d.set_ylim(-0.03, 0.7)
d.set_xlabel("$x\\to0$"); d.set_ylabel("share of the cell $x^2$ against $y$"); d.legend(frameon=False, fontsize=8, loc="upper left")
d.set_title("(d) lines are not enough: every line agrees,\nbut curved arcs do not", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_fan.pdf")); fig.savefig(os.path.join(HERE, "fig_fan.png"), dpi=150); plt.close(fig)

# ======================================================================== Step 4c
fig, ax = plt.subplots(1, 2, figsize=(10.8, 4.4))
a = ax[0]; a.set_aspect("equal")
a.fill([0, 3, 3], [0, 0, 3], color=BLUE, alpha=0.13); a.fill([0, 0, 3], [0, 3, 3], color=ORANGE, alpha=0.13)
a.plot([0, 3], [0, 3], color=VIOLET, lw=2)
a.fill([0, 3, 0], [0, 3, 3], hatch="//", fc="none", ec=ORANGE, lw=0, alpha=0.35)
a.fill([0, 3, 3], [0, 0, 3], hatch="\\\\", fc="none", ec=BLUE, lw=0, alpha=0.35)
a.text(2.2, 0.75, "audit source 1\n= the cone of design 2\n$w=(w_u+w_v,\\ w_u)$", color=BLUE, fontsize=8.8, ha="center", bbox=dict(fc="white", ec="none", alpha=0.85))
a.text(0.8, 1.75, "audit source 2\n= the cone of\ndesign 1\n$w=(w_s,\\ w_s+w_u)$", color=ORANGE, fontsize=8.8, ha="center", bbox=dict(fc="white", ec="none", alpha=0.85))
a.text(1.75, 2.9, "ray $t=cs$: audit 1 iff $c\\geq1$\n($c=1$: tie at the limit,\nbroken at order 3)", color=VIOLET, fontsize=8, ha="left", va="top", bbox=dict(fc="white", ec="none", alpha=0.85))
a.set_xlim(0, 3); a.set_ylim(0, 3); a.set_xlabel("weight $w_s$ (event rate $s\\to0$)"); a.set_ylabel("weight $w_t$ (share $t\\to0$)")
a.set_title("(a) the decision fan of the race; each study design\nfills exactly one cone (hatched)", fontsize=9.5)
b = ax[1]
ls = np.linspace(-6, 0, 400); S, T = np.meshgrid(10 ** ls, 10 ** ls)
P1 = T * S / (T * S + (1 - T) * S ** 2)
b.contourf(np.log10(S), np.log10(T), P1 > 0.5, levels=[-0.5, 0.5, 1.5], colors=[ORANGE, BLUE], alpha=0.13)
b.contour(np.log10(S), np.log10(T), P1, levels=[0.5], colors=[VIOLET], linewidths=1.6)
uu = np.geomspace(1e-6, 1, 60)
for u in (0.3, 0.03, 0.003):
    b.plot(np.log10(uu), np.log10(uu * u), color=ORANGE, lw=1.2, ls="--")
for v in (0.3, 0.03, 0.003):
    b.plot(np.log10(uu * v), np.log10(uu), color=BLUE, lw=1.2, ls="--")
b.text(-2.0, -5.3, "design 1: $t=s\\,u$\n(lines of fixed $u$)", color=ORANGE, fontsize=8.5); b.text(-5.9, -1.2, "design 2: $s=u\\,v$, $t=u$\n(lines of fixed $v$)", color=BLUE, fontsize=8.5)
b.text(-3.6, -2.5, "wall $\\pi^A_1=\\frac{1}{2}$", color=VIOLET, fontsize=9)
b.set_xlim(-6, 0); b.set_ylim(-6, 0); b.set_xlabel("$\\log_{10}s$"); b.set_ylabel("$\\log_{10}t$")
b.set_title("(b) the parameter plane: each design keeps its routes\nto the corner on one side of the decision wall", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_para.pdf")); fig.savefig(os.path.join(HERE, "fig_para.png"), dpi=150); plt.close(fig)

# ======================================================================== Step 4d
from pce import statfan as SF
from scipy.stats import norm as _norm
rng_f = np.random.default_rng(7)
fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.8))
a = ax[0, 0]; wp = np.linspace(0, 2, 200)
a.fill_between(wp, 0, np.maximum(1 - wp, 0), color=ORANGE, alpha=0.18)
a.fill_between(wp, np.maximum(1 - wp, 0), np.maximum(1 - wp / 2, 0), color=AQUA, alpha=0.18)
a.fill_between(wp, np.maximum(1 - wp / 2, 0), 2, color=BLUE, alpha=0.12)
a.plot(wp, np.maximum(1 - wp / 2, 0), color=VIOLET, lw=2.2); a.plot(wp, np.maximum(1 - wp, 0), color=INK2, lw=1.2, ls="--")
a.plot(0, 1, "o", color="black", ms=8, zorder=5); a.annotate("Theorem J: the band,\nimpact $\\to B$ SE", (0, 1), (0.25, 1.45), fontsize=8.5, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
a.text(1.25, 1.55, "invisible: exposure $\\to0$,\nimpact $\\to0$", color=BLUE, fontsize=8.5, ha="center")
a.text(0.18, 0.2, "exposed,\nimpact $\\to\\infty$", color=ORANGE, fontsize=8.5)
a.text(1.18, 0.2, "exposed,\nimpact $\\to0$", color=AQUA, fontsize=8.5)
a.text(1.32, 0.9, "exposure line\n$w_D+w_\\pi/2=1$", color=VIOLET, fontsize=8)
a.set_xlim(0, 2); a.set_ylim(0, 2); a.set_xlabel("$w_\\pi$ (adversary's share $\\pi\\sim s^{w_\\pi}$)"); a.set_ylabel("$w_D$ (its shift $D\\sim s^{w_D}$)")
a.set_title("(a) the statistical fan, $s=N^{-1/2}$: Theorem J\nis one point on the exposure line", fontsize=9.5)
b = ax[0, 1]; Ns = np.array([1e3, 3e3, 1e4, 3e4, 1e5]); cs = 2.0
for wD, col, lab in ((0.5, ORANGE, "$w_D=\\frac{1}{2}$: exposed"), (1.0, VIOLET, "$w_D=1$: the band"), (1.5, BLUE, "$w_D=\\frac{3}{2}$: invisible")):
    r = [SF.sim_J(int(N), 0.3, 1.5 * N ** (-wD / 2), 1.0, cs, 20000, rng_f)[0] for N in Ns]; b.plot(Ns, r, "o-", color=col, lw=1.8, label=lab)
b.axhline(1 - _norm.cdf(cs), color=GREY, ls=":", lw=1); b.text(1.1e3, 0.035, "false-alarm rate", fontsize=8, color=GREY)
b.set_xscale("log"); b.set_ylim(0, 1.05); b.set_xlabel("records $N$"); b.set_ylabel("exposure rate $P(Z>c_s)$"); b.legend(frameon=False, fontsize=8, loc="center right")
b.set_title("(b) Monte Carlo along three arcs\n($\\pi=0.3$, $D=1.5\\,s^{w_D}$)", fontsize=9.5)
c = ax[1, 0]; Ns2 = np.array([1e2, 1e3, 1e4, 1e5, 1e6])
for wA, col, lab in ((1, BLUE, "$w_A=1$: counts grow"), (2, ORANGE, "$w_A=2$: Poisson ray"), (3, VIOLET, "$w_A=3$: empirical wall"), (None, GREY, "fixed $P(A)=0.2$")):
    cov = [SF.sim_wald(int(N), 0.2 if wA is None else min(1.0, 3.0 * N ** (-wA / 2)), 0.3, 20000, rng_f)[0] for N in Ns2]
    c.plot(Ns2, cov, "o-" if wA else "s--", color=col, lw=1.8, label=lab)
c.axhline(0.95, color=INK2, lw=0.8, ls=":"); c.set_xscale("log"); c.set_ylim(-0.03, 1.03); c.set_xlabel("records $N$"); c.set_ylabel("coverage of the nominal 95% interval")
c.legend(frameon=False, fontsize=8, loc="center right"); c.set_title("(c) Gaussian bands are valid only where event\ncounts grow: $P(A)=3N^{-w_A/2}$", fontsize=9.5)
d = ax[1, 1]; cc = np.geomspace(1e-3, 1e3, 300)
d.plot(cc, 1 / (1 + cc), color=BLUE, lw=2, label="decoration $x\\mid y$: $\\pi^A_1=\\frac{1}{1+c}$")
d.plot(cc, (1 + cc) / (2 + cc), color=ORANGE, lw=2, label="decoration $x+y\\mid x$: $\\pi^A_1=\\frac{1+c}{2+c}$")
d.set_xscale("log"); d.set_ylim(-0.03, 1.03); d.set_xlabel("arc coefficient $c$ (along $y=c\\,x$)"); d.set_ylabel("limiting $\\pi^A_1$"); d.legend(frameon=False, fontsize=8, loc="center left")
d.set_title("(d) one Newton polyhedron, two decorations:\nlimit sets $[0,1]$ and $[\\frac{1}{2},1]$", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_stat.pdf")); fig.savefig(os.path.join(HERE, "fig_stat.png"), dpi=150); plt.close(fig)

# ======================================================================== Poisson band and Step 4e
from pce import poissonband as PBd
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw=dict(width_ratios=[1.15, 1]))
a = ax[0]; al = float(_norm.sf(2.0)); mos = np.geomspace(0.003, 3e3, 40)
bands = np.array([PBd.band(m, 0.3, al, 0.05) for m in mos])
a.plot(mos, bands[:, 1], color=VIOLET, lw=2.2, label="exact band $B_{\\mathrm{pois}}$ (pooled SEs)")
a.axhline(PBd.gauss_band(0.3, al, 0.05), color=BLUE, lw=1.4, ls="--", label="J's Gaussian band $B=0.232$")
a2 = a.twinx(); a2.plot(mos, bands[:, 0], color=ORANGE, lw=1.6, ls="-.", label="excess flags $\\Delta_{\\max}$ (counts)"); a2.set_ylabel("$\\Delta_{\\max}$ (expected flags)", color=ORANGE)
a2.set_yscale("log"); a2.tick_params(axis="y", colors=ORANGE); a2.spines["right"].set_visible(True); a2.spines["right"].set_color(ORANGE)
a.axvspan(0.003, 0.3, color=GREY, alpha=0.12); a.axvspan(0.3, 30, color=ORANGE, alpha=0.07)
tr = a.get_xaxis_transform(); a.text(0.006, 0.9, "wall side:\ncount units", fontsize=8, color=INK2, transform=tr); a.text(0.6, 0.93, "Poisson ray", fontsize=8, color=ORANGE, transform=tr); a.text(60, 0.93, "Gaussian cone", fontsize=8, color=BLUE, transform=tr)
a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("genuine expected flags $\\mu_O$"); a.set_ylabel("band in pooled standard errors")
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); a.legend(h1 + h2, l1 + l2, frameon=True, framealpha=1, edgecolor="white", fontsize=7.8, loc="center right", bbox_to_anchor=(1.0, 0.62))
a.set_title("(a) the exact band ($\\pi=0.3$, $\\alpha=0.023$, $\\beta=0.05$)", fontsize=9.5, pad=14)
b = ax[1]; b.axis("off"); b.set_xlim(0, 10); b.set_ylim(0, 7.2)
for i, (lab, col, ex) in enumerate((("$\\mathcal{M}_1$", GREY, "$(1-(q_1-q_2)^2,\\,(q_1-q_2)^2,\\,0)$"),
                                    ("$\\mathcal{M}_2$", ORANGE, "$x_1(x_1-x_2)^2$"),
                                    ("$\\mathcal{M}_3\\supseteq\\cdots\\supseteq\\mathcal{M}_\\infty$", VIOLET, ""),
                                    ("subtraction-free maps", BLUE, ""))):
    pad = 0.7 * i; b.add_patch(FancyBboxPatch((0.2 + pad, 0.2 + pad), 9.6 - 2 * pad, 6.8 - 2 * pad, boxstyle="round,pad=0.02,rounding_size=0.25", fc="white" if i else "#fafafa", ec=col, lw=1.6))
    b.text(0.45 + pad, 6.88 - pad, lab, color=col, fontsize=9, weight="bold", va="top")
    if ex: b.text(9.55 - pad, 6.88 - pad, "leaves next: " + ex, color=col, fontsize=7.6, va="top", ha="right")
b.text(5, 3.45, "stochastic matrices, $\\lambda$, $\\epsilon$, $M_S$,\npooling $\\mu_n$, products $x\\otimes y$,\nconditioning $C_A$, $\\mathfrak{C}_A$, Bayes weights $\\pi^A$,\nBayesian inverse $K^\\dagger$",
       ha="center", va="center", fontsize=8.6, color=BLUE)
b.set_title("(b) the graded classes: every layer operation is\nsubtraction-free, hence in $\\mathcal{M}_\\infty$", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_graded.pdf")); fig.savefig(os.path.join(HERE, "fig_graded.png"), dpi=150); plt.close(fig)

# ======================================================================== Step 4f
fig = plt.figure(figsize=(11.5, 3.9)); gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1])
a = fig.add_subplot(gs[0]); a.axis("off"); a.set_xlim(0, 10); a.set_ylim(0, 10)
boxes = [(5, 9.0, "level 0: cells in $(x,y)$\nedge $w=1$ of $P(A)$, root $c^*=1$ (double)", ORANGE),
         (5, 6.4, "level 1: $y=x(1+y_1)$, $x=t$\nedge $w=1$, root $c^*=1$ (double)", ORANGE),
         (5, 3.8, "level 2: $y_1=t(1+y_2)$\nnon-degenerate: cones and a ray $w=\\frac{1}{2}$", BLUE)]
for x0, y0, tx, col in boxes:
    a.add_patch(FancyBboxPatch((x0 - 3.9, y0 - 0.9), 7.8, 1.8, boxstyle="round,pad=0.02,rounding_size=0.2", fc="white", ec=col, lw=1.6)); a.text(x0, y0, tx, ha="center", va="center", fontsize=8.3)
for y0 in (8.1, 5.5):
    a.annotate("", (5, y0 - 0.8), (5, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2))
a.text(5, 1.0, "limits: $1$ (leave early), $\\frac{b^2}{1+b^2}$ on $y=x+x^2+bx^{5/2}$,\n$0$ (leave late or follow $y=x+x^2$ exactly)", ha="center", fontsize=8.3, color=VIOLET)
a.set_title("(a) the refinement tree of $(y-x-x^2)^2$ against $x^5$", fontsize=9.5)
b = fig.add_subplot(gs[1]); bb = np.linspace(-4, 4, 400)
b.plot(bb, bb ** 2 / (1 + bb ** 2), color=VIOLET, lw=2)
b.set_xlabel("coefficient $b$ of $x^{5/2}$"); b.set_ylabel("limiting share of source 1"); b.set_ylim(-0.03, 1.03)
b.set_title("(b) the limit map at the bottom node", fontsize=9.5)
c = fig.add_subplot(gs[2]); cc = np.linspace(-0.6, 0.4, 400)
for K, ae, col in ((8, -1 / 8, ORANGE), (12, -5 / 128, BLUE)):
    c.plot(cc, 4 * (cc - ae) ** 2 / (4 * (cc - ae) ** 2 + 1), color=col, lw=2, label=f"$K={K}$: zero at $c={'-1/8' if K == 8 else '-5/128'}$")
c.set_xlabel("coefficient $c$ at the cut exponent $(K-2)/2$"); c.set_ylabel("limiting share of source 1"); c.set_ylim(-0.02, 0.6); c.legend(frameon=False, fontsize=8, loc="upper center")
c.set_title("(c) $(y^2-x^2-x^3)^2$ against $x^K$: the root's\ncoefficient is the zero of the limit map", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_degen.pdf")); fig.savefig(os.path.join(HERE, "fig_degen.png"), dpi=150); plt.close(fig)

# ======================================================================== Step 5
fig, ax = plt.subplots(1, 2, figsize=(10.8, 4.0))
a = ax[0]; ls_ = np.linspace(-1, -40, 400); s_ = np.exp(ls_)
for (c1, e1, c2, e2), col, lab in (((1.0, 1.0, 3.0, 2.0), BLUE, "pooling $s+3s^2$: $\\to\\min(1,2)=1$"), ((5.0, 1.5, 2.0, 2.5), ORANGE, "pooling $5s^{3/2}+2s^{5/2}$: $\\to\\frac{3}{2}$")):
    lead = np.log(max(c1, c2) * 0 + c1) / ls_ + e1
    a.plot(-ls_, np.log(c1 * s_ ** e1 + c2 * s_ ** e2) / ls_, color=col, lw=2, label=lab); a.axhline(min(e1, e2), color=col, lw=0.8, ls=":")
    a.fill_between(-ls_, lead - np.log(2) / np.abs(ls_), lead + np.log(2) / np.abs(ls_), color=col, alpha=0.12)
a.set_xlabel("$|\\log s|$"); a.set_ylabel("$\\mathrm{Log}_s=\\log(\\cdot)/\\log s$"); a.set_ylim(0.4, 2.2); a.legend(frameon=False, fontsize=7.8, loc="upper right")
a.set_title("(a) under the log map pooling becomes min: the curve stays\nwithin $\\log 2/|\\log s|$ of its larger term (shaded)", fontsize=9.5)
b = ax[1]; b.axis("off"); b.set_aspect("equal")
V = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]]); b.add_patch(Polygon(V, closed=True, fc="#f3f3f3", ec=INK2, lw=1.2))
x0 = 0.55 * V[0] + 0.45 * V[1]; b.plot(*x0, "o", color="black", ms=6, zorder=5); b.text(x0[0] - 0.33, x0[1] - 0.08, "$x$ ($x_c=0$)", fontsize=8.5)
th = np.linspace(0, np.pi, 60); b.fill(np.r_[x0[0] + 0.28 * np.cos(th)], np.r_[x0[1] + 0.28 * np.sin(th)], color=BLUE, alpha=0.18)
b.text(0.86, 0.36, "$C_x$: every direction\ninto the triangle\n(a half-plane)", fontsize=8, color=BLUE)
b.annotate("", x0 + np.array([-0.05, 0.42]), x0, arrowprops=dict(arrowstyle="-|>", color=VIOLET, lw=1.8)); b.text(-0.15, 0.5, "$g\\in C_x^*$:\nraises $c$\n($g_c\\geq\\gamma$)", fontsize=8, color=VIOLET)
b.annotate("", x0 + np.array([0, -0.3]), x0, arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.6)); b.text(x0[0] + 0.04, x0[1] - 0.29, "normal cone $N_x=-C_x^*$", fontsize=8, color=ORANGE)
b.text(0.5, 0.92, "$c$", fontsize=10, ha="center"); b.set_xlim(-0.2, 1.3); b.set_ylim(-0.4, 1.0)
b.set_title("(b) the cotangent cone at a boundary law: a tied act is\ninjection-proof iff its utility gap lies in $C_x^*$", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_duality.pdf")); fig.savefig(os.path.join(HERE, "fig_duality.png"), dpi=150); plt.close(fig)
