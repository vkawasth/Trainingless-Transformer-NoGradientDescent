"""Figure for the sections on hiding sheets and global pooling of "A tangent structure for injected data". (a) the C4 hiding sheet with
the worst case of an event and its tangent cone; (b) one chord: the domain polynomial q and the decision rhat - r_c, with a hole and
the genuine crossings; (c) the global picture: same-act pairs that are safe, flip inside the domain, or leave it, by distance."""
import json, os, sys, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
HERE = os.path.dirname(os.path.abspath(__file__)); ENG = os.path.join(HERE, "..", "engine"); sys.path.insert(0, ENG)
import warnings; warnings.filterwarnings("ignore")
from pce import covers as CV, restriction as RS, lifting as LF, injection as IJ
from scipy.spatial import HalfspaceIntersection, ConvexHull
D = json.load(open(os.path.join(ENG, "..", "lab", "restriction.json")))
BLUE, ORANGE, VIOLET, AQUA, GREY, CRIT = "#2a78d6", "#eb6834", "#6250d6", "#1baf7a", "#8c8c8c", "#d03b3b"; INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.0))
a = ax[0]; m, E = 4, [(0, 1), (1, 2), (2, 3), (0, 3)]; w, a1, a2 = 0.4, np.array([.8, .3, .7, .25]), np.array([.2, .7, .15, .7])
mu, cc, v = CV.truth_data(w, a1, a2, E); F = CV.factorise(m, E, cc); sg = [float(np.sign(v[0] / F["comps"][0]["v0"][0]))]; A, b, who = CV.sheet_constraints(mu, F, sg)
z0, _ = CV.chebyshev(A, b); hs = HalfspaceIntersection(np.c_[A, -b], z0); V = hs.intersections[ConvexHull(hs.intersections).vertices]
X4 = list(itertools.product((0, 1), repeat=4)); ev = np.array([1.0 if x[0] == x[1] == x[2] == 1 else 0.0 for x in X4])
Fz = lambda z: ev @ CV.law_m(*CV.explain(mu, F, sg, z[0], z[1:], m=4)[:3]).ravel()
ys = np.linspace(V[:, 0].min(), V[:, 0].max(), 120); xs = np.linspace(V[:, 1].min(), V[:, 1].max(), 120)
Z = np.array([[Fz(np.array([y, x])) if np.all(A @ np.array([y, x]) <= b + 1e-12) else np.nan for y in ys] for x in xs])
cs = a.contourf(ys, xs, Z, levels=20, cmap="Blues"); a.add_patch(Polygon(V, closed=True, fill=False, ec=INK2, lw=1.2))
zs = np.array(D["sheet_min"]["z"]); a.scatter(*zs, color=CRIT, zorder=5, s=25)
act = [k for k in range(len(b)) if abs(A[k] @ zs - b[k]) < 1e-7]
for k in act:
    t = np.array([-A[k][1], A[k][0]]); t /= np.linalg.norm(t)
    for sgn in (1, -1):
        d = sgn * t
        if all(A[j] @ d <= 1e-9 for j in act): a.annotate("", zs + 0.25 * d, zs, arrowprops=dict(arrowstyle="-|>", color=CRIT, lw=1.4))
a.text(zs[0] + 0.12, zs[1] + 0.12, f"worst case {D['sheet_min']['value']:.4f}:\nDF $\\cdot$ u $\\geq$ 0 on the cone", fontsize=7.5, color=CRIT)
a.set_xlabel("$y=\\log t$"); a.set_ylabel("$x=\\log\\lambda$"); a.set_aspect("equal")
a.set_title("(a) the tangent structure on a hiding sheet:\nthe worst case sits at a vertex, the cone decides", fontsize=9)
a = ax[1]; W = LF.we(); P0 = IJ.as_vec(W["P"]); c = list(W["ps"][0] - np.r_[W["UP"], 0.0]); r_c = -c[0] / c[1]
act_ = lambda P: (None if not RS.rhat_map.dom(P) else (1 if RS.rhat_map.fn(P)[0] > r_c else -1))
rng = np.random.default_rng(11)
while True:
    P = np.clip(P0 + rng.normal(scale=0.06, size=8), 1e-4, None); P /= P.sum(); Q = np.clip(P0 + rng.normal(scale=0.06, size=8), 1e-4, None); Q /= Q.sum()
    if act_(P) is not None and act_(P) == act_(Q):
        ex = RS.exact_pooling(P, Q, r_c, act_)
        if ex["q_roots"] and ex["genuine"]: break
es = np.linspace(0, 1, 2001); qv = np.array([RS._q((1 - e) * P + e * Q) for e in es]); rv = np.array([RS.rhat_map((1 - e) * P + e * Q)[0] - r_c if RS.rhat_map.dom((1 - e) * P + e * Q) else np.nan for e in es])
a2 = a.twinx(); a.plot(es, np.sign(qv) * np.abs(qv) ** (1 / 3), color=GREY, lw=1.2, label="$\\mathrm{sgn}(q)|q|^{1/3}$ (domain: $q>0$)"); a.axhline(0, color=INK2, lw=0.6)
a2.plot(es, rv, color=VIOLET, lw=2, label="$\\hat r-r_c$ (act T where $>0$)"); a2.axhline(0, color=VIOLET, lw=0.6, ls=":")
for x in ex["q_roots"]: a.axvline(x, color=CRIT, lw=0.8, ls="--")
for x in ex["genuine"]: a2.scatter([x], [0], color=ORANGE, zorder=5, s=25)
a.axvspan(*ex["q_roots"][:2], color=CRIT, alpha=0.12) if len(ex["q_roots"]) >= 2 else None
a.set_xlabel("share $\\varepsilon$ of $Q$"); a.set_ylabel("domain polynomial (cube root)"); a2.set_ylabel("decision $\\hat r-r_c$", color=VIOLET)
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); a.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc="lower left")
a.set_title("(b) one chord, decided exactly: a hole (red) and\ngenuine act changes (orange) as roots of degree-6 polynomials", fontsize=9)
a = ax[2]; st = D["pooling_stats"]; ks = list(st.keys()); cats = ["safe", "flip inside the domain", "leaves the domain"]; cols = [BLUE, ORANGE, CRIT]
left = np.zeros(len(ks))
for cat, col in zip(cats, cols):
    vals = np.array([st[k][cat] for k in ks]) / np.array([sum(st[k].values()) for k in ks]); a.barh(range(len(ks)), vals, left=left, color=col, label=cat); left += vals
a.set_yticks(range(len(ks))); a.set_yticklabels(ks, fontsize=7.5); a.set_xlabel("share of same-act pairs (150 each)"); a.legend(frameon=False, fontsize=7.2, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
a.set_title("(c) globally, pooling fails by leaving the domain\n(the pooled covariances take the odd sign class), not by flipping", fontsize=9)
plt.tight_layout(); fig.savefig(os.path.join(HERE, "fig_global.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_global.png"), dpi=150, bbox_inches="tight"); print("ok", ex)
