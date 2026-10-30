"""Figure for Gap 3 (the decision and the ring of contingency tables). (a) the worked truth: the margin fibre (tables with its pair
tables), the hiding set R inside it, and the lattice of tables at a small n; (b) the worst case as a tropical limit: the weighted count
-(1/s) log sum exp(-s U) falls to the lattice minimum as s grows; (c) over the margin fibre the worst case is piecewise linear in the
margins, with kinks exactly at the chamber walls (where the binding cell changes); (d) Study 11c: settling with the exact conditional
probe and with the plug-in probe, against n / n*_c."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ENG = os.path.join(HERE, "..", "engine"); sys.path.insert(0, ENG)
from pce import ehrhart as EH, lab as L
D = json.load(open(os.path.join(HERE, "ehrhart.json"))); S = json.load(open(os.path.join(ENG, "study11c_exact", "results_study11c.json")))
BLUE, ORANGE, VIOLET, AQUA, GREY, CRIT = "#2a78d6", "#eb6834", "#6250d6", "#1baf7a", "#8c8c8c", "#d03b3b"; INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 4, figsize=(18, 4.1), gridspec_kw=dict(width_ratios=[1.25, 1, 1, 1.1]))

# (a)
a = ax[0]; t = D["truth"]; F, R, m0 = t["fibre_mT"], t["R_mT"], t["m0"]
a.plot(F, [0, 0], color=INK2, lw=2, solid_capstyle="butt"); a.fill_between(R, -0.25, 0.25, color=BLUE, alpha=0.18, lw=0)
a.plot(R, [0, 0], color=BLUE, lw=5, solid_capstyle="butt")
n = 150; js = np.arange(np.ceil(n * (F[0] - m0)), np.floor(n * (F[1] - m0)) + 1); pts = m0 + js / n
inR = (pts >= R[0]) & (pts <= R[1])
a.scatter(pts[~inR], np.zeros((~inR).sum()), s=14, color="white", edgecolor=INK2, zorder=3, lw=0.8)
a.scatter(pts[inR], np.zeros(inR.sum()), s=14, color=BLUE, edgecolor=BLUE, zorder=3)
a.scatter([m0], [0], marker="*", s=120, color=ORANGE, zorder=4, edgecolor="white", lw=0.5)
a.annotate("truth", (m0, 0), (m0 + 0.012, -0.45), ha="center", fontsize=8, color=ORANGE, arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))
a.annotate("a cell reaches 0\n(lattice end)", (F[0], 0), (F[0], -0.7), ha="left", fontsize=7.5, color=INK2)
a.annotate("a cell reaches 0", (F[1], 0), (F[1], -0.7), ha="right", fontsize=7.5, color=INK2)
a.annotate("hiding set $R$: ends where\nsome $a_{ji}$ reaches 0 or 1", ((R[0] + R[1]) / 2, 0.25), ((R[0] + R[1]) / 2, 0.95), ha="center",
           fontsize=7.5, color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
a.set_ylim(-1.1, 1.5); a.set_yticks([]); a.spines["left"].set_visible(False); a.set_xlabel("triple moment $m_T$ (pair tables fixed)")
a.set_title(f"(a) tables with the worked truth's pair tables, n = {n}:\nthe fibre is a lattice segment (step 1/n); $R$ is {(R[1]-R[0])/(F[1]-F[0]):.0%} of it,"
            " with irrational ends", fontsize=9)

# (b)
a = ax[1]; tr = D["tropical"]; key = [k for k in tr if k.startswith("A any harm")][0]; nn = int(key.split("n=")[1])
mu = None
P = np.zeros((2, 2, 2))
from fractions import Fraction as Fr
w, a1, a2 = 0.4, [0.8, 0.3, 0.7], [0.2, 0.7, 0.15]; P = L.law(w, a1, a2)
u = lambda x: 1.0 - float(sum(x) > 0); c = EH.walsh(u); U0 = float(sum(P[x] * u(x) for x in EH.X))
jR = np.arange(np.ceil(nn * (R[0] - m0)), np.floor(nn * (R[1] - m0)) + 1); vals = U0 + c * jR / nn
ss = np.logspace(1, 8, 300)
gap = np.array([vals.min() - EH.tropical(vals, s) for s in ss]); keep = np.abs(gap) > 1e-13; ss, gap = ss[keep], gap[keep]
a.loglog(ss, np.maximum(-gap, 1e-17) * 0 + np.abs(gap), color=BLUE, lw=2, label=r"$\min_T U_a-\left(-\frac{1}{s}\log\sum_T e^{-sU_a(T)}\right)$")
a.loglog(ss, np.log(len(vals)) / ss, color=ORANGE, lw=1.2, ls="--", label=r"bound $\log\#/s$")
a.set_xlabel("weight $s$"); a.set_ylabel("distance to the lattice worst case"); a.legend(frameon=False, fontsize=7.5, loc="lower left")
a.set_title(f"(b) not a section but a tropical limit: the weighted\ncount over {len(vals)} tables in $nR$ converges to the worst case (n = {nn})", fontsize=9)

# (c)
a = ax[2]; rng = np.random.default_rng(3)
for _ in range(50):
    Dd = rng.normal(size=(2, 2, 2)); Dd -= Dd.mean(); ts = np.linspace(-1, 1, 401) * 0.6 * P.min() / np.abs(Dd).max()
    res = [EH.worst_lp(P + t * Dd, u) for t in ts]; bind = [tuple(z) for _, z in res]
    if len(set(bind)) >= 2: break
Wv = np.array([r[0] for r in res]); length = np.array([np.subtract(*EH.eps_range(P + t * Dd)[::-1]) for t in ts])
a.plot(ts, Wv, color=BLUE, lw=2, label="worst case of act A over the fibre")
a2_ = a.twinx(); a2_.plot(ts, length, color=VIOLET, lw=1.3, ls="--", label="fibre length (leading term of the count)")
a2_.spines["top"].set_visible(False); a2_.set_ylabel("fibre length", color=VIOLET); a2_.tick_params(axis="y", colors=VIOLET)
for i in range(1, len(ts)):
    if bind[i] != bind[i - 1]: a.axvline(0.5 * (ts[i] + ts[i - 1]), color=CRIT, lw=0.8, ls=":")
a.set_xlabel("position $t$ along a line of pair tables"); a.set_ylabel("worst-case utility")
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2_.get_legend_handles_labels(); a.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.5, loc="upper center")
a.set_title("(c) over the whole fibre the worst case is an integer\nprogramme: linear between chamber walls (dotted)", fontsize=9)

# (d)
a = ax[3]; cols = {"D2": BLUE, "D3": VIOLET, "D5": AQUA}
for d, col in cols.items():
    cu = S[d]["curve"]; ns = np.array(sorted(int(k) for k in cu)); x = ns / S[d]["nstar_c"]
    a.plot(x, [cu[str(k)]["pred_c"] for k in ns], color=col, lw=1.2)
    a.scatter(x, [cu[str(k)]["exact"] for k in ns], color=col, s=22, zorder=3, label=f"{d} exact")
    a.scatter(x, [cu[str(k)]["delta"] for k in ns], facecolor="white", edgecolor=col, s=22, zorder=3, label=f"{d} plug-in")
cu = S["D4"]["curve"]; ns = np.array(sorted(int(k) for k in cu))
a.axhline(0.5, color=GREY, lw=0.6); a.set_xscale("log"); a.set_ylim(0, 1.02)
a.set_xticks([0.5, 1, 2, 4, 8]); a.set_xticklabels(["0.5", "1", "2", "4", "8"]); a.minorticks_off()
a.set_xlabel("$n / n^*_c$"); a.set_ylabel("share of replicates settled"); a.legend(frameon=False, fontsize=7, ncol=2, loc="lower right")
a.set_title("(d) Study 11c: the exact conditional probe settles\nas predicted (lines: normal theory with $\\tau_c$)", fontsize=9)

fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_ehrhart.pdf")); fig.savefig(os.path.join(HERE, "fig_ehrhart.png"), dpi=150)
print("ok")
