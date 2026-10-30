"""Figures for the companion paper "A lab for latent actors". Reads engine/study11_lab/results_study11.json. Descriptive only.

fig_ooda      the OODA loop drawn on the geometry (all readers)
fig_segment   Theorem A: event probabilities along the hiding segment; the rare-severe / common-mild trade-off
fig_tempo     Theorem B: learning curves; perturbation exponents; the null distribution
fig_voi       Theorem C: exact VOI per probe; graded VOI against |v_S|; probe choice; negative VOI at level one
fig_game      persuasion and disclosure: errors by strategy and policy; unravelling; hiding widths; catch-up
fig_decide    Theorem D: the decision on the segment (D2), settling curves (Study 11b), the stratum plateau (D4)
fig_inject    Part II: the hidden channel (two segments), Study 12 tilt and detection, the stealth-tilt frontier, tetrad-test power
fig_oneline   Theorem H and G(4): the one line (model part inside positivity part) and the obstruction (tetrads vs eps)
fig_worked    the worked example: cells and the Walsh character, the decision on the segment, the one line in harm units, the Stanley-Reisner complex
fig_toric     the worked example, Step 13: the moment segment of CP^1, the sphere with its circles, and the Birch and Cayley polynomials on the line
fig_worked4   the four-variable example: cells and facets, three-way interactions, the fibre projected with the model line, the second-triple defence
fig_sources   Part II, sources (Study 13): bands of the apparent triple harm, settling curves, source-statistic paths, tilt rates
fig_lifting   lifting: the adversarial block, dilation, the value of a probe in its size, backtracking
fig_capstone  Theorem L: the pair line with its chambers, model part and walls; one margin over five capacities
fig_descent   the worked descent complex: the nerve with cochain dimensions, the differential d0, the latent graph with its sign
              cochain, the labelled cochain of one Study 13 replicate
fig_filtration the descent filtration D0 > ... > D5 with the worked example and one witness per stage; the value each stage computes
fig_markov    Markov categories: the two-type string diagram, de Finetti for batches on the line, Blackwell values, Fisher information
fig_ml        ML/RL/GAN: a bandit learning which triple to probe; inspection-game dynamics with and without commitment
"""
import itertools, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
HERE = os.path.dirname(os.path.abspath(__file__)); ENG = os.path.join(HERE, "..", "engine"); sys.path.insert(0, ENG); sys.path.insert(0, os.path.join(ENG, "study11_lab"))
from pce.lab import law, moments, fit_v, segment, explanation, r_of_t, event_prob
import run_study11 as S11
R = json.load(open(os.path.join(ENG, "study11_lab", "results_study11.json")))
BLUE, ORANGE, VIOLET, AQUA, GREY, CRIT = "#2a78d6", "#eb6834", "#6250d6", "#1baf7a", "#8c8c8c", "#d03b3b"; INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
COL = {"G": BLUE, "A": ORANGE, "D": AQUA, "M": VIOLET}
def save(fig, name): fig.savefig(os.path.join(HERE, name + ".pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, name + ".png"), dpi=150, bbox_inches="tight"); plt.close(fig)

# ---------------- OODA
fig, ax = plt.subplots(figsize=(8.4, 4.4)); ax.set_xlim(0, 8.4); ax.set_ylim(0, 4.4); ax.axis("off")
B = [(0.2, 2.45, "Observe", "windows and tables\n(the sheaf revealed so far)", GREY), (4.4, 2.45, "Orient", "the latent fibre: a segment under\npair tables; strata $v_i=0$", VIOLET),
     (4.4, 0.25, "Decide", "robust act, and which probe:\n$\\arg\\max_S|\\hat v_S|$ (Thm C)", BLUE), (0.2, 0.25, "Act", "run the treatment or the probe;\nits outcome is a new window", ORANGE)]
for x, y, t, s, c in B:
    ax.add_patch(FancyBboxPatch((x, y), 3.8, 1.65, boxstyle="round,pad=0.02,rounding_size=0.1", fc="white", ec=c, lw=2))
    ax.text(x + 1.9, y + 1.28, t, ha="center", fontsize=12, weight="bold", color=c); ax.text(x + 1.9, y + 0.55, s, ha="center", va="center", fontsize=9, color=INK)
ar = lambda p, q, c=INK: ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, color=c, lw=1.4))
ar((4.0, 3.27), (4.4, 3.27)); ar((6.3, 2.45), (6.3, 1.9)); ar((4.4, 1.07), (4.0, 1.07)); ar((2.1, 1.9), (2.1, 2.45))
ax.text(4.2, 2.17, "tempo: $0$ (pairs only), $n^{-1/2}$ (after a triple),\n$n^{-1/4}$ for the merge (Thm B)", ha="center", va="center", fontsize=8.6, color=CRIT)
save(fig, "fig_ooda")

# ---------------- Theorem A
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
for c in ("G", "A", "D"):
    cv = R["11a"]["curves"][c]; ax[0].plot(cv["r"], cv["pE"], color=COL[c], lw=2, label={"G": "G generic", "A": "A agreement on $x_3$", "D": "D dominance"}[c])
    ax[0].scatter([cv["r_true"]], [cv["pE_true"]], color=COL[c], edgecolor=INK, zorder=5, s=30)
    ax[1].plot(cv["r"], cv["p4"], color=COL[c], lw=2)
ax[0].set_xlabel("$r=(w_2-w_1)/\\sqrt{w_1w_2}$ along the hiding segment"); ax[0].set_ylabel("P(any harm among $x_0,x_1,x_2$)"); ax[0].legend(frameon=False, fontsize=7.5)
ax[0].set_title("(a) a 3-variable event is affine in $r$; worst cases at the\nbox walls (dots: truth; G lies under A: they differ only in $x_3$)", fontsize=9)
ax[1].set_xlabel("$r$"); ax[1].set_ylabel("P(all four harmed)"); ax[1].set_title("(b) a 4-variable event is quadratic in $r$", fontsize=9)
w, a1, a2 = S11.COND["G"]; P = law(w, a1, a2); mu, M, _ = moments(P); v = fit_v(mu, M); vt = np.sqrt(w * (1 - w)) * (np.array(a1) - np.array(a2)); v = v if np.dot(v, vt) >= 0 else -v
lo, hi = segment(mu, v); ts = np.exp(np.linspace(np.log(lo), np.log(hi), 200)); ws = 1 / (1 + ts ** 2); sev = [np.max(np.abs(explanation(mu, v, t)[1] - explanation(mu, v, t)[2])) for t in ts]
ax[2].plot(ws, sev, color=BLUE, lw=2); ax[2].scatter([w], [np.max(np.abs(np.array(a1) - np.array(a2)))], color=BLUE, edgecolor=INK, zorder=5)
ax[2].set_xlabel("weight of actor 1 (subgroup size)"); ax[2].set_ylabel("largest difference between actors"); ax[2].set_title("(c) identical pair tables: a rare, very different\nsubgroup or a common, mildly different one", fontsize=9)
plt.tight_layout(); save(fig, "fig_segment")

# ---------------- Theorem B
lc = R["11c"]["curves"]; ns = np.array(lc["n"])
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
for key, col, lab in (("rmse_r_G", BLUE, "$r$ after one triple (G)"), ("width_G", GREY, "width of the segment, pairs only (G)"), ("rmse_v3_A", ORANGE, "$v_3$ at agreement (A)"), ("norm_v_M", VIOLET, "$\\|v\\|$ at the merge (M)")):
    y = np.array(lc[key]); ax[0].loglog(ns, y, "o-", color=col, lw=1.8, ms=4, label=lab + (f"  slope {R['11c']['slopes'][key]:.2f}" if key in R["11c"]["slopes"] else "  slope 0"))
for sl, y0 in ((-0.5, 0.6), (-0.25, 0.25)):
    ax[0].loglog(ns, y0 * (ns / ns[0]) ** sl, ":", color=INK2, lw=0.8)
ax[0].set_xlabel("samples per window $n$"); ax[0].set_ylabel("error / width"); ax[0].legend(frameon=False, fontsize=7); ax[0].set_title("(a) three tempos: $0$, $n^{-1/2}$, $n^{-1/4}$ (dotted guides)", fontsize=9)
sv = np.logspace(-6, -3, 7)
for c in ("G", "D", "A", "M"):
    rr = R["11b"]["relations"][c]; ax[1].loglog(sv, rr["responses"], "o-", color=COL[c], ms=4, label=f"{c}: exponent {rr['exponent']:.2f}")
ax[1].set_xlabel("perturbation of the pair moments $s$"); ax[1].set_ylabel("response of the fitted actors"); ax[1].legend(frameon=False, fontsize=7.5)
ax[1].set_title("(b) the merge answers like $\\sqrt{s}$; agreement on one\nvariable is regular (exponent 1)", fontsize=9)
ne = np.array(R["11b"]["null_exponents"]); ax[2].hist(ne, bins=25, color=GREY, alpha=0.8, label="200 generic truths (null)")
for c in ("G", "A", "D", "M"): ax[2].axvline(R["11b"]["relations"][c]["exponent"], color=COL[c], lw=2, label=c)
ax[2].axvline(0.75, color=CRIT, ls="--", lw=1, label="detection rule 0.75"); ax[2].set_xlabel("response exponent"); ax[2].legend(frameon=False, fontsize=7)
ax[2].set_title("(c) calibrated null and the detection rule", fontsize=9)
plt.tight_layout(); save(fig, "fig_tempo")

# ---------------- Theorem C
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9)); tri = ["".join(map(str, S)) for S in S11.TRIPLES]
xx = np.arange(4); cost = 1.235                      # post hoc: a cost at which the best act changes along G's segment; Savage regret
def voi_cost(c_):
    P, mu, M, mS, v, vt = S11.exact(c_)
    if c_ == "M": return [0.0] * 4
    ts_, ps = S11.seg_curve(mu, v, S11.E3); f = 1 - cost * ps                       # E[u_treat] along the segment; u_nothing = 0
    regret_before = min(np.max(np.maximum(0, -f)), np.max(np.maximum(0, f)))        # minimax regret over {treat, nothing}
    return [0.0 if abs(np.prod([v[i] for i in S])) < 1e-12 else float(regret_before) for S in S11.TRIPLES]
for k, c in enumerate(("G", "A", "M")): ax[0].bar(xx + (k - 1) * 0.27, voi_cost(c), width=0.25, color=COL[c], label=c)
ax[0].set_xticks(xx); ax[0].set_xticklabels([f"triple {t}" for t in tri]); ax[0].set_ylabel("minimax-regret VOI (exact)"); ax[0].legend(frameon=False)
ax[0].set_title("(a) Savage minimax-regret VOI (post hoc, cost 1.235):\nprobes touching the coincident variable (3, in A) are worthless", fontsize=9)
rng = np.random.default_rng(99); xs, ys, cs = [], [], []
for _ in range(150):
    w, a1, a2 = S11.random_truth(rng); P = law(w, a1, a2); mu, M, _ = moments(P); v = fit_v(mu, M); vt = np.sqrt(w * (1 - w)) * (a1 - a2); v = v if np.dot(v, vt) >= 0 else -v
    ts_, ps = S11.seg_curve(mu, v, S11.E3); rs = r_of_t(ts_); before = max(0.0, S11.u_treat(ps.max())); rtrue = r_of_t(np.sqrt((1 - w) / w))
    if max(0.0, S11.u_treat(ps.min())) - before < 1e-9: continue
    for S in S11.TRIPLES:
        vS = abs(np.prod(v[list(S)])); ex2 = sum(P[x] * np.prod([(x[i] - mu[i]) for i in S]) ** 2 for x in itertools.product((0, 1), repeat=4))
        se = np.sqrt(ex2 / 1000) / vS; ins = (rs >= rtrue - 2 * se) & (rs <= rtrue + 2 * se)
        after = max(0.0, S11.u_treat(ps[ins].max())) if ins.any() else max(0.0, S11.u_treat(np.interp(rtrue, rs, ps)))
        xs.append(vS); ys.append(after - before)
ax[1].scatter(xs, ys, s=10, color=BLUE, alpha=0.5); ax[1].set_xscale("log"); ax[1].set_xticks([0.5, 1, 2, 4, 8]); ax[1].set_xticklabels(["1/2", "1", "2", "4", "8"]); ax[1].minorticks_off(); ax[1].set_xlabel("$|v_S|$ of the probe"); ax[1].set_ylabel("realised VOI ($n=1000$)")
ax[1].set_title(f"(b) graded VOI rises with $|v_S|$ (registered within-truth\nSpearman {R['11c']['L7_mean_within_truth_spearman']:.3f}, target 0.8: failed narrowly)", fontsize=9)
ax[2].plot(R["11c"]["L8_pi"], R["11c"]["L8_h1"], "o-", color=ORANGE, ms=4, label="after adding the BC window")
ax[2].axhline(R["11c"]["L8_h0"], color=BLUE, lw=1.5, label="before (AB, AC)"); ax[2].set_xlabel("type-1 share $\\pi$ in the BC trial's population")
ax[2].set_ylabel("robustness of the best act"); ax[2].legend(frameon=False, fontsize=7.5); ax[2].set_title("(c) level one: adding a conflicting window never lowers\nrobustness (it can only jump to vacuity, Prop. 1)", fontsize=9)
plt.tight_layout(); save(fig, "fig_voi")

fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.semilogx(ns, lc["agree_G"], "o-", color=BLUE, label="G: rule = oracle's best triple"); ax.semilogx(ns, 1 - np.array(lc["pick3_A"]), "s-", color=ORANGE, label="A: rule avoids variable 3")
ax.set_ylim(0, 1.03); ax.set_xlabel("samples per pair window $n$"); ax.set_ylabel("share of replications"); ax.legend(frameon=False, fontsize=7.5)
ax.set_title("choosing the probe from pair tables: $\\arg\\max_S|\\hat v_S|$", fontsize=9); plt.tight_layout(); save(fig, "fig_probe")

# ---------------- disclosure game
G = R["11d"]["game"]; st = "FPQS"; po = "NKMR"
fig, ax = plt.subplots(1, 4, figsize=(15.5, 3.8), gridspec_kw={"width_ratios": [1, 1, 1.1, 1.1]})
for k, (key, tit) in enumerate((("false_approval", "(a) harmful drugs approved"), ("false_rejection", "(b) safe drugs rejected"))):
    Mx = np.array([[G[s + p][key] for p in po] for s in st]); im = ax[k].imshow(Mx, cmap="Oranges" if k == 0 else "Blues", vmin=0, vmax=max(0.01, Mx.max()))
    for i in range(4):
        for j in range(4): ax[k].text(j, i, f"{100 * Mx[i, j]:.1f}%", ha="center", va="center", fontsize=8, color=("white" if Mx[i, j] > 0.5 * max(0.01, Mx.max()) else INK))
    ax[k].set_xticks(range(4)); ax[k].set_xticklabels(["naive", "skeptical", "mandated\ntriple", "registration"], fontsize=7.5)
    ax[k].set_yticks(range(4)); ax[k].set_yticklabels(["full", "pairs only", "pooled", "selective"] if k == 0 else [""] * 4)
    ax[k].set_title(tit, fontsize=9); ax[k].spines[:].set_visible(False)
ax[0].set_ylabel("sponsor strategy"); ax[0].set_xlabel("regulator policy"); ax[1].set_xlabel("regulator policy")
rows = R["11d"]["rows"]; tau = R["11d"]["tau"]; p = np.array([r["p"] for r in rows]); mid = np.array([r["mid"] for r in rows]); hi = np.array([r["hi"] for r in rows])
discK = np.ones_like(p); discN = (p <= tau) >= (mid <= tau)
bins = np.linspace(p.min(), p.max(), 16); cen = 0.5 * (bins[1:] + bins[:-1]); idx = np.digitize(p, bins) - 1
fr = lambda d: [np.mean(d[idx == b]) if np.any(idx == b) else np.nan for b in range(15)]
ax[2].plot(cen, fr(discK), "o-", color=BLUE, label="skeptical regulator"); ax[2].plot(cen, fr(discN.astype(float)), "s-", color=ORANGE, label="naive regulator")
ax[2].axvline(tau, color=CRIT, ls="--", lw=1); ax[2].set_xlabel("true P(any harm)"); ax[2].set_ylabel("share disclosing the triple"); ax[2].legend(frameon=False, fontsize=7.5)
ax[2].set_title("(c) unravelling under skepticism;\nharmful sponsors hide from a naive regulator", fontsize=9)
ax[3].hist(np.array(R["illustrations"]["I3_rounds"]), bins=np.arange(1, 25) - 0.5, color=GREY); ax[3].set_xlabel("post-market rounds until the triple is seen")
ax[3].set_ylabel("number of false approvals (pairs only, naive)"); ax[3].set_title(f"(d) catch-up: {len(R['illustrations']['I3_rounds'])} false approvals, mean {R['illustrations']['I3_mean_rounds']:.1f} rounds", fontsize=9)
ax[3].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.tight_layout(); save(fig, "fig_game")

fig, ax = plt.subplots(figsize=(4.6, 3.2)); width = np.array([r["hi"] - r["lo"] for r in rows])
ax.scatter(p, width, s=6, color=GREY, alpha=0.6); ax.axvline(tau, color=CRIT, ls="--", lw=1); ax.set_xlabel("true P(any harm)"); ax.set_ylabel("width of the hiding interval")
ax.set_title("what pair-only disclosure hides, by drug", fontsize=9); plt.tight_layout(); save(fig, "fig_hiding")

# ---------------- ML
I = R["illustrations"]; fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
ma = lambda y, k=150: np.convolve(y, np.ones(k) / k, mode="valid")
for key, col, lab in (("bandit", BLUE, "contextual bandit (learned)"), ("rule", INK, "closed form $\\arg\\max_S|\\hat v_S|$"), ("random", GREY, "random probe")):
    ax[0].plot(ma(np.array(I["I1_curves"][key])), color=col, lw=1.6, label=lab)
ax[0].set_xlabel("episode"); ax[0].set_ylabel("realised VOI (moving average)"); ax[0].legend(frameon=False, fontsize=7.5); ax[0].set_title("(a) RL learns the probe rule that Theorem C gives in closed form", fontsize=9)
tr = np.array(I["I2_traj"]); trS = np.array(I["I2_trajS"]); eq = I["I2_eq"]
ax[1].plot(tr[:, 0], tr[:, 1], color=ORANGE, lw=1, label="simultaneous gradient play"); ax[1].plot(trS[:, 0], trS[:, 1], color=BLUE, lw=2, label="regulator commits first")
ax[1].scatter([eq[0]], [eq[1]], color=INK, zorder=5, label="mixed equilibrium"); ax[1].set_xlabel("P(sponsor hides)"); ax[1].set_ylabel("P(regulator skeptical)")
ax[1].set_xlim(-0.02, 1.02); ax[1].set_ylim(-0.02, 1.02); ax[1].legend(frameon=False, fontsize=7.5)
ax[1].set_title("(b) inspection game: GAN-like rotation without\ncommitment, convergence with it (stylised)", fontsize=9)
plt.tight_layout(); save(fig, "fig_ml")
# ---- fig_decide: Theorem D and Study 11b ----
sys.path.insert(0, os.path.join(ENG, "study11b_settle"))
import design11b as D11
from pce import segment_decision as SD
from numpy.polynomial import polynomial as Pn
RB = json.load(open(os.path.join(ENG, "study11b_settle", "results_study11b.json")))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
tn, spec, u0, a0, d = D11.build("D2"); P_, mu_, v_, tau_, r0_, Rw = D11.truth(tn); ps_ = [SD.U_poly(mu_, v_, u, range(3)) for u in D11._acts(spec)]
rr = np.linspace(Rw[0], Rw[1], 200)
ax[0].plot(rr, Pn.polyval(rr, ps_[0]), color=BLUE, lw=2, label="act A (treat): slope $-v_S$")
ax[0].plot(rr, Pn.polyval(rr, ps_[1]), color=ORANGE, lw=2, label="act C (flat: slope 0)")
ax[0].axhline(u0, color=CRIT, lw=1, ls="--", label="requirement $u_0$")
ax[0].axvspan(r0_ - d, r0_ + d, color=AQUA, alpha=0.15, label=f"post-probe interval at the settling radius $d={d}$")
ax[0].axvline(r0_, color=INK2, lw=1, ls=":"); ax[0].text(r0_ + 0.02, Pn.polyval(Rw[1], ps_[0]), "truth $r_0$", fontsize=8, color=INK2, va="top")
for x_ in Rw: ax[0].axvline(x_, color=INK2, lw=1.5)
ax[0].set_xlabel("$r$ along the hiding segment (walls marked)"); ax[0].set_ylabel("expected utility")
ax[0].set_title("(a) design D2: on the whole segment the set rule picks C;\nonce the post-probe interval fits inside the band it picks A", fontsize=9)
ax[0].legend(frameon=False, fontsize=7.5, loc="upper left")
cols = {"D1": BLUE, "D2": ORANGE, "D3": AQUA, "D5": VIOLET}; xx_ = np.exp(np.linspace(np.log(0.5), np.log(8), 50))
for k, c in cols.items():
    b = RB[k]; ns = b["ns"]; ns_ = sorted(int(n) for n in b["predicted"]); x = [n / b["nstar"] for n in ns_]
    ax[1].plot(x, [b["predicted"][str(n)] for n in ns_], color=c, lw=1.6, label=f"{k} predicted ($n^*={b['nstar']:.0f}$)")
    ax[1].plot(x, [b["empirical"][str(n)] for n in ns_], "o", color=c, ms=6, mec="white", mew=1)
ax[1].axhline(0.5, color=GREY, lw=0.8, ls=":"); ax[1].axvline(1, color=GREY, lw=0.8, ls=":")
ax[1].set_xscale("log"); ax[1].set_xticks([0.5, 1, 2, 4, 8]); ax[1].set_xticklabels(["1/2", "1", "2", "4", "8"]); ax[1].minorticks_off(); ax[1].set_xlabel("$n / n^*$, with $n^* = (z\\tau/d)^2$"); ax[1].set_ylabel("share of replicates settled on the oracle act")
ax[1].set_title("(b) Study 11b: settling curves, predicted (lines) and observed (dots);\nthe median is at $n^*$ (S1, S2 hold)", fontsize=9); ax[1].legend(frameon=False, fontsize=7.5, loc="upper left")
b = RB["D4"]; ns_ = sorted(int(n) for n in b["predicted"])
ax[2].plot(ns_, [b["predicted"][str(n)] for n in ns_], color=CRIT, lw=1.6, label="predicted plateau")
ax[2].plot(ns_, [b["empirical"][str(n)] for n in ns_], "o", color=CRIT, ms=6, mec="white", mew=1, label="observed")
ax[2].set_ylim(0, 1); ax[2].set_xscale("log"); ax[2].set_xlabel("$n$ (draws per window)"); ax[2].set_ylabel("share settled")
ax[2].set_title("(c) D4, the truth on the crossing ($d=0$): the decision never settles;\nthe rate stays on the predicted plateau (S3 holds)", fontsize=9); ax[2].legend(frameon=False, fontsize=7.5)
plt.tight_layout(); save(fig, "fig_decide")

# ---- fig_inject: Part II and Study 12 ----
sys.path.insert(0, os.path.join(ENG, "study12_inject"))
import design12 as D12
import run_study12 as R12
from pce import injection as IJ
RS = json.load(open(os.path.join(ENG, "study12_inject", "results_study12.json"))); H = RS["hidden"]
fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
ee = np.linspace(0, 0.2, 200); rpath = H["r0"] + ee * H["s"] / H["vT"]; Rw = D12.hidden()["R"]
ax[0].axhspan(Rw[0], Rw[1], color=BLUE, alpha=0.08, label="regulator's hiding segment $R$ (unchanged)")
ax[0].plot(ee, rpath, color=CRIT, lw=2, label="$\\hat r$ under injection: $r_0+\\varepsilon s/v_T$")
ax[0].axhline(H["rc"], color=ORANGE, ls="--", lw=1.2, label="switch point: treat A = flat act C")
eh = float(H["eps_half_emp"])
for x_, lab_, c_, y_, ha_ in ((eh, "$\\varepsilon_{1/2}=0.058$\n(N = $10^5$)", ORANGE, 1.50, "right"), (H["eps_star"], "$\\varepsilon^*=0.071$", ORANGE, 1.25, "left"), (H["eps_det"], "$\\varepsilon_{det}=0.164$", VIOLET, 1.50, "left")):
    ax[0].axvline(x_, color=c_, lw=1, ls=":"); ax[0].text(x_ + (0.002 if ha_ == "left" else -0.002), y_, lab_, fontsize=8, color=c_, ha=ha_)
ax[0].axvspan(eh, H["eps_det"], color=ORANGE, alpha=0.08)
ax[0].set_xlabel("injected share $\\varepsilon$ (whole records, pair tables untouched)"); ax[0].set_ylabel("$r$")
ax[0].set_title("(a) the hidden channel: $\\hat r$ walks along the segment;\nshaded: decision taken over, nothing flagged", fontsize=9)
ax[0].legend(frameon=False, fontsize=7, loc="lower left")
ks = sorted(float(k) for k in H["pred_A"])
ax[1].plot(ks, [H["pred_A"][str(k) if str(k) in H["pred_A"] else repr(k)] for k in ks], color=BLUE, lw=1.6, label="share choosing the genuine act A: predicted")
ax[1].plot(ks, [H["emp_A"][str(k) if str(k) in H["emp_A"] else repr(k)] for k in ks], "o", color=BLUE, ms=5, mec="white", label="observed")
ax[1].plot(ks, [H["pred_flag"][str(k) if str(k) in H["pred_flag"] else repr(k)] for k in ks], color=VIOLET, lw=1.6, label="share flagged: predicted")
ax[1].plot(ks, [H["emp_flag"][str(k) if str(k) in H["emp_flag"] else repr(k)] for k in ks], "s", color=VIOLET, ms=5, mec="white", label="observed")
ax[1].axvspan(eh, H["eps_det"], color=ORANGE, alpha=0.08); ax[1].set_xlabel("injected share $\\varepsilon$"); ax[1].set_ylabel("share of replicates")
ax[1].set_title("(b) Study 12 (H1-H3): the decision flips near 6% injection;\ndetection starts near 15%", fontsize=9); ax[1].legend(frameon=False, fontsize=7, loc="center right")
T = D12.tetrad_design(); P4 = T["P"]; Nj = IJ.tetrad_jacobian(P4)
tiltF = lambda X: (lambda mu, M, mS: mS((0, 1, 2)) / np.prod(fit_v(mu, M)[:3]))(*moments(X)); tilt = IJ.influence(tiltF, P4).ravel()
_, SigT = R12.tetrad_stat(R12.exact_tabs(P4, R12.PAIRS4)); L = np.linalg.cholesky(SigT); Nw = np.linalg.solve(L, Nj)
k_ = int(np.argmax(tilt)); dv = np.abs(Nw[:, k_] - Nw @ IJ.as_vec(P4)).max(); ds = np.linspace(0, 1.2 * dv, 60)
front = [IJ.stealth_tilt_lp(P4, tilt, Nw, d_)[0] for d_ in ds]
ax[2].plot(ds, front, color=AQUA, lw=2, label="best first-order tilt of $\\hat r$ at that visibility")
ax[2].plot([0], [T["tilt_s"]], "o", color=AQUA, ms=8, mec="white", label=f"stealth law (tetrads blind): tilt {T['tilt_s']:.1f}")
ax[2].plot([dv], [T["tilt_v"]], "D", color=CRIT, ms=7, mec="white", label=f"vertex law (point mass): tilt {T['tilt_v']:.1f}")
ax[2].set_xlabel("first-order tetrad displacement per unit $\\varepsilon$ (noise units)"); ax[2].set_ylabel("first-order tilt $d\\hat r/d\\varepsilon$")
ax[2].set_title("(c) m = 4: the stealth-tilt frontier is concave and piecewise linear;\ninvisibility costs about 6% of the tilt here", fontsize=9); ax[2].legend(frameon=False, fontsize=7, loc="lower right")
Tt = RS["tetrad"]; xs_ = np.arange(3); wd = 0.36
for j, (nm, c_) in enumerate((("stealth", AQUA), ("vertex", CRIT))):
    ax[3].bar(xs_ + (j - 0.5) * wd, [Tt["emp"][f"{nm}_{n}"] for n in D12.N_TET], width=wd - 0.04, color=c_, label=f"{nm}: observed")
    ax[3].plot(xs_ + (j - 0.5) * wd, [Tt["pred"][f"{nm}_{n}"][0] for n in D12.N_TET], "_", color=INK, ms=18, mew=2)
ax[3].axhline(0.05, color=GREY, ls=":", lw=1); ax[3].set_xticks(xs_); ax[3].set_xticklabels([f"N = {n}" for n in D12.N_TET])
ax[3].set_ylabel("tetrad test rejection rate (level 0.05)"); ax[3].plot([], [], "_", color=INK, ms=12, mew=2, label="predicted")
ax[3].set_title("(d) Study 12 (H4): equal tilt, opposite fate;\nthe stealth law stays at the test's level", fontsize=9); ax[3].legend(frameon=False, fontsize=7, loc="center right")
plt.tight_layout(); save(fig, "fig_inject")

# ---- fig_oneline: Theorem H and G(4) ----
from pce import duality as DU
from scipy.linalg import null_space
fig, ax = plt.subplots(1, 2, figsize=(14, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
h0 = D12.hidden(); (ml, mh), (pl, ph), m0 = DU.m3_line(h0["P"]); vT = h0["vT"]; mc = h0["rc"] * vT
ax[0].plot([pl, ph], [0, 0], color=GREY, lw=10, solid_capstyle="butt", alpha=0.35, label="positivity part: all laws with these pair tables (adversary)")
ax[0].plot([ml, mh], [0, 0], color=BLUE, lw=10, solid_capstyle="butt", alpha=0.8, label="model part: the regulator's hiding segment $v_T R$")
ax[0].plot([m0], [0], "o", color=INK, ms=9, zorder=5); ax[0].text(m0, -0.2, "truth", ha="center", fontsize=8)
ax[0].plot([mc], [0], "|", color=ORANGE, ms=28, mew=3, zorder=5); ax[0].text(mc, -0.42, "decision boundary $r_c$", ha="center", fontsize=8, color=ORANGE)
for x_ in (ml, mh): ax[0].plot([x_], [0], "|", color=VIOLET, ms=28, mew=3, zorder=5)
push_right = (mc > m0); near, far = (mh, ml) if push_right else (ml, mh)
ax[0].text(near, 0.42, "detection boundary\n(wall on the pushed side)", ha="center", fontsize=8, color=VIOLET); ax[0].text(far, 0.42, "wall", ha="center", fontsize=8, color=VIOLET)
ax[0].annotate("", xy=(mc, 0.12), xytext=(m0, 0.12), arrowprops=dict(arrowstyle="->", color=CRIT, lw=1.5))
ax[0].text(0.5 * (m0 + mc), 0.2, "injection moves\n$m_T$ by $\\varepsilon s$", ha="center", fontsize=8, color=CRIT)
ax[0].set_ylim(-0.6, 0.7); ax[0].set_yticks([]); ax[0].spines["left"].set_visible(False)
ax[0].set_xlabel("triple moment $m_T$ (the regulator reads $r = m_T / v_T$)")
ax[0].set_title("(a) Theorem H: the adversary's segment and the regulator's are two parts of one line;\nthe decision boundary is nearer than the detection boundary (design D2)", fontsize=9)
ax[0].legend(frameon=False, fontsize=7.5, loc="lower right", bbox_to_anchor=(1.0, -0.02))
es = np.array([1e-3, 2e-3, 5e-3, 1e-2, 2e-2]); rng_ = np.random.default_rng(4)
for name, (b1, b2), col in (("generic truth", ([.8, .3, .7, .25], [.2, .7, .15, .7]), BLUE), ("two-coincidence stratum", ([.8, .3, .4, .5], [.2, .7, .4, .5]), CRIT)):
    Pq = law(.35, np.array(b1), np.array(b2)); pq, Bq, Jq, Hq = DU.jet(Pq); Kq = null_space(Jq, rcond=1e-7)
    A = Kq @ rng_.normal(size=Kq.shape[1]); A /= np.linalg.norm(A); q = 0.5 * np.einsum("kij,i,j->k", Hq, A, A); b = np.linalg.pinv(Jq, rcond=1e-7) @ (-q)
    st = [np.max(np.abs(DU._tet_vec(pq + Bq @ (e * A)))) for e in es]; bt = [np.max(np.abs(DU._tet_vec(pq + Bq @ (e * A + e * e * b)))) for e in es]
    ax[1].loglog(es, st, "--", color=col, lw=1.4, label=f"{name}: straight stealthy injection")
    ax[1].loglog(es, bt, "-", color=col, lw=2, label=f"{name}: bent injection")
ax[1].set_xlabel("injected share $\\varepsilon$"); ax[1].set_ylabel("largest tetrad $|T(P_\\varepsilon)|$")
ax[1].set_title("(b) Theorem G(4): bending removes the $\\varepsilon^2$ term where the model is smooth\n(slope 3); on the singular stratum the obstruction keeps it (slope 2)", fontsize=9)
ax[1].legend(frameon=False, fontsize=7)
plt.tight_layout(); save(fig, "fig_oneline")

# ---- fig_worked: the worked example ----
WE = json.load(open(os.path.join(HERE, "worked_example.json")))
fig = plt.figure(figsize=(15, 9.6)); gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.22)
ax = fig.add_subplot(gs[0, 0]); keys = list(WE["cells"]); vals = [WE["cells"][k] for k in keys]
chi_ = {k: (1 if (k.count("0") % 2 == 0) else -1) for k in keys}
cols = [ORANGE if chi_[k] < 0 else AQUA for k in keys]
ax.barh(range(8), vals, color=cols, height=0.7)
for i, (k, v_) in enumerate(zip(keys, vals)):
    ax.text(v_ + 0.004, i, f"{v_:.3f}   $\\chi={'+' if chi_[k] > 0 else '-'}1$", va="center", fontsize=8)
ax.set_yticks(range(8)); ax.set_yticklabels([f"ABC = {k}" for k in keys]); ax.invert_yaxis(); ax.set_xlim(0, 0.42)
ax.set_xlabel("probability of the cell (the truth, exact to three decimals)")
ax.set_title("(a) step 0-1: the eight cells; the hidden channel adds $s\\chi$.\n$s\\leq 0.027$ (rarest orange cell, 110) and $s\\geq -0.013$ (rarest green cell, 111)", fontsize=9)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=ORANGE, label="$\\chi=-1$: loses mass when $s>0$"), Patch(color=AQUA, label="$\\chi=+1$: gains mass when $s>0$")], frameon=False, fontsize=7.5, loc="lower right")
ax = fig.add_subplot(gs[0, 1]); L2 = WE["level2"]; Dd = WE["decision"]; Rr = L2["R"]; rr = np.linspace(Rr[0], Rr[1], 50)
UT = Dd["U_T_line"][0] + Dd["U_T_line"][1] * rr
ax.plot(rr, UT, color=BLUE, lw=2, label="triple T: $1-P(\\mathrm{any\\ harm})$, slope $-v_T=0.00635$")
ax.axhline(Dd["U_P"], color=ORANGE, lw=2, label="pair AB: $0.872-P(A\\vee B)=0.332$ (flat)")
ax.axvline(L2["r0"], color=INK2, ls=":", lw=1); ax.text(L2["r0"] + 0.02, 0.3392, "truth $r_0=-0.408$", fontsize=8, color=INK2)
ax.axvline(Dd["r_c"], color=CRIT, ls="--", lw=1); ax.text(Dd["r_c"] + 0.02, 0.3376, "switch $r_c=-0.566$", fontsize=8, color=CRIT)
ax.plot([Rr[0]], [UT[0]], "v", color=BLUE, ms=8); ax.text(Rr[0] + 0.04, UT[0] - 0.0002, "worst of T at the wall: 0.33075", fontsize=8, color=BLUE, va="top"); ax.set_ylim(UT[0] - 0.0012, UT[-1] + 0.0006)
ax.set_xlabel("$r$ on the hiding segment $[-0.763,\\ 0.573]$ (walls: $a_{2A}=0$, $a_{1B}=0$)"); ax.set_ylabel("expected utility")
ax.set_title("(b) steps 7-8: before the probe the set rule picks the pair (0.332 > 0.33075);\nat the truth the triple is better (0.333); Savage's regret already picks the triple", fontsize=9)
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
ax = fig.add_subplot(gs[1, 0]); OL = WE["one_line"]; H_ = WE["hidden_channel"]
ax.plot(OL["positivity_part_harm"], [0, 0], color=GREY, lw=12, solid_capstyle="butt", alpha=0.35, label="level one: all laws with these pairs  [0.654, 0.694]")
ax.plot(OL["model_part_harm"], [0, 0], color=BLUE, lw=12, solid_capstyle="butt", alpha=0.85, label="level two: the hiding segment  [0.66077, 0.66925]")
ax.plot([0.667], [0], "o", color=INK, ms=9, zorder=5); ax.text(0.667, -0.22, "truth 0.667", ha="center", fontsize=8)
ax.plot([0.668], [0], "|", color=CRIT, ms=30, mew=3, zorder=6); ax.text(0.668, 0.3, "decision boundary 0.668", ha="center", fontsize=8, color=CRIT)
ax.plot([OL["model_part_harm"][1]], [0], "|", color=VIOLET, ms=30, mew=3, zorder=6); ax.text(OL["model_part_harm"][1] + 0.0003, -0.42, "detection\nboundary 0.66925", ha="left", fontsize=8, color=VIOLET)
ax.annotate("", xy=(0.6685, 0.13), xytext=(0.667, 0.13), arrowprops=dict(arrowstyle="->", color=CRIT, lw=1.5))
ax.text(0.6635, 0.45, f"injection of a share $\\varepsilon$ adds $0.027\\,\\varepsilon$ to the apparent harm:\nflip at $\\varepsilon^*={H_['eps_star']:.3f}$, flagged at $\\varepsilon_{{det}}={H_['eps_det']:.3f}$", fontsize=8, color=CRIT)
ax.set_xlim(0.652, 0.696); ax.set_ylim(-0.7, 0.8); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel("triple-combination harm (the line of Theorem H in harm units)")
ax.set_title("(c) steps 1, 2, 10, 11: one line; the decision boundary (0.668) is nearer the truth\nthan the detection boundary (0.66925), so a stealthy takeover exists", fontsize=9)
ax.legend(frameon=False, fontsize=7.5, loc="lower left")
ax = fig.add_subplot(gs[1, 1]); from matplotlib.patches import Polygon
L = {"d1": (0.0, 0.0), "d2": (1.6, -0.25), "d3": (1.1, 0.55), "w1": (0.9, 1.7), "w2": (0.9, -1.45)}
for apex, col in (("w1", BLUE), ("w2", ORANGE)):
    for a_, b_ in itertools.combinations(("d1", "d2", "d3"), 2): ax.add_patch(Polygon([L[apex], L[a_], L[b_]], closed=True, color=col, alpha=0.12, lw=0))
ax.add_patch(Polygon([L["d1"], L["d2"], L["d3"]], closed=True, color=GREY, alpha=0.35, lw=0))
for a_, b_ in itertools.combinations(L, 2):
    if {a_, b_} != {"w1", "w2"}: ax.plot(*zip(L[a_], L[b_]), color=INK2, lw=1, ls="--" if {a_, b_} in ({"d1", "d2"}, {"w2", "d3"}) else "-")
for k_, (x_, y_) in L.items(): ax.plot([x_], [y_], "o", color=INK, ms=6); ax.text(x_ + 0.06, y_ + 0.06, k_, fontsize=9)
ax.text(0.8, -1.95, "two tetrahedra on d1d2d3\n(an actor absent)", ha="center", fontsize=8)
B = {"w1": (3.4, 1.4), "w2": (3.4, -1.2), "d1": (2.6, 0.05), "d2": (4.3, -0.3), "d3": (4.0, 0.6)}
for i_, col in (("d1", VIOLET), ("d2", VIOLET), ("d3", VIOLET)): ax.add_patch(Polygon([B["w1"], B["w2"], B[i_]], closed=True, color=col, alpha=0.12, lw=0))
for i_ in ("d1", "d2", "d3"): ax.plot(*zip(B["w1"], B[i_]), color=INK2, lw=1); ax.plot(*zip(B["w2"], B[i_]), color=INK2, lw=1)
ax.plot(*zip(B["w1"], B["w2"]), color=CRIT, lw=2.5)
for k_, (x_, y_) in B.items(): ax.plot([x_], [y_], "o", color=INK, ms=6); ax.text(x_ + 0.06, y_ + 0.06, k_, fontsize=9)
ax.text(3.45, -1.95, "three pages on the axis w1w2\n(both present, differing in one variable)", ha="center", fontsize=8)
ax.text(5.0, 1.75, "glue equal labels: the complex has\nfacets {w1,d1,d2,d3}, {w2,d1,d2,d3},\n{w1,w2,d1}, {w1,w2,d2}, {w1,w2,d3};\nminimal non-faces {w1,w2,di,dj};\nGroebner basis w1(1 - w1) di dj (i<j);\nideal = (w1)&(w2)&(d1,d2)&(d1,d3)&(d2,d3)\n\nthe truth: w1w2 = 0.24,\nd = (0.45, -0.40, 0.30), all non-zero:\nit lies on no facet, so the fibre is\nflat there (a segment over the pairs)".replace("&", " $\\cap$ "), fontsize=8, va="top")
ax.set_xlim(-0.3, 7.6); ax.set_ylim(-2.3, 2.0); ax.axis("off")
ax.set_title("(d) step 6: the Stanley-Reisner complex of the fibre over an independent law", fontsize=9)
save(fig, "fig_worked")

# ---- fig_toric: Step 13 ----
WE = json.load(open(os.path.join(HERE, "worked_example.json"))); TO = WE["toric"]
from fractions import Fraction as Fr
s_lo_, s_hi_ = float(Fr(TO["s_range"][0])), float(Fr(TO["s_range"][1])); sb = TO["birch_s"]; smono = float(Fr(TO["monotone_midpoint_s"]))
mp = [h - 0.667 for h in WE["level2"]["interval"]]; s_dec, s_det = 0.668 - 0.667, WE["level2"]["interval"][1] - 0.667
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1.35, 0.8, 1.05]})
a0 = ax[0]; a0.plot([s_lo_, s_hi_], [0, 0], color=GREY, lw=12, solid_capstyle="butt", alpha=0.35, label="moment segment of CP$^1$ = all laws with these pairs")
a0.plot(mp, [0, 0], color=BLUE, lw=12, solid_capstyle="butt", alpha=0.85, label="the two-type model's part (the hiding segment)")
marks = [(s_lo_, "fixed point\ncell 111 = 0", INK, 0.32, "center"), (s_hi_, "fixed point\ncell 110 = 0", INK, 0.32, "center"), (0.0, "truth", INK, -0.25, "center"),
         (sb, "Birch point\n(max entropy)", AQUA, 0.32, "center"), (smono, "midpoint\n(monotone fibre)", VIOLET, 0.32, "center"),
         (s_dec, "decision\nboundary", CRIT, 0.62, "center"), (s_det, "detection boundary", ORANGE, -0.55, "left")]
for x_, lab_, c_, y_, ha_ in marks:
    a0.plot([x_], [0], "|" if lab_ != "truth" else "o", color=c_, ms=26 if lab_ != "truth" else 9, mew=2.5, zorder=6)
    a0.text(x_, y_, lab_, ha=ha_, va="center", fontsize=7.5, color=c_)
a0.set_ylim(-0.85, 0.95); a0.set_yticks([]); a0.spines["left"].set_visible(False); a0.set_xlim(s_lo_ - 0.003, s_hi_ + 0.003)
a0.set_xlabel("moment coordinate $s$ = third cumulant $m_{ABC} - 0.002592$  (triple harm $= 0.667 + s$)")
a0.set_title("(a) the boundary fibre is a Delzant segment of lattice length 0.040;\nits facet values at the truth, 0.027 and 0.013, are the channel capacities", fontsize=9)
a0.legend(frameon=False, fontsize=7.2, loc="lower center", bbox_to_anchor=(0.5, -0.02))
a1 = ax[1]; th = np.linspace(0, 2 * np.pi, 200); a1.plot(np.cos(th), np.sin(th), color=INK2, lw=1)
h_of = lambda x_: -1 + 2 * (x_ - s_lo_) / (s_hi_ - s_lo_)
def lat(x_, col, lw, ls="-", lab=None):
    y_ = h_of(x_); r_ = np.sqrt(max(0.0, 1 - y_ * y_)); a1.plot(r_ * np.cos(th[:101]), y_ + 0.18 * r_ * np.sin(th[:101]) * 0 - 0.12 * r_ * np.sin(th[:101]), color=col, lw=lw, ls=ls, label=lab)
    a1.plot(r_ * np.cos(th[100:]), y_ - 0.12 * r_ * np.sin(th[100:]), color=col, lw=lw * 0.6, ls=":")
y1, y2 = h_of(mp[0]), h_of(mp[1]); a1.axhspan(y1, y2, xmin=0.05, xmax=0.95, color=BLUE, alpha=0.12)
lat(0.0, INK, 1.8, lab="truth"); lat(sb, AQUA, 1.6, lab="Birch circle"); lat(smono, VIOLET, 1.6, "--", lab="equator (monotone fibre)")
lat(s_dec, CRIT, 1.4, lab="decision boundary"); lat(s_det, ORANGE, 1.4, lab="detection boundary")
a1.plot([0, 0], [-1, 1], "o", color=INK, ms=6); a1.text(0.06, 1.04, "fixed point: Q of the adversary (cell 110 = 0)", fontsize=7); a1.text(0.06, -1.12, "fixed point (cell 111 = 0)", fontsize=7)
a1.set_aspect("equal"); a1.axis("off"); a1.set_xlim(-1.25, 1.25); a1.set_ylim(-1.3, 1.25)
a1.set_title("(b) CP$^1$: each law with these pairs is a circle;\nheight is the moment map; the blue band is the hiding segment", fontsize=9)
a1.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2)
a2 = ax[2]; ss = np.linspace(s_lo_, s_hi_, 400)
bc = [float(Fr(c)) for c in TO["birch_cubic"]]; dq = [float(Fr(c)) for c in TO["hyperdet_quadratic"]]
B_ = np.polyval(bc, ss); D_ = np.polyval(dq, ss)
a2.plot(ss, B_ / np.max(np.abs(B_)), color=AQUA, lw=2, label="Birch binomial $p_{000}p_{011}p_{101}p_{110}-p_{001}p_{010}p_{100}p_{111}$ (a cubic on the line)")
a2.plot(ss, D_ / np.max(np.abs(D_)), color=VIOLET, lw=2, label="Cayley hyperdeterminant $= m_{ABC}^2+4\\Delta_m$ (a quadratic, $>0$)")
a2.axhline(0, color=GREY, lw=0.8); a2.axvline(sb, color=AQUA, ls=":", lw=1); a2.axvline(0, color=INK, ls=":", lw=1)
a2.text(sb, -0.9, " one real root:\n the Birch point", fontsize=7.5, color=AQUA); a2.text(0.012, 0.12, "roots of Det: $-0.00259\\pm0.0127i$\n(complex: the line never\nleaves real rank 2)", fontsize=7.5, color=VIOLET)
a2.set_xlabel("moment coordinate $s$"); a2.set_ylabel("scaled value"); a2.set_ylim(-1.1, 1.1)
a2.set_title("(c) two hypersurfaces met by the line: the no-three-way model\n(once, Birch) and the hyperdeterminant (only at complex points)", fontsize=9)
a2.legend(frameon=False, fontsize=6.6, loc="upper left")
plt.tight_layout(); save(fig, "fig_toric")

# ---- fig_worked4 ----
W4 = json.load(open(os.path.join(HERE, "worked_example4.json")))
sys.path.insert(0, ENG); import worked_example4 as E4
fig, ax = plt.subplots(1, 4, figsize=(20, 4.8), gridspec_kw={"width_ratios": [0.9, 1.0, 1.1, 1.1]})
a = ax[0]; cells = W4["cells"]; grid = np.zeros((4, 4)); keys = ["00", "01", "10", "11"]
for i, ab in enumerate(keys):
    for j, cd in enumerate(keys): grid[i, j] = float(Fr(cells[ab + cd]))
a.imshow(grid, cmap="Blues", vmin=0, vmax=grid.max() * 1.15)
fac = set(W4["fibre"]["facet_cells"])
for i, ab in enumerate(keys):
    for j, cd in enumerate(keys):
        k = ab + cd; a.text(j, i, f"{grid[i, j]:.4f}", ha="center", va="center", fontsize=8, color=INK)
        if k in fac: a.add_patch(plt.Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False, ec=CRIT, lw=2))
a.set_xticks(range(4)); a.set_xticklabels([f"CD={k}" for k in keys], fontsize=8); a.set_yticks(range(4)); a.set_yticklabels([f"AB={k}" for k in keys], fontsize=8)
a.set_title("(a) stage 1, 3: the sixteen cells (exact);\nred: the nine facets, whose values are the Kaehler data", fontsize=9)
a = ax[1]; fn = list(W4["faces"]); xx_ = np.arange(4); wd = 0.36
a.bar(xx_ - wd / 2, [np.log(W4["three_way"]["truth_odds_ratios"][k]) for k in fn], width=wd, color=BLUE, label="the truth")
a.bar(xx_ + wd / 2, [np.log(W4["three_way"]["global_maxent_face_odds_ratios"][k]) for k in fn], width=wd, color=AQUA, label="global pairwise max-entropy law (induced)")
a.axhline(0, color=INK2, lw=1); a.set_xticks(xx_); a.set_xticklabels(fn); a.set_ylabel("log three-way odds ratio of the face")
a.set_title("(b) stage 2: three-way interactions on the four faces;\neach face's own Birch point is 0; marginalising the pairwise\nmaximum-entropy law induces a small negative one on every face", fontsize=9)
a.legend(frameon=False, fontsize=7.5, loc="upper right")
a = ax[2]; verts = E4.verts; cA = np.array([np.prod([x[i] - E4.mu[i] for i in (0, 1, 2)]) for x in E4.X]); cB = np.array([np.prod([x[i] - E4.mu[i] for i in (0, 1, 3)]) for x in E4.X])
pts = np.array([[cA @ (E4.p + E4.B @ y), cB @ (E4.p + E4.B @ y)] for y, _ in verts])
from scipy.spatial import ConvexHull
hh = ConvexHull(pts); poly = pts[hh.vertices]; a.fill(poly[:, 0], poly[:, 1], color=GREY, alpha=0.3, label="all laws with these pairs (projected)")
a.plot(pts[:, 0], pts[:, 1], "o", color=INK2, ms=3, label="projected vertices (20)")
rr = np.linspace(E4.R4[0], E4.R4[1], 50); a.plot(E4.vS["ABC"] * rr, E4.vS["ABD"] * rr, color=BLUE, lw=3, label="the two-type model (hiding segment)")
a.plot([E4.mS((0, 1, 2))], [E4.mS((0, 1, 3))], "o", color=INK, ms=8, label="truth")
mE = W4["three_way"]["global_maxent_m_S"]; a.plot([mE["ABC"]], [mE["ABD"]], "s", color=AQUA, ms=7, label="global max-entropy law")
fb = W4["three_way"]["face_birch_m_S"]; a.axvline(fb["ABC"], color=AQUA, ls=":", lw=1); a.axhline(fb["ABD"], color=AQUA, ls=":", lw=1)
cap = W4["decision4"]["capacity16_in_face_units"]; a.annotate("", xy=(E4.mS((0, 1, 2)) + cap, E4.mS((0, 1, 3))), xytext=(E4.mS((0, 1, 2)), E4.mS((0, 1, 3))), arrowprops=dict(arrowstyle="->", color=CRIT, lw=1.6))
a.text(E4.mS((0, 1, 2)) + 0.002, E4.mS((0, 1, 3)) + 0.0015, "ABC channel on 16 cells", fontsize=7.5, color=CRIT)
a.set_xlabel("$m_{ABC}$"); a.set_ylabel("$m_{ABD}$")
a.set_title("(c) stages 3-5: the 5-dimensional fibre projected on two faces is the\nproduct of their lines; the model is a line (one r); dotted: face Birch points", fontsize=9); a.legend(frameon=False, fontsize=6.6, loc="lower left")
a = ax[3]; D4 = W4["decision4"]; ee = np.linspace(0, 0.07, 50); sh = D4["shift_r_per_eps"]
a.plot(ee, E4.r0 - sh * ee, color=CRIT, lw=2, label="$\\hat r_{ABC}$ under the ABC channel")
a.plot(ee, np.full_like(ee, E4.r0), color=BLUE, lw=2, label="$\\hat r_{ABD}=\\hat r_{ACD}=\\hat r_{BCD}$ (unchanged)")
a.fill_between(ee, E4.r0 - 1.96 * D4["sd_rdiff_n1e5"], E4.r0 + 1.96 * D4["sd_rdiff_n1e5"], color=BLUE, alpha=0.12, label="second-triple test, n = $10^5$ (95%)")
a.axhline(D4["r_c"], color=ORANGE, ls="--", lw=1.2, label="decision boundary $r_c$")
a.axvline(D4["eps_second_triple_n1e5"], color=BLUE, ls=":", lw=1); a.axvline(D4["eps_star"], color=ORANGE, ls=":", lw=1)
a.text(D4["eps_second_triple_n1e5"] - 0.001, -0.6, f"caught at\n{100 * D4['eps_second_triple_n1e5']:.1f}%", fontsize=7.5, color=BLUE, ha="right")
a.text(D4["eps_star"] + 0.001, -0.6, f"flip at\n{100 * D4['eps_star']:.1f}%", fontsize=7.5, color=ORANGE)
a.set_xlabel("injected share $\\varepsilon$ (ABC channel, whole records)"); a.set_ylabel("$r$")
a.set_title(f"(d) stage 5: a second triple catches the channel before it flips\nthe decision; staying on the model instead costs {100 * W4['decision4_model_channel']['eps_star']:.1f}%", fontsize=9)
a.legend(frameon=False, fontsize=6.8, loc="upper right")
plt.tight_layout(); save(fig, "fig_worked4")
# ---- fig_sources: Theorem J and Study 13 ----
S13 = json.load(open(os.path.join(ENG, "study13_sources", "results_study13.json")))
NB13 = S13["design"]["NB"]; Ns13 = np.array([24 * n for n in NB13]); key = lambda a, n: f"{a}|{n}"
fig, axs = plt.subplots(1, 4, figsize=(15.5, 3.7))
a = axs[0]
for adv, rule, col, lab in (("A0", "glue", GREY, "genuine (A0), filter"), ("A1b", "glue", BLUE, "sometimes adversarial (A1b), filter"),
                            ("A2", "naive", ORANGE, "fixed share (A2), pooling"), ("A2", "glue", AQUA, "fixed share (A2), filter")):
    q = np.array([S13["sim"][key(adv, n)]["harm_q"][rule] for n in NB13])
    if adv == "A0": a.plot(Ns13, q[:, 0], ":", color=INK2, lw=1); a.plot(Ns13, q[:, 4], ":", color=INK2, lw=1)
    else: a.fill_between(Ns13, q[:, 0], q[:, 4], color=col, alpha=0.13, lw=0)
    a.plot(Ns13, q[:, 2], "o-", color=col, lw=1.6, ms=3.5, label=lab)
a.axhline(0.667, color=INK2, lw=0.8, ls=":"); a.axhline(0.668, color=CRIT, lw=1, ls="--")
a.text(Ns13[0], 0.6683, "switch 0.668", color=CRIT, fontsize=7); a.text(Ns13[0], 0.6663, "truth 0.667", color=INK2, fontsize=7)
a.set_xscale("log"); a.set_xlabel("triple records $N$"); a.set_ylabel("apparent triple harm"); a.legend(frameon=False, fontsize=6.6, loc="upper right")
a.set_title("(a) bands of the apparent triple harm: 95% of\nreplicates (shaded; dotted: genuine), median (line)", fontsize=9)
a = axs[1]
for rule, col, lab in (("naive", ORANGE, "pooling"), ("glue", BLUE, "gluing filter"), ("drop_C_triple", AQUA, "oracle: drop C's triples"), ("write_off_C", VIOLET, "write C off")):
    a.plot(Ns13, [S13["pred"][key("A1b", n)][rule]["right"] for n in NB13], "-", color=col, lw=1.5, label=lab)
    a.plot(Ns13, [S13["sim"][key("A1b", n)]["rates"][rule]["right"] for n in NB13], "o", color=col, ms=4)
a.plot(Ns13, [S13["sim"][key("A0", n)]["rates"]["glue"]["right"] for n in NB13], "--", color=GREY, lw=1.2, label="filter, no adversary")
a.axhline(0.5, color=INK2, lw=0.6, ls=":"); a.set_xscale("log"); a.set_xlabel("triple records $N$"); a.set_ylabel("share settled on the right act")
a.legend(frameon=False, fontsize=6.6, loc="upper left"); a.set_title("(b) delay under A1b: predicted (lines), observed (dots)", fontsize=9)
a = axs[2]; cs13 = S13["design"]["c_s"]; sm13 = S13["design"]["gm"]["sm"]; NN = np.geomspace(Ns13[0], Ns13[-1], 100)
aA2 = S13["design"]["a2_share"] * S13["design"]["s"] / (sm13 * np.sqrt(4.5 / NN))
for lab, col, mean in (("A0: genuine", GREY, 0 * NN), ("A1: sometimes, risk 0.2", BLUE, S13["design"]["a_risk20"] + 0 * NN),
                       ("A1b: sometimes, risk 0.5", VIOLET, cs13 + 0 * NN), ("A2: fixed share", ORANGE, aA2)):
    a.fill_between(NN, mean - 1, mean + 1, color=col, alpha=0.18, lw=0); a.plot(NN, mean, color=col, lw=1.8, label=lab)
a.axhline(cs13, color=CRIT, ls="--", lw=1); a.text(NN[-1], cs13 - 0.6, f"$c_s$ = {cs13:.2f}", color=CRIT, fontsize=7, ha="right")
a.set_xscale("log"); a.set_xlabel("triple records $N$"); a.set_ylabel("source C's gluing statistic $Z_C$ (mean $\\pm$ 1)")
a.legend(frameon=False, fontsize=6.6, loc="upper left"); a.set_title("(c) paths of the gluing statistic: just enough\nrides flat under $c_s$; a fixed share grows like $\\sqrt{N}$", fontsize=9)
a = axs[3]
a.plot(Ns13, [S13["sim"][key("A2", n)]["rates"]["naive"]["wrong"] for n in NB13], "o-", color=ORANGE, lw=1.6, ms=4, label="pooling under A2: tilted")
a.plot(Ns13, [S13["pred"][key("A2", n)]["naive"]["wrong"] for n in NB13], "-", color=ORANGE, lw=0.8, alpha=0.6)
gw = [max(S13["sim"][key(adv, n)]["rates"]["glue"]["wrong"] for adv in ("A0", "A1", "A1b", "A3", "A2")) for n in NB13]
a.plot(Ns13, gw, "s-", color=BLUE, lw=1.6, ms=4, label="gluing filter, worst adversary")
a.plot(Ns13, [S13["sim"][key("A1b", n)]["rates"]["glue"]["open"] for n in NB13], "^--", color=VIOLET, lw=1.2, ms=4, label="gluing filter under A1b: still open (delay)")
a.set_xscale("log"); a.set_xlabel("triple records $N$"); a.set_ylabel("share of replicates"); a.legend(frameon=False, fontsize=6.6, loc="center left")
a.set_title("(d) the filter turns a tilt into a bounded delay", fontsize=9)
for a in axs:
    a.set_xticks(Ns13); a.set_xticklabels([f"{N // 1000}k" for N in Ns13]); a.minorticks_off()
plt.tight_layout(); save(fig, "fig_sources")

# ---- fig_lifting ----
from pce import lifting as LFT
Wl = LFT.we(); Pl = Wl["P"]
fig, axs = plt.subplots(1, 4, figsize=(15.5, 3.6))
a = axs[0]; ee = np.linspace(0, 0.1, 400)
a.plot(ee, Pl[0, 0, 1] / (1 - ee), color=BLUE, lw=1.8, label="any genuine law (unconstrained Q)")
a.plot(ee, Pl[0, 0, 1] + Pl[1, 1, 1] * ee / (1 - ee), color=ORANGE, lw=1.8, label="keeping the pair tables (the line)")
a.axhline(0.128, color=CRIT, ls="--", lw=1); a.text(0.1, 0.12805, "chamber wall P(001) = 0.128", color=CRIT, fontsize=7, ha="right", va="bottom")
for x, col, t in ((1 / 128, BLUE, "1/128"), (1 / 14, ORANGE, "1/14")): a.axvline(x, color=col, lw=0.8, ls=":"); a.text(x, 0.12705, " " + t, color=col, fontsize=7.5)
a.set_ylim(0.1269, 0.1285); a.set_xlabel("injected share $\\varepsilon$ the decision maker allows"); a.set_ylabel("largest genuine $P'(001)$ in the block")
a.legend(frameon=False, fontsize=6.6, loc="upper left"); a.set_title("(a) the adversarial lift: block of explanations\nreaches the other chamber", fontsize=9)
a = axs[1]; tab, pC = LFT.dilation(Pl)
for k, (ab, (lo_, hi_, tr)) in enumerate(sorted(tab.items())):
    a.plot([lo_, hi_], [k, k], color=BLUE, lw=5, solid_capstyle="butt"); a.plot([tr], [k], "o", color=INK, ms=4)
    a.text(hi_ + 0.008, k, f"width {hi_ - lo_:.3f}", va="center", fontsize=7, color=INK2)
a.plot([0.28], [4], "D", color=ORANGE, ms=5); a.text(0.288, 4, "P(C) = 0.28: a point", va="center", fontsize=7, color=INK2)
a.set_yticks(range(5)); a.set_yticklabels(["A,B = 0,0", "0,1", "1,0", "1,1", "no profile"]); a.set_xlim(0.1, 0.52)
a.set_xlabel("forecast of C (interval: pair tables only; dot: with the triple)"); a.set_title("(b) dilation: the population is pinned,\nthe individual is not", fontsize=9)
a = axs[2]; nn = np.geomspace(30, 1e6, 160); ev = np.array([LFT.evsi(n, Wl) for n in nn]); c_ = 1e-9; net = ev - c_ * nn; k = int(np.argmax(net))
a.plot(nn, ev * 1e5, color=BLUE, lw=1.8, label="EVSI$(n)$"); a.plot(nn, c_ * nn * 1e5, color=GREY, lw=1, ls="--", label="cost $10^{-9}n$")
a.plot(nn, net * 1e5, color=AQUA, lw=1.5, label="net value"); a.axhline(LFT.evpi(Wl) * 1e5, color=INK2, lw=0.7, ls=":")
a.plot([nn[k]], [net[k] * 1e5], "o", color=AQUA); a.text(nn[k], net[k] * 1e5 + 0.6, f"optimum n = {nn[k]:.0f}", fontsize=7, ha="center", color=INK2)
a.axvspan(30, 300, color=GREY, alpha=0.15); a.text(95, 7.5, "flat start", fontsize=7, ha="center", color=INK2)
a.set_xscale("log"); a.set_ylim(-1, 10.5); a.set_xlabel("triple records $n$ in the probe"); a.set_ylabel("value ($\\times10^{-5}$ utility)")
a.legend(frameon=False, fontsize=6.6, loc="center left"); a.set_title("(c) the value of a probe in its size (Bayes)", fontsize=9)
a = axs[3]; doses = np.linspace(0.25, 4, 16)
for retro, col, lab in ((False, ORANGE, "forward-only: accepted sections are final"), (True, BLUE, "backtracking: late failures revoke")):
    a.plot(doses, [LFT.backtrack([c] * 8, retro, draws=15000)[0] for c in doses], "o-", color=col, ms=3.5, lw=1.6, label=lab)
a.set_xlabel("adversary's dose per batch (batch sd), all 8 rounds"); a.set_ylabel("final pooled shift (pooled se)")
a.legend(frameon=False, fontsize=6.6, loc="center right"); a.set_title("(d) backtracking: being caught late\nis free unless the past is revoked", fontsize=9)
plt.tight_layout(); save(fig, "fig_lifting")

# ---- fig_capstone: Theorem L ----
fig, axs = plt.subplots(1, 2, figsize=(13, 3.3), gridspec_kw={"width_ratios": [1.6, 1]})
a = axs[0]; base = 0.667
a.plot([0.654, 0.694], [0, 0], color=GREY, lw=6, solid_capstyle="butt", label="positivity part of the pair line (laws with the pair tables)")
a.plot([0.660769, 0.66925], [0, 0], color=BLUE, lw=6, solid_capstyle="butt", label="model part M (hiding segment)")
a.axvspan(0.654, 0.668, ymin=0.55, ymax=0.95, color=AQUA, alpha=0.18); a.axvspan(0.668, 0.694, ymin=0.55, ymax=0.95, color=ORANGE, alpha=0.18)
a.text(0.661, 0.42, "chamber: approve the triple (T)", fontsize=8, color=INK2, ha="center"); a.text(0.681, 0.42, "chamber: approve the pair (P)", fontsize=8, color=INK2, ha="center")
a.axvline(0.668, color=CRIT, lw=1.5, ls="--"); a.text(0.6687, -0.3, "decision wall P(001) = 0.128", color=CRIT, fontsize=7.5, ha="left")
for x in (0.660769, 0.66925): a.plot([x, x], [-0.12, 0.12], color=BLUE, lw=1.5)
a.text(0.6603, -0.55, "detection walls (ends of M)", color=BLUE, fontsize=7.5, ha="left")
a.plot([base], [0], "o", color=INK, ms=6, zorder=5); a.text(base - 0.0004, -0.25, "truth", fontsize=7.5, ha="right")
a.annotate("", xy=(0.668, 0.2), xytext=(base, 0.2), arrowprops=dict(arrowstyle="<->", color=INK2, lw=1)); a.text(0.6665, 0.2, "margin 0.001", fontsize=7.5, ha="right", va="center", color=INK2)
a.set_ylim(-0.7, 1.0); a.set_yticks([]); a.spines["left"].set_visible(False); a.set_xlabel("triple harm along the pair line ($0.667+s$)")
a.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0, 1.18), ncol=2); a.set_title("(a) one picture: chambers, model part, two walls", fontsize=9, pad=24)
a = axs[1]
rows = [("vertex attack (any record)", 0.873, 0.001 / 0.873), ("observer's block (any genuine law)", 0.128, 1 / 128),
        ("hidden channel (pairs kept)", 0.027, 0.001 / 0.027), ("observer on the pair line", 0.014, 1 / 14)]
y = np.arange(len(rows))[::-1]
a.barh(y, [r[2] for r in rows], color=[VIOLET, AQUA, ORANGE, BLUE], height=0.55)
for yy, r in zip(y, rows): a.text(r[2] + 0.002, yy, f"0.001 / {r[1]} = {r[2]:.4f}", va="center", fontsize=7.5, color=INK2)
a.set_yticks(y); a.set_yticklabels([r[0] for r in rows], fontsize=7.5); a.set_xlim(0, 0.11); a.set_xlabel("injected share needed to cross the wall")
a.set_title("(b) robustness = margin / capacity (sampling: $n_{1/2}=(z\\sigma_\\ell/0.001)^2=25{,}000$)", fontsize=9)
plt.tight_layout(); save(fig, "fig_capstone")

# ---- fig_descent and fig_filtration: Section 17 Step 14 ----
WDj = json.load(open(os.path.join(HERE, "worked_descent.json")))
fig, axs = plt.subplots(1, 4, figsize=(16.5, 4.1), gridspec_kw={"width_ratios": [1.05, 1.25, 1.0, 1.15]})
a = axs[0]; pos = {"AB": (0, 0), "BC": (2, 0), "AC": (1, 1.6)}
for (u, w), lab in ((("AB", "BC"), "B"), (("AB", "AC"), "A"), (("BC", "AC"), "C")):
    a.plot(*zip(pos[u], pos[w]), color=BLUE, lw=2.2, zorder=1); mx = np.mean([pos[u], pos[w]], axis=0)
    a.text(mx[0] + (0 if lab == "B" else (-0.42 if lab == "A" else 0.42)), mx[1] - (0.25 if lab == "B" else -0.05), f"{lab}\n(2 cells)", ha="center", va="center", fontsize=7.5, color=INK2)
for k, (x, y) in pos.items():
    a.scatter([x], [y], s=900, color="white", edgecolor=BLUE, lw=2, zorder=2); a.text(x, y, k, ha="center", va="center", fontsize=9, weight="bold", color=INK)
a.text(1, 0.55, "hollow:\nAB ∩ BC ∩ AC = ∅\n(1 cell: total mass)", ha="center", va="center", fontsize=7, color=GREY)
a.text(1, -0.75, "C$^0$: 4+4+4 = 12   C$^1$: 2+2+2 = 6   C$^2$: 1\nrank d$^0$ = 5, rank d$^1$ = 1\nH$^0$ = 7, H$^1$ = H$^2$ = 0", ha="center", va="center", fontsize=7.8, color=INK)
a.set_xlim(-0.6, 2.6); a.set_ylim(-1.15, 2.0); a.axis("off"); a.set_title("(a) the nerve of the pair cover and its\nČech complex of signed measures", fontsize=9)
a = axs[1]; d0 = np.array(WDj["cech"]["d0"]); from matplotlib.colors import ListedColormap
a.imshow(d0, cmap=ListedColormap([ORANGE, "#f4f4f2", BLUE]), vmin=-1, vmax=1, aspect="auto")
for x in (3.5, 7.5): a.axvline(x, color=INK2, lw=0.8)
for y in (1.5, 3.5): a.axhline(y, color=INK2, lw=0.8)
a.set_xticks([1.5, 5.5, 9.5]); a.set_xticklabels(["AB (00,01,10,11)", "BC", "AC"], fontsize=7.5)
a.set_yticks([0.5, 2.5, 4.5]); a.set_yticklabels(["on B", "on A", "on C"], fontsize=7.5)
a.set_title("(b) the differential d$^0$: C$^0$ (12) → C$^1$ (6)\nblue +1, orange −1: restriction differences", fontsize=9)
a.text(5.5, 7.4, "kernel 7 = compatible signed families = image of the 8 cells;\nthe 1 lost dimension is χ$_{ABC}$, the line $s$ of Step 1", ha="center", fontsize=7.3, color=INK2)
a = axs[2]; P2 = {"A": (0, 0), "B": (2, 0), "C": (1, 1.6)}; Mv = WDj["D2"]["M"]; sg = WDj["D2"]["sigma"]; av = WDj["D3"]["abs_v"]
for (u, w), mval in ((("A", "B"), Mv[0]), (("A", "C"), Mv[1]), (("B", "C"), Mv[2])):
    a.plot(*zip(P2[u], P2[w]), color=(AQUA if mval > 0 else CRIT), lw=2.4, zorder=1); mx = np.mean([P2[u], P2[w]], axis=0)
    a.text(mx[0] + (0 if u + w == "AB" else (-0.62 if u + w == "AC" else 0.62)), mx[1] + (-0.25 if u + w == "AB" else 0.0), f"$M$ = {mval:+.4f}", ha="center", fontsize=7.5, color=INK2)
for (k, (x, y)), sgn, a_ in zip(P2.items(), sg, av):
    a.scatter([x], [y], s=700, color=(BLUE if sgn > 0 else ORANGE), zorder=2); a.text(x, y, k, ha="center", va="center", fontsize=9, weight="bold", color="white")
    a.text(x, y - 0.42 if y < 1 else y + 0.38, f"$v$ = {sgn * a_:+.4f}", ha="center", fontsize=7.5, color=INK)
a.text(1, 0.5, "cycle sign\n(−)(+)(−) = +:\nclass 0 in H$^1$(Γ; Z/2);\nlabel switch $v$ → −$v$", ha="center", va="center", fontsize=6.8, color=INK2)
a.set_xlim(-1.0, 3.0); a.set_ylim(-0.75, 2.2); a.axis("off"); a.set_title("(c) the latent graph Γ: $M_{ij}=v_iv_j$\n(edge green +, red −; node blue +, orange −)", fontsize=9)
a = axs[3]; mb = np.array(WDj["labels"]["m_b"]); src = np.array(WDj["labels"]["src"]); cols = {"G1": BLUE, "G2": AQUA, "C": ORANGE}
xs = np.arange(len(mb))
for j in ("G1", "G2", "C"):
    k = src == j; a.scatter(xs[k], mb[k] * 1e3, color=cols[j], s=18, label=f"{j}: z = {WDj['labels']['z_source'][j]:+.2f}", zorder=3)
    a.plot([xs[k].min(), xs[k].max()], [mb[k].mean() * 1e3] * 2, color=cols[j], lw=1.6)
a.axhline(WDj["labels"]["m_T_truth"] * 1e3, color=INK2, ls=":", lw=1); a.text(0, WDj["labels"]["m_T_truth"] * 1e3, " truth", fontsize=7, color=INK2, va="bottom")
for x in range(16, 20): a.annotate("", xy=(x, mb[x] * 1e3), xytext=(x, mb[x] * 1e3 - WDj["labels"]["shift_per_adv_batch"] * 1e3), arrowprops=dict(arrowstyle="->", color=CRIT, lw=0.8))
a.set_xlabel("batch (8 per source)"); a.set_ylabel("triple moment $m_T$ of the batch ($\\times10^{-3}$)")
a.legend(frameon=False, fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, title=f"energy of δ$^0m_T$ = 24 × (between {WDj['labels']['between'] * 1e6:.2f} + within {WDj['labels']['within'] * 1e6:.1f}) × 10$^{{-6}}$", title_fontsize=6.4)
a.set_title("(d) labels: the batch cochain of one Study 13\nreplicate (A1b; red arrows: injected shift)", fontsize=9)
plt.tight_layout(); save(fig, "fig_descent")

fig = plt.figure(figsize=(15.5, 6.2)); gs = fig.add_gridspec(5, 2, width_ratios=[1.05, 1.0], wspace=0.25, hspace=1.6)
a = fig.add_subplot(gs[:, 0]); from matplotlib.patches import FancyBboxPatch as FB
stages = [("D$_0$ compatible families", "#eef3fb"), ("D$_1$ glue (a joint law)", "#e3ecf9"), ("D$_2$ sign class 0", "#d6e4f6"),
          ("D$_3$ modulus class 0 (tetrads)", "#c8dbf2"), ("D$_4$ walls leave a segment", "#b9d1ee"), ("D$_5$ triple coherence", "#a9c6ea")]
wits = ["liar cycle", "exactly one harm", "product law injected (m = 4)", "walls cross (t$_{lo}$ > t$_{hi}$)", "hidden channel (χ$_{ABC}$)"]
for k, (lab, col) in enumerate(stages):
    x0, y0 = 0.02 + 0.075 * k, 0.03 + 0.075 * k; w, h = 0.96 - 0.15 * k, 0.94 - 0.15 * k
    a.add_patch(FB((x0, y0), w, h, boxstyle="round,pad=0.005,rounding_size=0.02", facecolor=col, edgecolor=BLUE, lw=1.2))
    a.text(x0 + 0.012, y0 + h - 0.035, lab, fontsize=8, color=INK, va="top")
    if k < 5: a.text(x0 + w - 0.012, y0 + 0.02, "× " + wits[k], fontsize=7.3, color=CRIT, ha="right", va="bottom")
a.plot([0.5], [0.5], "o", color=INK, ms=7); a.text(0.5, 0.455, "the drug example", ha="center", fontsize=7.5)
a.set_xlim(0, 1); a.set_ylim(0, 1); a.axis("off"); a.set_title("(a) the descent filtration: the drug example lies in D$_5$; each ring holds\na witness that passes every earlier stage and fails there", fontsize=9)
rows = [("D$_1$: gluable fraction λ*", 1.0, WDj["D1"]["lambda_liar"], "liar", 1.0, "≥"),
        ("D$_2$: cycle product $M_{01}M_{02}M_{12}$ (×10$^{-4}$)", WDj["D2"]["product"] * 1e4, np.prod(WDj["D2"]["witness_M"]) * 1e4, "exactly one harm", 0.0, ">"),
        ("D$_3$: |log tetrad| (four-variable example)", 0.0, max(map(abs, WDj["D3_m4"]["log_tetrads_injected"])), "product injected", 0.0, "="),
        ("D$_4$: segment length $t_{hi}-t_{lo}$", WDj["D4"]["t"][1] - WDj["D4"]["t"][0], WDj["D4"]["witness"]["t_hi"] - WDj["D4"]["witness"]["t_lo"], "walls cross", 0.0, ">"),
        ("D$_5$: spread of $r_S$ over triples (four-variable example)", 0.0, max(WDj["D5_m4"]["r_injected"].values()) - min(WDj["D5_m4"]["r_injected"].values()), "χ$_{ABC}$ at 5%", 0.0, "="),
        ]
for i, (lab, tv, wv, wl, thr, rel) in enumerate(rows):
    a = fig.add_subplot(gs[i, 1]); vals = [tv] + ([wv] if wv is not None else [])
    lo_, hi_ = min(vals + ([thr] if thr is not None else [])), max(vals + ([thr] if thr is not None else [])); pad = 0.15 * (hi_ - lo_ + 1e-9)
    a.set_xlim(lo_ - pad, hi_ + pad); a.set_ylim(-1, 1); a.set_yticks([])
    if thr is not None: a.axvline(thr, color=CRIT, lw=1, ls="--")
    a.plot([tv], [0], "o", color=INK, ms=6); a.text(tv, 0.35, f"{'four-variable example' if 'four' in lab else 'drug example'} {tv:.4g}", fontsize=7, ha="center", color=INK)
    if wv is not None: a.plot([wv], [0], "X", color=CRIT, ms=7); a.text(wv, -0.85, f"{wl} {wv:.4g}", fontsize=7, ha="center", color=CRIT)
    a.set_title(lab + (f"   (stage requires {rel} {thr:g})" if thr is not None else "   (statistical: Theorem J tests it)"), fontsize=7.8, loc="left", pad=2)
    a.tick_params(labelsize=6.5); a.spines["left"].set_visible(False)
fig.text(0.53, 0.965, "(b) the value each stage computes: the drug example (black) and its witness (red ×)", fontsize=9)
save(fig, "fig_filtration")

# ---- fig_markov ----
MKj = json.load(open(os.path.join(HERE, "markov.json")))
fig, axs = plt.subplots(1, 4, figsize=(16.5, 4.0), gridspec_kw={"width_ratios": [1.15, 1.25, 1.0, 0.85]})
a = axs[0]; from matplotlib.patches import FancyBboxPatch as FB2
def box(x, y, t, w=0.9, h=0.45, col=BLUE):
    a.add_patch(FB2((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", facecolor="white", edgecolor=col, lw=1.6)); a.text(x, y, t, ha="center", va="center", fontsize=8.5)
box(1.5, 0.3, "w = (0.6, 0.4)", w=1.5); a.plot([1.5, 1.5], [0.53, 1.15], color=INK, lw=1.4); a.text(1.6, 0.85, "Z (two types)", fontsize=7.5, color=INK2)
a.plot([1.5], [1.2], "o", color=INK, ms=7); a.text(1.7, 1.18, "copy", fontsize=7.5, color=INK2)
for x, lab, k in ((0.3, "K$_A$", 0), (1.5, "K$_B$", 1), (2.7, "K$_C$", 2)):
    a.plot([1.5, x], [1.2, 1.75], color=INK, lw=1.4); box(x, 2.0, lab, w=0.6)
    Km = MKj["factorisation"]["K"][k]; a.text(x, 2.55, f"[{Km[0][0]:.2f} {Km[0][1]:.2f}]\n[{Km[1][0]:.2f} {Km[1][1]:.2f}]", ha="center", fontsize=6.8, color=INK2)
    a.plot([x, x], [2.23, 2.33], color=INK, lw=1.4); a.text(x, 2.98, "ABC"[k], ha="center", fontsize=9, weight="bold")
a.text(1.5, -0.25, "composite = the 8 cells of the drug law (exact);\nthe walls are other factorisations, same pairs", ha="center", fontsize=7.2, color=INK2)
a.set_xlim(-0.4, 3.4); a.set_ylim(-0.6, 3.2); a.axis("off"); a.set_title("(a) the two-type model as a string diagram:\nstate, copy, three kernels", fontsize=9)
a = axs[1]; df = MKj["definetti"]
a.plot([-0.013, 0.027], [0, 0], color=GREY, lw=5, solid_capstyle="butt"); a.plot([-0.00623, 0.00225], [0, 0], color=BLUE, lw=5, solid_capstyle="butt")
a.axvline(0.001, color=CRIT, ls="--", lw=1); a.text(0.0012, 0.95, "switch", color=CRIT, fontsize=7)
a.axvline(0.00225, color=BLUE, ls=":", lw=1); a.text(0.0025, 0.82, "detection wall", color=BLUE, fontsize=7)
for sv, wv in zip([0.0, df["s_batches"][3]], df["weights"]):
    a.plot([sv, sv], [0, wv], color=ORANGE, lw=2.2); a.plot([sv], [wv], "o", color=ORANGE, ms=6)
a.text(-0.0004, 0.79, "3 genuine\nbatches: 3/4", fontsize=7, color=INK2, va="bottom", ha="right"); a.text(df["s_batches"][3], 0.27, "  adversarial\n  batch: 1/4", fontsize=7, color=INK2, va="bottom")
a.plot([df["s_pool"]], [0], "D", color=INK, ms=7, zorder=5); a.annotate("pool μ(π) = P$_{0.05}$:\nlooks genuine, is tilted", xy=(df["s_pool"], 0.02), xytext=(-0.011, 0.45), fontsize=7.2, color=INK, arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
a.set_xlim(-0.014, 0.012); a.set_ylim(-0.08, 1.08); a.set_xlabel("position on the pair line $s$"); a.set_ylabel("mixing weight π")
a.set_title("(b) de Finetti: the batch laws' mixing measure π\nis not a Dirac; pooling (μ) forgets it", fontsize=9)
a = axs[2]; bw = MKj["blackwell"]; base_b = bw["bayes"]["none"]; base_m = bw["maxmin"]["none"]
groups = ["none", "noisy", "exact"]; xx = np.arange(3)
a.bar(xx - 0.22, [(bw["bayes"][g] - base_b) * 1e4 for g in groups], width=0.4, color=BLUE, label="Bayes (uniform prior)")
a.bar(xx + 0.22, [(bw["maxmin"][g] - base_m) * 1e4 for g in groups], width=0.4, color=AQUA, label="ex-ante maxmin (optimal rule)")
a.bar([1.0], [(bw["fixed_rule_noisy"] - base_m) * 1e4], width=0.25, color=CRIT, label="fixed rule: T if signal high")
a.axhline(0, color=INK2, lw=0.8); a.set_xticks(xx); a.set_xticklabels(["no probe", "noisy probe\n(garbling)", "exact probe"])
a.text(1.6, 1.6, "maxmin value is 0\nfor every probe here\n(a saddle point)", fontsize=7, color=AQUA, ha="center")
a.set_ylabel("value of the probe (×10$^{-4}$ utility)"); a.legend(frameon=False, fontsize=6.8, loc="upper left")
a.set_title("(c) Blackwell: monotone for Bayes and for\noptimal maxmin; only a fixed rule goes negative", fontsize=9)
a = axs[3]; fi = MKj["fisher"]; wins = ["AB", "BC", "AC", "ABC"]
a.bar(range(4), [fi[w] for w in wins], color=[GREY, GREY, GREY, ORANGE]); a.set_xticks(range(4)); a.set_xticklabels(wins)
for i, w in enumerate(wins): a.text(i, fi[w] + 3, f"{fi[w]:.1f}", ha="center", fontsize=7.5)
a.set_ylabel("Fisher information about $s$ per record"); a.set_ylim(0, 200)
a.set_title("(d) sufficiency: the pairs carry\nnothing about the hidden coordinate", fontsize=9)
plt.tight_layout(); save(fig, "fig_markov")

print("figures done")
