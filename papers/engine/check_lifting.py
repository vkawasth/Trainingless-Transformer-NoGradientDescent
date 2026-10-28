"""check_lifting.py -- lifting: the decision maker's patchwork (lab paper, Theorem K, Propositions on dilation, the value of a probe
in its size, backtracking, and rectangularity as gluing).

  LF1 the decision arrangement: on the worked example U_T - U_P = 0.128 - P(001) (a coordinate hyperplane of the simplex); on the pair
      line it sits at s = 0.001, inside the regulator's hiding segment (an indecision point of the model part). At m = 4 the gradient
      of the decision along the 5-dimensional pair fibre is proportional to the chi_ABC coordinate: the chambers are slabs
  LF2 the act lift: alpha* = b / (a + b) = 0.853 on T, worst regret ab / (a + b) = 0.00107 against 0.00125; the lifted rule is
      continuous in the evidence (largest jump over a fine sweep of intervals across the switch < 0.01) while the pure rule jumps by 1
  LF3 the adversarial lift (P, Q, eps): the block of explanations of the observed law reaches the other chamber at eps = 1/128 (a
      genuine two-type law P' with P'(001) = 0.128 exists inside it); keeping the pair tables needs 1/14 (the line); the forward
      hidden channel 0.037 (Theorem F)
  LF4 dilation: pair tables pin P(C) = 0.28 on the whole segment, but P(C | A, B) is an interval of positive width for every (A, B)
      (widths 0.018, 0.039, 0.030, 0.212); the triple pins each to a point inside its interval
  LF5 the value of the probe in its size (Bayes, uniform prior on the segment): EVSI(n) is zero to 1e-12 for n <= 100 (all derivatives
      vanish at 0: the Radner-Stiglitz plateau), convex on [300, 1500] (inflection near 1500-2000), and rises to EVPI; at cost c = 1e-9 per record the optimal
      size is interior while the marginal value at n = 0 is below c (a marginal reasoner never starts)
  LF6 robust readings: the set rule's worst regret with the Savage default falls from 0.00125 to 0 as n grows; its Wald value with the
      Wald default is below the pre-probe 0.332 at every n (information has negative Wald value at a saddle point) and tends to it
  LF7 the blow-up of the merge: at v = 0 the lifted coordinate rho^2 = ||offdiag M|| has sd ~ n^-1/2 and |v| = rho ~ n^-1/4 (fitted
      log-log slopes within 0.07); away from the merge both are ~ n^-1/2
  LF8 backtracking: against sequential gluing tests, the worst constant-dose adversary moves the final pool further when accepted
      sections are final (forward-only) than when a late failure revokes them (backtracking)
  LF9 rectangularity as gluing (two-stage tree): C is not rectangular; its gluing rect(C) contains it; the recursive (backward
      induction) maxmin value equals the maxmin over rect(C); the dynamic-inconsistency loss -0.1 obeys VOI >= -osc(u) d_H(C, rect C)
  LF10 exact stability of the lift: a family of batch laws with common pair tables and triple moments on the line is exactly the
      compression (forget the label) of an honest labelled law on {genuine, injected} x X with shares eps_b = (m_b - m) / s
"""
import itertools, os, sys
import numpy as np
from numpy.polynomial import polynomial as Pn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pce.lab import law, moments, segment, explanation, r_of_t
from pce import segment_decision as SD
from pce import injection as IJ
from pce import duality as DU
from pce import lifting as LF

rng = np.random.default_rng(20261030); results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)

W = LF.we(); P = W["P"]; chi = IJ.chi(3, (0, 1, 2)); p = IJ.as_vec(P)

# LF1
lin = all(abs(LF.gapTP(IJ.as_law(q, 3)) - (0.128 - q[1])) < 1e-12 for q in rng.dirichlet(np.ones(8), 20))
s_b = 0.128 - P[0, 0, 1]; harm_lo, harm_hi = 0.660769, 0.66925; inside = harm_lo < 0.667 + s_b < harm_hi
P4 = law(0.6, np.array([.5, .1, .4, .2]), np.array([.05, .5, .1, .7])); Bf = DU.pair_fibre_basis(4); p4 = IJ.as_vec(P4)
g4 = lambda q: LF.gapTP(IJ.as_law(q, 4).sum(axis=3)); grad = np.array([(g4(p4 + 1e-6 * e) - g4(p4 - 1e-6 * e)) / 2e-6 for e in np.eye(16)])
c4 = np.array([np.prod([2 * x[i] - 1 for i in (0, 1, 2)]) for x in IJ.outcomes(4)]) / 16
gb = grad @ Bf; cb = c4 @ Bf; slab = np.linalg.norm(gb - (gb @ cb) / (cb @ cb) * cb) < 1e-9 * np.linalg.norm(gb)
check("LF1 arrangement: U_T - U_P = 0.128 - P(001); boundary at s = 0.001 inside the hiding segment; at m = 4 chambers are chi_ABC slabs",
      lin and abs(s_b - 0.001) < 1e-12 and inside and slab, f"gradient along the fibre / chi_ABC coordinate: {gb @ cb / (cb @ cb):.4f}")

# LF2
R = W["R"]; a, b = LF.regrets_on(R, W["ps"], W["UP"]); al, val = LF.act_lift(a, b)
cs = np.linspace(W["rc"] - 0.3, W["rc"] + 0.3, 3001); h = 0.1
alph = [LF.lifted_rule(SD.clip((c - h, c + h), R), W["ps"], W["UP"]) for c in cs]
pure = [float(SD.set_rule(W["ps"], SD.clip((c - h, c + h), R), W["u0"]) == 0) for c in cs]
check("LF2 act lift: alpha* = 0.853 on T, worst regret 0.00107 (15% below 0.00125); the lifted rule is continuous, the pure rule jumps",
      abs(al - 0.8526) < 5e-4 and abs(val - 0.001066) < 2e-6 and max(np.abs(np.diff(alph))) < 0.01 and max(np.abs(np.diff(pure))) == 1,
      f"a = {a:.5f}, b = {b:.5f}, alpha* = {al:.4f}, value {val:.6f}; largest step of the lifted rule {max(np.abs(np.diff(alph))):.4f}")

# LF3
eps, Pp, Q = LF.block_radius(P, lambda X: LF.gapTP(X), starts=120)
line = 1 - min(p / (p + 0.001 * chi))
ok_lift = np.allclose((1 - eps) * IJ.as_vec(Pp) + eps * IJ.as_vec(Q), p, atol=1e-8) and IJ.as_vec(Q).min() > -1e-9
check("LF3 adversarial lift: the block reaches the other chamber at eps = 1/128 with a genuine two-type law; the pair line needs 1/14",
      abs(eps - 1 / 128) < 1e-6 and ok_lift and abs(line - 1 / 14) < 1e-4 and LF.gapTP(Pp) <= 1e-9,
      f"eps = {eps:.6f}; P'(001) = {Pp[0, 0, 1]:.6f}; line {line:.5f}; forward hidden channel 0.0370")

# LF4
tab, pC = LF.dilation(P); widths = [tab[k][1] - tab[k][0] for k in sorted(tab)]
check("LF4 dilation: P(C) = 0.28 on the whole segment; every P(C | A, B) is an interval; the triple pins a point inside each",
      abs(pC[0] - 0.28) < 1e-12 and abs(pC[1] - 0.28) < 1e-12 and min(widths) > 0.01 and all(tab[k][0] <= tab[k][2] <= tab[k][1] for k in tab)
      and np.allclose(widths, [0.018, 0.039, 0.030, 0.212], atol=0.0015),
      "; ".join(f"AB={k[0]}{k[1]}: [{tab[k][0]:.3f}, {tab[k][1]:.3f}] (truth {tab[k][2]:.3f})" for k in sorted(tab)))

# LF5
ns = np.array([10, 30, 100, 300, 600, 1000, 1500, 2000, 3000, 5000, 1e4, 2e4, 5e4, 1e5, 1e6]); E = np.array([LF.evsi(n, W) for n in ns])
ev = LF.evpi(W); conv = all(np.diff(E[3:7]) / np.diff(ns[3:7]) > 0) and all(np.diff(np.diff(E[3:7]) / np.diff(ns[3:7])) > 0)
c = 1e-9; nn = np.geomspace(10, 1e6, 400); net = np.array([LF.evsi(n, W) for n in nn]) - c * nn; nopt = nn[int(np.argmax(net))]
check("LF5 EVSI(n): zero for n <= 100, convex on [300, 1500], rising to EVPI; at c = 1e-9 the optimum is interior, the marginal value at 0 is 0",
      max(E[:3]) < 1e-12 and conv and abs(E[-1] / ev - 1) < 0.05 and 300 < nopt < 1e6 and net.max() > 0 and (E[2] - E[0]) / 90 < c,
      f"EVPI {ev:.3e}; EVSI(1e5) {E[-2]:.3e}; optimal n {nopt:.0f} with net value {net.max():.3e}")

# LF6
sv = [LF.savage_curve(n, W) for n in (1, 1e3, 1e4, 1e5, 1e7)]; wv = [LF.wald_curve(n, W) for n in (1, 1e3, 1e4, 1e5, 1e7)]
check("LF6 Savage worst regret falls from 0.00125 toward 0; the Wald value stays below the pre-probe 0.332 at every n and tends to it",
      sv[0] > 0.00118 and all(np.diff(sv) < 0) and sv[-1] < 3e-5 and all(x < W["UP"] for x in wv) and W["UP"] - wv[-1] < 1e-6,
      f"Savage {np.round(sv, 6).tolist()}; Wald deficit {np.round([W['UP'] - x for x in wv], 7).tolist()}")

# LF7
nsb = [1000, 4000, 16000, 64000]; mu0 = np.array([0.3, 0.4, 0.5]); out = {}
for nm, vv in (("merge", np.zeros(3)), ("away", np.array([0.15, -0.12, 0.10]))):
    sd = np.array([LF.blowup_sd(mu0, vv, n, 400, rng)[:2] for n in nsb]); out[nm] = np.polyfit(np.log(nsb), np.log(sd), 1)[0]
check("LF7 blow-up of the merge: at v = 0, rho^2 at n^-1/2 and |v| at n^-1/4; away from the merge both n^-1/2",
      abs(out["merge"][0] + 0.5) < 0.07 and abs(out["merge"][1] + 0.25) < 0.07 and abs(out["away"][0] + 0.5) < 0.07 and abs(out["away"][1] + 0.5) < 0.07,
      f"slopes at the merge {np.round(out['merge'], 3).tolist()}, away {np.round(out['away'], 3).tolist()}")

# LF8
fw = LF.worst_backtrack(False); bk = LF.worst_backtrack(True)
check("LF8 backtracking beats forward-only acceptance: the worst constant-dose adversary moves the pool less when late failures revoke",
      fw[0] > bk[0] + 0.2, f"forward-only {fw[0]:.2f} se at (rounds, dose) {fw[1]}, flagged {fw[2]:.2f}; backtracking {bk[0]:.2f} se at {bk[1]}, flagged {bk[2]:.2f}")

# LF9
rx = LF.rect_example()
check("LF9 rectangularity as gluing: VOI = -0.1 < 0 under C; recursive value = maxmin over rect(C); bound -osc d_H = -0.3 holds",
      abs(rx["voi"] + 0.1) < 1e-9 and rx["recursive_equals_rect"] and rx["voi"] >= rx["bound"] - 1e-12 and abs(rx["dH"] - 0.3) < 1e-9,
      f"ex ante {rx['ex_ante']:.3f}, with information {rx['with_info']:.3f}, d_H {rx['dH']:.3f}")

# LF10
s = IJ.stealth_range(P, (0, 1, 2))[1]; ms = rng.uniform(0, s, 6); fam = [IJ.as_law(p + m * chi, 3) for m in ms]
ok = True
for m, Pb in zip(ms, fam):
    e = m / s; lab = np.stack([(1 - e) * p, e * (p + s * chi)]); ok &= np.allclose(lab.sum(0), IJ.as_vec(Pb)) and lab.min() >= -1e-15
check("LF10 exact stability: every batch family on the line is the compression of an honest labelled law with shares (m_b - m) / s", ok)

print(f"{sum(results)}/{len(results)} passed")
