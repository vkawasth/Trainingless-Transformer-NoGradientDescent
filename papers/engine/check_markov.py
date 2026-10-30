"""check_markov.py -- Markov-category readings with fully worked small examples (lab paper, Section "Markov categories: factorisation,
sufficiency, de Finetti and Blackwell"). Writes lab/markov.json for the figure.

  MK1 factorisation: the drug law is the composite I --w--> Z --copy--> Z^3 --(K_A x K_B x K_C)--> X^3 with w = (0.6, 0.4) and the
      three 2x2 kernels; every cell agrees; the two wall explanations of the hiding segment are different factorisations with the same
      pair tables and different triple
  MK2 sufficiency on the line: for the family P + s chi_ABC the pair windows carry Fisher information 0 about s (they are ancillary for it)
      and the triple window carries sum_x 1/P(x) = 177.41; the efficient sd per record is 1/sqrt(177.41) = 0.0751, below the moment
      estimator's 0.0797 (sigma_m)
  MK3 de Finetti for batches: four batches with shares (0, 0, 0, 0.2) of the hidden channel have mixing measure (3/4) delta_P +
      (1/4) delta_{P_0.2} (not a Dirac); the pool mu(pi) is exactly the whole-pool injection P_0.05, inside the stealth window, so every
      pooling rule sees Theorem F at share 0.05; records from a genuine and the adversarial batch are not exchangeable (defect 0.0028), records from two genuine
      batches are (defect 0)
  MK4 the Dirac test: Var_pi(m_T) = pi (1 - pi) (Delta m)^2 = (3/16)(0.0054)^2 = 5.47e-6 and is 0 iff pi is a Dirac; it equals the
      population version of the between-batch energy that Theorem J's tests estimate
  MK5 Blackwell on a three-point slice (walls and truth): exact >= noisy >= none, the noisy experiment a garbling of the exact one; Bayes
      values (uniform prior) 0.334327, 0.334327, 0.334744 (the noisy probe is on a plateau at this prior); ex-ante maxmin values all
      0.332 (the saddle point: value of information 0); a fixed rule "T if the signal is high" has worst case 0.331875 < 0.332
  MK6 Blackwell monotonicity holds for both Bayes and ex-ante maxmin on 300 random decision problems and random garblings; negative value
      arises only from a fixed (non-optimal) rule or from prior-by-prior updating (Proposition "the price of not gluing")
"""
import json, os, sys
import numpy as np
from numpy.polynomial import polynomial as Pn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce.lab import law, moments, segment, explanation
from pce import segment_decision as SD
from pce import injection as IJ
from pce import sources as SR
from pce import lifting as LF
from pce import markov as MK

rng = np.random.default_rng(20261029); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
r6 = lambda x: float(np.round(x, 6))

W = LF.we(); P = W["P"]

# MK1
Pf, Wz, K = MK.factorise(0.6, [.5, .1, .4], [.05, .5, .1])
mu, M, mS = moments(P); v = SD.v_closed(M); lo, hi = segment(mu, v); walls = []
for t in (lo * (1 + 1e-9), hi * (1 - 1e-9)):
    w_, a1, a2 = explanation(mu, v, t); Lw, _, _ = MK.factorise(w_, a1, a2); walls.append((w_, a1, a2, Lw))
same_pairs = all(np.allclose(moments(L)[1], M, atol=1e-10) for *_, L in walls); diff_triple = abs(moments(walls[0][3])[2]((0, 1, 2)) - moments(walls[1][3])[2]((0, 1, 2))) > 1e-4
OUT["factorisation"] = {"w": Wz.tolist(), "K": [k.tolist() for k in K], "cells": {"".join(map(str, x)): r6(Pf[x]) for x in MK.XS},
                        "walls": [{"w": r6(a), "a1": [r6(z) for z in b], "a2": [r6(z) for z in c], "triple_harm": r6(1 - L[0, 0, 0])} for a, b, c, L in walls]}
check("MK1 the drug law is the composite of w, copy and three kernels; the two walls are other factorisations with the same pairs and a different triple",
      np.abs(Pf - P).max() < 1e-15 and same_pairs and diff_triple, f"wall triple harms {[w_['triple_harm'] for w_ in OUT['factorisation']['walls']]}")

# MK2
fi = {"".join("ABC"[i] for i in w): MK.fisher_on_line(P, w) for w in [(0, 1), (1, 2), (0, 2), (0, 1, 2)]}
sm = SR.sigma_m(P); OUT["fisher"] = {k: r6(x) for k, x in fi.items()}; OUT["fisher"]["sd_efficient"] = r6(1 / np.sqrt(fi["ABC"])); OUT["fisher"]["sigma_m"] = r6(sm)
check("MK2 Fisher information about s: 0 in every pair window, sum 1/P(x) = 177.41 in the triple; efficient sd 0.0751 < sigma_m 0.0797",
      max(fi["AB"], fi["BC"], fi["AC"]) < 1e-12 and abs(fi["ABC"] - (1 / IJ.as_vec(P)).sum()) < 1e-9 and abs(fi["ABC"] - 177.408) < 1e-3 and 1 / np.sqrt(fi["ABC"]) < sm,
      f"triple {fi['ABC']:.3f}")

# MK3
Q = SR.hidden_Q(P, 0.027); atoms, wts, pool, dirac = MK.batch_mixing(P, Q, [0, 0, 0, 0.2])
d_ga = MK.exch_defect(atoms[0], atoms[1]); d_gg = MK.exch_defect(P, P)
rp = SD.from_tables(SD.exact_tables(pool))[2]; rw = SD.from_tables(SD.exact_tables(IJ.mix(P, Q, 0.05)))[2]
OUT["definetti"] = {"weights": wts.tolist(), "dirac": dirac, "pool_equals_P_0.05": bool(np.abs(pool - IJ.mix(P, Q, 0.05)).max() < 1e-15),
                    "rhat_pool": r6(rp), "rhat_whole_pool": r6(rw), "defect_genuine_adversarial": r6(d_ga), "defect_genuine_genuine": r6(d_gg),
                    "s_batches": [0.0, 0.0, 0.0, 0.2 * 0.027], "s_pool": 0.05 * 0.027}
check("MK3 de Finetti: mixing measure (3/4, 1/4), not a Dirac; mu(pi) = P_0.05 exactly (same rhat as whole-pool injection, inside the stealth window); exchangeability fails only across the adversarial batch",
      not dirac and np.allclose(wts, [0.75, 0.25]) and OUT["definetti"]["pool_equals_P_0.05"] and abs(rp - rw) < 1e-12 and d_ga > 1e-3 and d_gg == 0,
      f"rhat {rp:.4f}; defect {d_ga:.4f}")

# MK4
dm = 0.2 * 0.027; var = 0.75 * 0.25 * dm ** 2
mts = [SR.mT_stats(a.sum(axis=())) [0] if False else moments(a)[2]((0, 1, 2)) for a in atoms]; var2 = float(np.sum(wts * (np.array(mts) - np.sum(wts * np.array(mts))) ** 2))
OUT["dirac_test"] = {"delta_m": dm, "var_pi": var, "var_from_atoms": var2}
check("MK4 the Dirac test: Var_pi(m_T) = (3/16)(0.0054)^2 = 5.47e-6, zero iff pi is a Dirac", abs(var - var2) < 1e-15 and abs(var - 5.4675e-6) < 1e-9, f"{var:.4e}")

# MK5
th = [W["R"][0], W["r0"], W["R"][1]]; U = np.array([[Pn.polyval(r, W["ps"][0]) for r in th], [W["UP"]] * 3])
Ex = np.eye(3); En = np.array([[.9, .1], [.6, .4], [.1, .9]]); E0 = np.ones((3, 1)); G = En; G0 = np.ones((2, 1))
bay = {k: MK.bayes_value(E, U, [1 / 3] * 3) for k, E in (("none", E0), ("noisy", En), ("exact", Ex))}
mm = {k: MK.maxmin_value(E, U)[0] for k, E in (("none", E0), ("noisy", En), ("exact", Ex))}
fixed = MK.rule_value(En, U, [1, 0])
OUT["blackwell"] = {"states_r": [r6(x) for x in th], "U": U.tolist(), "E_noisy": En.tolist(), "bayes": {k: r6(x) for k, x in bay.items()},
                    "maxmin": {k: r6(x) for k, x in mm.items()}, "fixed_rule_noisy": r6(fixed)}
check("MK5 Blackwell on the slice: noisy = exact G, none = noisy G0; Bayes 0.334327 <= 0.334327 <= 0.334744 (noisy on a plateau); maxmin all 0.332; fixed rule 0.331875",
      np.allclose(Ex @ G, En) and np.allclose(En @ G0, E0) and bay["none"] <= bay["noisy"] + 1e-15 <= bay["exact"] + 1e-15 and abs(bay["noisy"] - bay["none"]) < 1e-12
      and all(abs(x - 0.332) < 1e-9 for x in mm.values()) and abs(fixed - 0.331875) < 1e-9, f"Bayes {bay}; maxmin {mm}")

# MK6
ok = True; nstrict = 0
for _ in range(300):
    nT, nS, nA = 3, 3, int(rng.integers(2, 4)); E1 = rng.dirichlet(np.ones(nS), nT); Gm = rng.dirichlet(np.ones(2), nS); E2 = E1 @ Gm
    Ur = rng.normal(size=(nA, nT)); pr = rng.dirichlet(np.ones(nT))
    b1, b2 = MK.bayes_value(E1, Ur, pr), MK.bayes_value(E2, Ur, pr); m1, m2 = MK.maxmin_value(E1, Ur)[0], MK.maxmin_value(E2, Ur)[0]
    ok &= b1 >= b2 - 1e-12 and m1 >= m2 - 1e-9; nstrict += (m1 > m2 + 1e-9)
check("MK6 Blackwell monotonicity for Bayes and for ex-ante maxmin over 300 random problems and garblings", ok, f"maxmin strictly better in {nstrict}")

json.dump(OUT, open(os.path.join(HERE, "..", "lab", "markov.json"), "w"), indent=1, default=float)
print(f"{sum(results)}/{len(results)} passed")
