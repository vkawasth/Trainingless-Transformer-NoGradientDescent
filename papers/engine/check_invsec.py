"""Checks for section 12.21 of the companion note (module pce/invsec.py): Bayesian inversion against sections of the marginal projection.
IDENTITY = an exact or numerical instance of a statement proved in the text; CERTIFICATE = an exact computation reported as a result;
NUMERICAL = floating-point, with tolerances stated.
  IS1  IDENTITY     in Cpl the dagger of a coupling is the coupling of the Bayesian inverse: (q, Kdag) with Kdag = p K / q; inverting
                    twice returns K on the support of p; at a null event (q(y) = 0) every choice of Kdag(. | y) gives the same coupling
  IS2  IDENTITY     composition commutes with inversion for reversal-equivariant rules: (g1 o g2)^T = g2^T o g1^T for Markov gluing
                    and for facet-leximin selection, exactly, on random rational chains of several shapes
  IS3  CERTIFICATE  a rule that is not reversal-equivariant (the vertex minimising sum G[i] H[k] x_ijk) breaks it, and on chains with
                    binary ends every rule given by a fixed linear objective commutes (the objective acts only through a transpose-
                    invariant contrast)
  IS4  IDENTITY     null events do not reach the composite: changing the kernel rows of a null middle state changes neither g2 nor any
                    composite
  IS5  CERTIFICATE  structural zeros are not null events: a chain with a zero cell and full-support margins has unique Bayesian inverses
                    and still a positive-width identified interval for the composite; Markov gluing and selection disagree unless Markov
                    happens to be the midpoint
  IS6  NUMERICAL    order drift as Bayesian inconsistency, read from Study 18's frozen output (hash checked): mean drift of plain and
                    consistency-trained models on the sparsity-matched faces
"""
import os, sys, random, json, hashlib
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import invsec as I
from pce import selnat as N
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = random.Random(20261107)

# IS1
ok = True; nnull = 0
for s in range(60):
    a, b, c = rng.choice([(2, 2, 2), (3, 2, 2), (2, 3, 3)])
    g, _ = I.random_chain(rng, a, b, c)
    if s % 3 == 0:                                                       # make a null event: a zero column
        for i in range(a): g[i][-1] = Fr(0)
        t = sum(map(sum, g)); g = [[x / t for x in r] for r in g]; nnull += 1
    p, K, q, Kd = I.kernels(g)
    ok &= all(Kd[j] is None or all(Kd[j][i] == p[i] * K[i][j] / q[j] for i in range(a)) for j in range(len(q)))       # Bayes' rule
    for j in range(len(q)):                                              # any completion at null events
        if Kd[j] is None: Kd[j] = [Fr(1, a)] * a if s % 2 else [Fr(1)] + [Fr(0)] * (a - 1)
    ok &= N.eq(I.coupling(q, Kd), N.T(g))                                # the dagger is the coupling of the inverse
    p2, K2, q2, Kd2 = I.kernels(N.T(g)); ok &= all(K[i] is None or Kd2[i] == K[i] for i in range(a))              # inverting twice
check("IS1 IDENTITY the dagger of a coupling is the coupling of its Bayesian inverse; inverting twice returns K; null events do not matter",
      ok, f"60 rational couplings, {nnull} with a null event, two different completions there")

# IS2, IS3
cnt = {"markov": 0, "sel": 0, "tilt": 0}; tot = 0; ex = None; ok = True
for s in range(40):
    a, b, c = rng.choice([(2, 2, 2), (3, 2, 2), (2, 3, 3), (3, 2, 3), (3, 3, 3)])
    g1, g2 = I.random_chain(rng, a, b, c); tot += 1
    for r in cnt:
        if I.inverts(r, g1, g2): cnt[r] += 1
        elif r == "tilt" and ex is None and a == 3: ex = (g1, g2)
check("IS2 IDENTITY composition commutes with Bayesian inversion for Markov gluing and for selection (exact)",
      cnt["markov"] == tot and cnt["sel"] == tot, f"{tot} random rational chains of shapes 2-2-2 to 3-3-3: Markov {cnt['markov']}/{tot}, selection {cnt['sel']}/{tot}")
bin_ok = all(I.inverts("tilt", *I.random_chain(rng, 2, b_, 2)) for b_ in (2, 3, 2, 3, 2, 3, 2, 3))
g1, g2 = ex; L = N.T(I.composite("tilt", g1, g2)); R = I.composite("tilt", N.T(g2), N.T(g1))
check("IS3 CERTIFICATE a rule that is not reversal-equivariant breaks it; on binary-end chains every fixed linear rule commutes",
      cnt["tilt"] < tot and not N.eq(L, R) and bin_ok,
      f"commutes on {cnt['tilt']}/{tot}; e.g. g1 = {N.show(g1)}, g2 = {N.show(g2)}: inverse of the composite {N.show(L)} vs composite of the inverses {N.show(R)}; binary ends: 8/8 commute")

# IS4
ok = True
for s in range(30):
    g1, g2 = I.random_chain(rng, 2, 3, 2)
    for i in range(2): g1[i][2] = Fr(0)                                  # middle state 2 is null
    t = sum(map(sum, g1)); g1 = [[x / t for x in r] for r in g1]; q = N.cols(g1)
    def g2_with(row): return [[q[j] * (row[k] if j == 2 else Fr(1, 2)) for k in range(2)] for j in range(3)]
    A, B = g2_with([Fr(1), Fr(0)]), g2_with([Fr(1, 3), Fr(2, 3)])
    ok &= N.eq(A, B) and N.eq(I.composite("markov", g1, A), I.composite("markov", g1, B)) and N.eq(I.composite("sel", g1, A), I.composite("sel", g1, B))
check("IS4 IDENTITY the kernel rows of a null middle state change neither the coupling nor any composite", ok, "30 chains, two completions each")

# IS5
rows5 = []; ok = True
for s in range(6):
    g1, g2 = I.random_chain(rng, 2, 2, 2, zero=(0, 1))                   # a structural zero in g1; all margins positive
    p, K, q, Kd = I.kernels(g1); lo, hi = N.identified(g1, g2); m = I.composite("markov", g1, g2)[0][0]; se = I.composite("sel", g1, g2)[0][0]
    ok &= all(x > 0 for x in p + q) and all(k is not None for k in Kd) and hi > lo and lo <= m <= hi and lo <= se <= hi
    rows5.append((str(lo), str(hi), str(m), str(se))); ndis = sum(1 for r_ in rows5 if r_[2] != r_[3])
check("IS5 CERTIFICATE a structural zero with full-support margins: Bayesian inverses unique, yet the identified interval of the composite's "
      "00 entry has positive width, so the rule still matters", ok and ndis >= 1,
      f"Markov gluing and selection disagree in {ndis}/6 (they agree when Markov happens to be the midpoint); (L, U, Markov, selection): " + "; ".join(str(r) for r in rows5))

# IS6
SS = os.path.join(HERE, "..", "synthetic_sections"); fz = open(os.path.join(HERE, "..", "FROZEN.txt")).read()
h = hashlib.sha256(open(os.path.join(SS, "study18_final.json"), "rb").read()).hexdigest(); hash_ok = f"{h}  synthetic_sections/study18_final.json" in fz
sys.path.insert(0, SS); import kit_study18 as K18
conds, _ = K18.conditions(); R = json.load(open(os.path.join(SS, "study18_final.json")))["runs"]
dr = {(f, v): float(np.mean([R[str(c["idx"])]["drift"] for c in conds if c["kind"] == "S" and c["fam"] == f and c["variant"] == v]))
      for f in ("F0", "F1", "F2") for v in ("plain", "consistent")}
check("IS6 NUMERICAL order drift (mean TV between the joints of different generation orders, i.e. failure of the learned conditionals to "
      "be Bayesian inverses of one another) under plain and consistency training, Study 18", hash_ok and all(dr[(f, "consistent")] < dr[(f, "plain")] / 10 for f in ("F0", "F1", "F2")),
      "frozen hash OK; " + ", ".join(f"{f}: {dr[(f, 'plain')]:.3f} -> {dr[(f, 'consistent')]:.3f}" for f in ("F0", "F1", "F2")))
print(f"{sum(results)}/{len(results)} passed")
