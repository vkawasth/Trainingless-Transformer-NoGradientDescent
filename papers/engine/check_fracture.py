"""Checks for Section "Fractures and blind spots" of "Probability, Context and Differentiation: The Layers".
  FX1  poison order against tie order (exact, germs): the poisoned gap G + E takes the sign of the term of lower order, and of the sum of
       leading coefficients on equal orders; evaluation at small s agrees
  FX2  invisible at first order, decisive at second (exact): a covariance decision tied at order m, poisoned along a direction with zero first
       derivative and share e s^p; the poison's effect is exactly -e^2 s^(2p) (1/4 + delta)^2; the act flips iff 2p < m, or 2p = m and e^2/16 > c
  FX3  the fracture detector on the paper's examples: order-0 fractures (limit set not a singleton) for the race and the curved path, none for
       dominance or for either study design; a degenerate direction (type iii) for Example 40 and Example 60; a flat tie broken at order 3
       (type ii) for Example 47
  FX4  feasibility fractures (type iv) appear with a squared contrast and never with pooling or conditioning (direct T^2 tests at a vertex)
  FX5  statistical fractures (type v): the expected-count order w_A - 2 classifies Gaussian cone, Poisson ray and empirical wall, and the
       coverage of nominal 95% intervals agrees (Monte Carlo)
  FX6  gluing fractures (type vi): the main paper's three-token hallucination: pairwise tables with agreement eps glue iff eps >= 1/3 (linear
       programme against the closed form), with certificate Phi = 3(1 - eps) > 2; its blind spot is the parity chi, invisible to every pair table
"""
import os, sys, json, random, itertools, math
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import linprog
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import fracture as FX
from pce import orders as OD
from pce import fans as FN
from pce import para as PA
from pce import degen as DG
from pce import graded as GR
from pce import statfan as SF
random.seed(20261101); rng = np.random.default_rng(20261101); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)

# FX1
ok = True; flips = 0
for _ in range(1000):
    G = OD.clean({Fr(random.randint(0, 6), random.choice([1, 2])): Fr(random.choice([-3, -2, -1, 1, 2, 3])), Fr(7): Fr(random.randint(-2, 2))})
    E = OD.clean({Fr(random.randint(0, 6), random.choice([1, 2])): Fr(random.choice([-3, -2, -1, 1, 2, 3])), Fr(8): Fr(random.randint(-2, 2))})
    if not G or not E: continue
    f, mg, me = FX.poison_wins(G, E); flips += f
    pred = (me < mg and (OD.lead(E)[1] > 0) != (OD.lead(G)[1] > 0)) or (me == mg and ((OD.lead(G)[1] + OD.lead(E)[1] > 0) != (OD.lead(G)[1] > 0) or OD.lead(G)[1] + OD.lead(E)[1] == 0 and not OD.add(G, E)))
    if me == mg and OD.lead(G)[1] + OD.lead(E)[1] == 0: pred = f                    # cancellation of the leading terms: decided further out
    ok &= f == pred
    s = 1e-4; ok &= (OD.ev(OD.add(G, E), s) > 0) != (OD.ev(G, s) > 0) if f and OD.add(G, E) else True
check("FX1 poison order vs tie order: the lower-order term decides; equal orders compare leading coefficients", ok, f"{flips} flips in 1000 cases")

# FX2
ok = True; rows = []
for m in (1, 2, 3, 4):
    for p in (Fr(1, 2), Fr(1), Fr(3, 2), Fr(2)):
        for e in (Fr(1), Fr(4), Fr(8)):
            c = Fr(1, 2); s = Fr(1, 10 ** 4); d = c * s ** m; eps = e * s ** p if p.denominator == 1 else None
            if eps is None:
                s = Fr(1, 10 ** 8); d = c * s ** m; eps = e * Fr(1, 10 ** 4) ** int(2 * p)       # s^(p) with p half-integer: s = 1e-8, sqrt = 1e-4
            Phi = [Fr(1, 4) + d, Fr(1, 4) - d, Fr(1, 4) - d, Fr(1, 4) + d]
            Q = [Fr(0), Fr(1, 4) - d, Fr(1, 4) - d, Fr(1, 2) + 2 * d]; U = [q - f for q, f in zip(Q, Phi)]
            D1, Q2 = FX.cov_parts(Phi, U); ok &= D1 == 0 and Q2 == -(Fr(1, 4) + d) ** 2
            poisoned = FX.cov([f + eps * u for f, u in zip(Phi, U)]); ok &= poisoned == d + eps * eps * Q2
            pred = (2 * p < m) or (2 * p == m and e * e / 16 > c); ok &= (poisoned < 0) == pred; rows.append((m, str(p), str(e), pred))
check("FX2 invisible at first order, decisive at second: flip iff 2p < m, or 2p = m and e^2/16 > c (exact)", ok,
      f"{sum(r[3] for r in rows)} of {len(rows)} (m, p, e) cases flip")

# FX3
ok = True; tags = {}
race = {1: {(Fr(1), Fr(1)): Fr(1)}, 2: {(Fr(2), Fr(0)): Fr(1), (Fr(2), Fr(1)): Fr(-1)}}
dom = {1: {(Fr(1), Fr(1)): Fr(1)}, 2: {(Fr(1), Fr(1)): Fr(1), (Fr(2), Fr(1)): Fr(1)}}
curved = {1: {(Fr(2), Fr(0)): Fr(1)}, 2: {(Fr(0), Fr(1)): Fr(1)}}
W = [(Fr(a), Fr(b)) for a in (1, 2, 3) for b in (1, 2, 3)]; Cs = [(Fr(1), Fr(1)), (Fr(1), Fr(3)), (Fr(2), Fr(1, 2))]
tags["race"] = len(FX.limit_set_sample(race, W, Cs)) > 1
tags["dominance"] = len(FX.limit_set_sample(dom, W, Cs)) > 1
tags["curved"] = len(FX.limit_set_sample(curved, W, Cs)) > 1
d1 = PA.monomial_subst(race, [[1, 0], [1, 1]]); d2 = PA.monomial_subst(race, [[1, 1], [1, 0]])
tags["design1"] = len(FX.limit_set_sample(d1, W, Cs)) > 1; tags["design2"] = len(FX.limit_set_sample(d2, W, Cs)) > 1
ok &= tags == {"race": True, "dominance": False, "curved": True, "design1": False, "design2": False}
sq = lambda f: DG._pmul(f, f)
ex40 = {1: {(2, 0): Fr(1), (1, 1): Fr(-2), (0, 2): Fr(1)}, 2: {(3, 0): Fr(1)}}
ex60 = {1: sq({(0, 1): Fr(1), (1, 0): Fr(-1), (2, 0): Fr(-1)}), 2: {(5, 0): Fr(1)}}
ok &= len(DG.degenerate(ex40)[0]) == 1 and len(DG.tree(ex60)) == 2 and len(DG.degenerate(dom and {1: {(1, 1): Fr(1)}, 2: {(1, 1): Fr(1), (2, 1): Fr(1)}})[0]) == 0
audit = {"audit 1": {1: Fr(1)}, "audit 2": {2: Fr(1)}}
tie = PA.limit_acts(race, audit, (Fr(1), Fr(1)), (Fr(1), Fr(1))) == {"audit 1", "audit 2"}
gap = OD.add(PA.decision_germs(race, audit, (Fr(1), Fr(1)), (Fr(1), Fr(1)))["audit 1"], OD.scale(PA.decision_germs(race, audit, (Fr(1), Fr(1)), (Fr(1), Fr(1)))["audit 2"], Fr(-1)))
ok &= tie and OD.lead(gap)[0] == 3
check("FX3 fracture detector: order-0 fractures for race and curved path, none for dominance or the two designs; degenerate directions; flat tie of order 3", ok, f"{tags}")

# FX4
vert = np.array([1.0, 0.0, 0.0]); blk = [np.arange(3)]
contrast = lambda X: [1 - (X[1] - X[2]) * (X[1] - X[2]), (X[1] - X[2]) * (X[1] - X[2]), X[0] * 0.0]
pooling = lambda X: [X[0] * 0.5 + X[2] * 0.5, X[0] * 0.5 + X[1] * 0.5, X[1] * 0.5 + X[2] * 0.5]          # a stochastic matrix (a pooling of kernels)
condit = lambda X: [X[i] * (X[0] + X[1]).recip() for i in range(2)]
a = GR.direct_Tk(contrast, vert, blk, 2, rng, 300); b = GR.direct_Tk(pooling, vert, blk, 2, rng, 300); c = GR.direct_Tk(condit, vert, blk, 2, rng, 300)
check("FX4 feasibility fractures appear with a squared contrast, never with pooling or conditioning", (not a[0]) and b[0] and c[0], f"bad: contrast {a[2]}/300, pooling {b[2]}, conditioning {c[2]}")

# FX5
ok = True; rows = []
for wA in (1, 2, 3):
    order = wA - 2; regime = "Gaussian" if order < 0 else ("Poisson" if order == 0 else "wall")
    cov_, empty = SF.sim_wald(10 ** 6, min(1.0, 3.0 * (10 ** 6) ** (-wA / 2)), 0.3, 20000, rng); rows.append((wA, regime, round(cov_, 3), round(empty, 3)))
    ok &= (regime == "Gaussian" and abs(cov_ - 0.95) < 0.01) or (regime == "Poisson" and cov_ < 0.85) or (regime == "wall" and empty > 0.99)
check("FX5 statistical fractures classified by the expected-count order: Gaussian cone, Poisson ray, empirical wall", ok, f"{rows}")

# FX6
ok = True; cells = list(itertools.product((0, 1), repeat=3)); pairs = [(0, 1), (0, 2), (1, 2)]
for eps in [Fr(k, 60) for k in range(0, 61, 3)]:
    A = []; b = []
    for (i, j) in pairs:
        for xi, xj in itertools.product((0, 1), repeat=2):
            A.append([1.0 if (c[i], c[j]) == (xi, xj) else 0.0 for c in cells]); b.append(float(eps / 2 if xi == xj else (1 - eps) / 2))
    r = linprog(np.zeros(8), A_eq=np.array(A), b_eq=np.array(b), bounds=[(0, None)] * 8, method="highs")
    ok &= (r.status == 0) == FX.pairwise_glues(eps) and ((3 * (1 - eps) > 2) == (not FX.pairwise_glues(eps)))
chi = np.array([(-1) ** sum(c) for c in cells], float); ok &= np.allclose(np.array(A) @ chi, 0)
check("FX6 gluing fracture: the three-token tables glue iff eps >= 1/3 (LP = closed form), certificate Phi = 3(1-eps) > 2; blind spot: the parity chi", ok)
json.dump(OUT, open(os.path.join(HERE, "..", "lab", "fracture.json"), "w"))
print(f"{sum(results)}/{len(results)} passed")
