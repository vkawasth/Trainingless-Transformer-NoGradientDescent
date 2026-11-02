"""Checks for the revised Section "Fractures and blind spots" of "Probability, Context and Differentiation: The Layers".
  FB1  the fragility exponent: at distance delta from a wall, the least flipping injection share scales like delta along a direction with a
       non-zero first derivative and like delta^(1/2) along a first-derivative-free one; in the covariance example it is exactly
       sqrt(delta) / (1/4 + delta); a linear decision at a boundary law with its gap in C_x^* cannot be flipped by any injection
  FB2  reach (exact): on the race's ray t = c s three outcomes break together: the limit (not a singleton across the ray), the audit act
       (tied at c = 1), and the value of learning the source label (zero in both cones, 1 - max(c, 1)/(1 + c) on the ray)
  FB3  observational fracture (exact): for a pooled law with two sources on A, the forgetting fibre contains states that audit source 1 and
       states that audit source 2; no pooled statistic of any order decides the act
  FB4  decision bands near walls, Gaussian regime: P(wrong act) -> Phi(-|D| sqrt(N P(A)) / sigma) (Monte Carlo); on the decorated fan's
       decision wall of the race (tie at c = 1, decided at order q = 1, P(A) of order m = 2), the act is reliable iff N s^(m + 2q) -> infinity:
       with s = N^(-a), a = 0.15 gives the right act, a = 0.35 a coin toss
  FB5  decision bands on the Poisson ray (exact against Monte Carlo): with N P(A) -> lambda, P(wrong act) tends to a fixed number given by
       independent Poisson cell counts, including P(no event) = e^(-lambda); two decorations of one Newton polygon give different bands
"""
import os, sys, json, math, random, itertools
from fractions import Fraction as Fr
import numpy as np
from scipy.stats import norm
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import fracture as FX
from pce import fracband as FB
from pce import jumps as JP
from pce import fans as FN
from pce import effects as EF
from pce import tangent_layers as TLM
rng = np.random.default_rng(20261101); random.seed(20261101); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)

# FB1
F = lambda P: P[3] - (P[2] + P[3]) * (P[1] + P[3]); ds = [1e-2, 1e-3, 1e-4, 1e-5]; e1 = []; e2 = []; ok = True
for d in ds:
    Phi = np.array([.25 + d, .25 - d, .25 - d, .25 + d]); U2 = np.array([0, .25 - d, .25 - d, .5 + 2 * d]) - Phi; U1 = np.array([0, .5, .5, 0]) - Phi
    a = FX.flip_share_line(F, Phi, U1); b = FX.flip_share_line(F, Phi, U2); e1.append(a); e2.append(b)
    ok &= abs(b - math.sqrt(d) / (0.25 + d)) < 1e-9
s1 = np.polyfit(np.log(ds), np.log(e1), 1)[0]; s2 = np.polyfit(np.log(ds), np.log(e2), 1)[0]; ok &= abs(s1 - 1) < 0.02 and abs(s2 - 0.5) < 0.02
for _ in range(200):                                       # linear decisions at boundary ties: unbreakable iff the gap lies in C_x^*
    x = np.array([float(v) for v in [0.5, 0.5, 0.0, 0.0]]); g = rng.integers(-3, 4, size=4).astype(float); g -= g[0]
    g[1] = 0.0
    inside = EF.dual_cone_member(list(x), list(g)); flips = any(np.dot(g, TLM.rand_feasible(x, rng)) < -1e-12 for _ in range(300))
    ok &= inside == (not flips)
check("FB1 fragility exponent: flip share ~ delta (first order) and ~ delta^(1/2) (first-derivative-free; exactly sqrt(delta)/(1/4+delta)); ties with gap in C_x^* unbreakable", ok,
      f"slopes {s1:.3f}, {s2:.3f}")

# FB2
ok = True
race = {1: {(Fr(1), Fr(1)): Fr(1)}, 2: {(Fr(2), Fr(0)): Fr(1), (Fr(2), Fr(1)): Fr(-1)}}
for c in (Fr(1, 3), Fr(1), Fr(3)):
    p1, act, voi = FX.race_outcomes(c); lim = FN.arc_limit(race, (Fr(1), Fr(1)), (Fr(1), c)); ok &= lim[1] == p1 and voi == 1 - max(p1, 1 - p1) and voi > 0
    ok &= (act == "tie") == (c == 1)
for w in ((Fr(2), Fr(1)), (Fr(1), Fr(2))):                  # off the ray: one source owns A, the label is worth nothing
    lim = FN.arc_limit(race, w, (1, 1)); p1 = lim.get(1, Fr(0)); ok &= 1 - max(p1, 1 - p1) == 0
check("FB2 reach on the race's ray: limit, audit act (tie at c = 1) and value of the source label (0 in both cones, > 0 on the ray) break together", ok)

# FB3
ok = True; X = ["a", "b", "c"]; A = lambda x: x in ("a", "b")
for _ in range(100):
    w = [Fr(random.randint(1, 5)) for _ in X]; P = {x: v / sum(w) for x, v in zip(X, w)}; CP = {x: v / (P["a"] + P["b"]) for x, v in P.items() if A(x)}
    acts = set()
    for q in (Fr(9, 10), Fr(1, 10)):
        Lp = {(0, x): q * v for x, v in CP.items()}; Lp.update({(1, x): (1 - q) * v for x, v in CP.items()})
        Lam = JP.fibre_lift(Lp, P, A, 0); ok &= JP.forget(Lam) == P
        wts = JP.weights(JP.cond_lab(Lam, A)); acts.add(0 if wts.get(0, 0) > wts.get(1, 0) else 1)
    ok &= acts == {0, 1}
check("FB3 observational fracture: one pooled law, two labelled states on its forgetting fibre, two different audit acts", ok)

# FB4
ok = True; rows = []
for (t, s, Ns) in ((0.12, 0.1, (2000, 8000, 20000)), (0.11, 0.1, (4500, 18000))):        # event counts >= 40: Gaussian regime
    f = [t * s, (1 - t) * s * s]; g = [1.0, -1.0]
    for N in Ns:
        pred = FB.predicted_gauss(N, f, g); mc, emp = FB.simulate(N, f, g, 40000, rng); ok &= abs(mc - pred) < 0.02 + 0.15 * pred; rows.append((N, round(pred, 3), round(mc, 3)))
res = {}
for a in (0.15, 0.35):
    ws = []
    for N in (1e6, 1e8):
        s = N ** -a; t = s; f = [t * s, (1 - t) * s * s]; ws.append(FB.simulate(N, f, [1.0, -1.0], 20000, rng)[0])
    res[a] = ws
ok &= res[0.15][-1] < 0.05 and res[0.35][-1] > 0.3 and res[0.15][1] <= res[0.15][0] + 0.01
check("FB4 Gaussian band for the decision: P(wrong) -> Phi(-|D| sqrt(N P(A))/sigma); on the decorated wall reliability needs N s^(m+2q) -> infinity", ok,
      f"(N, predicted, MC) {rows}; wall tie with s = N^-a: a=0.15 {[round(x, 3) for x in res[0.15]]}, a=0.35 {[round(x, 3) for x in res[0.35]]}")
OUT["FB4"] = {str(k): v for k, v in res.items()}

# FB5
ok = True; rows = []
for name, ell in (("x | y at c = 2", [1 / 3, 2 / 3]), ("x + y | x at c = 2", [3 / 4, 1 / 4])):
    D = ell[0] - ell[1]
    for lam in (1.0, 3.0, 8.0):
        ex, em = FB.poisson_exact(lam, ell, [1.0, -1.0], D); N = 10 ** 7; PA = lam / N; f = [PA * ell[0], PA * ell[1]]
        mc, mce = FB.simulate(N, f, [1.0, -1.0], 100000, rng); ok &= abs(mc - ex) < 0.006 and abs(em - math.exp(-lam)) < 1e-9; rows.append((name, lam, round(ex, 4), round(mc, 4)))
ok &= rows[2][2] != rows[5][2]
check("FB5 Poisson-ray decision band: exact Poisson sums match Monte Carlo, include e^-lambda for no event, and differ between two decorations", ok, f"{rows}")
OUT["FB5"] = rows
json.dump(OUT, open(os.path.join(HERE, "..", "lab", "fracband.json"), "w"))
print(f"{sum(results)}/{len(results)} passed")
