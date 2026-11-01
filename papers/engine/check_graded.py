"""Checks for the Poisson-regime band (Step 4d) and Step 4e (graded classes) of "Probability, Context and Differentiation: The Layers".
  PB1  the exact conditional test: k_alpha(T+1) <= k_alpha(T) + 1 for all T (so the rejection region grows with X_C) and the exposure is
       increasing in mu_C; at the null it is at most alpha (the exact test is conservative)
  PB2  the band is exact: at mu_C_max the exposure equals beta (to 1e-6); Monte Carlo agrees within error, and 1.2 mu_C_max exceeds beta
  PB3  the Gaussian limit: B_pois -> J's B = (Phi^-1(1 - alpha) - Phi^-1(1 - beta)) sqrt(pi/(1 - pi)) as mu_O grows (monotone approach)
  PB4  the Poisson ray: at bounded mu_O the band is a fixed number above B; the blind spot: the test cannot reject when T <= t_b
       (t_b = 3 for pi = 0.3, alpha = 1 - Phi(2))
  PB5  the wall side: as mu_O -> 0, Delta_max -> lambda* with P(Poisson(lambda*) >= t_b + 1) = beta; the band in standard errors diverges and
       the adversary's expected share of the flags tends to 1 (it owns the rare event)
  GC1  the chain: the zero section T^k -> T^{k+1} preserves and reflects membership, so M_{k+1} is contained in M_k
  GC2  M_1 is not M_2: q -> (1 - (q1-q2)^2, (q1-q2)^2, 0) keeps first-order cones at the vertex and fails at order 2
  GC3  M_2 is not M_3: q -> (1 - g, g, 0), g = x1 (x1 - x2)^2, passes order 2 and fails at order 3 at the vertex
  GC4  units: a Weil number with positive constant term lies in T^k H, and so does its inverse
  GC5  subtraction-free maps lie in every M_k: conditioning, the labelled conditioning, the Bayes-weight map (pi, p) -> pi^A, the Bayesian
       inverse, normalised products and random normalised positive polynomial maps, at corners, for k = 3 and 4
  GC6  contrast: random normalised maps built with a difference (a square of a contrast) fail somewhere at order 2 or 3
"""
import os, sys, json, random, math, itertools, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
from scipy.stats import norm, poisson
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import poissonband as PB
from pce import graded as GR
from pce import tangent as TG
from pce import tangent_layers as TLM
from pce.tangent import HD
rng = np.random.default_rng(20261031); random.seed(20261031); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
pi = 0.3; alpha = float(norm.sf(2.0)); beta = 0.05

# PB1
ok = all(PB.k_alpha(T + 1, pi, alpha) <= PB.k_alpha(T, pi, alpha) + 1 for T in range(0, 3000))
for mo in (0.1, 3.0, 50.0):
    mu0 = pi * mo / (1 - pi); ex = [PB.exposure(mu0 * f, mo, pi, alpha) for f in (1, 1.5, 2, 3, 5, 8)]
    ok &= all(b >= a - 1e-15 for a, b in zip(ex, ex[1:])) and ex[0] <= alpha + 1e-12
check("PB1 exact conditional test: k(T+1) <= k(T)+1; exposure increasing in mu_C; at most alpha under the null", ok)

# PB2
ok = True; mc = []
for mo in (0.5, 3.0, 30.0):
    m = PB.mu_C_max(mo, pi, alpha, beta); ok &= abs(PB.exposure(m, mo, pi, alpha) - beta) < 1e-5
    reps = 200000; XC = rng.poisson(m, reps); XO = rng.poisson(mo, reps); ks = np.array([PB.k_alpha(int(t), pi, alpha) for t in range(int((XC + XO).max()) + 1)])
    rate = float((XC >= ks[XC + XO]).mean()); ok &= abs(rate - beta) < 0.003; mc.append(round(rate, 4))
    ok &= PB.exposure(1.2 * m, mo, pi, alpha) > beta
check("PB2 the band is exact: exposure = beta at mu_C_max; Monte Carlo agrees; beyond it exposure exceeds beta", ok, f"Monte Carlo exposure {mc}")

# PB3
Bg = PB.gauss_band(pi, alpha, beta); seq = [(mo, PB.band(mo, pi, alpha, beta)[1]) for mo in (10.0, 100.0, 1000.0, 10000.0)]
ok = all(b2 < b1 for (_, b1), (_, b2) in zip(seq, seq[1:])) and abs(seq[-1][1] - Bg) < 0.01
check("PB3 Gaussian limit: B_pois decreases to J's band B as mu_O grows", ok, f"B = {Bg:.4f}; B_pois {[(m, round(b, 4)) for m, b in seq]}")
OUT["PB3"] = [(m, b) for m, b in seq]

# PB4
tb = PB.blind_total(pi, alpha); d3, b3 = PB.band(3.0, pi, alpha, beta)
ok = tb == 3 and all(PB.k_alpha(T, pi, alpha) == T + 1 for T in range(tb + 1)) and PB.k_alpha(tb + 1, pi, alpha) <= tb + 1 and b3 > 2 * Bg
pblind = float(poisson.cdf(tb, PB.mu_C_max(3.0, pi, alpha, beta) + 3.0))
check("PB4 the Poisson ray: band a fixed number above B; blind spot T <= t_b where no adversary can be exposed", ok,
      f"t_b = {tb}; at mu_O = 3: Delta_max = {d3:.3f} flags, B_pois = {b3:.3f} SE (= {b3 / Bg:.2f} B); P(T <= t_b) = {pblind:.3f}")

# PB5
from scipy.optimize import brentq
lam = brentq(lambda l: poisson.sf(tb, l) - beta, 0.01, 20)
rows = []
ok = True
for mo in (0.1, 0.01, 0.001):
    d, b = PB.band(mo, pi, alpha, beta); share = (d + pi * mo / (1 - pi)) / (d + pi * mo / (1 - pi) + mo); rows.append((mo, round(d, 4), round(b, 2), round(share, 4)))
ok &= abs(rows[-1][1] - lam) < 0.01 and rows[-1][2] > rows[0][2] and rows[-1][3] > 0.999
check("PB5 the wall side: Delta_max -> lambda* (counts), the band in SEs diverges, the adversary's share of the flags -> 1", ok,
      f"lambda* = {lam:.4f}; (mu_O, Delta_max, B_pois, share) {rows}")

# GC1
ok = True
for _ in range(300):
    k = random.choice([1, 2, 3]); n = 4; x = np.array([0.0, 0.3, 0.7, 0.0]); c = np.zeros((2 ** k, n)); c[0] = x
    for S in range(1, 2 ** k):
        z = rng.integers(-2, 3, size=n).astype(float); z[1] -= z.sum(); c[S] = z
    ok &= TG.in_T(c, k) == TG.in_T(GR.zero_embed(c, k), k + 1)
check("GC1 the zero section T^k -> T^(k+1) preserves and reflects membership: M_(k+1) is contained in M_k", ok)

# GC2, GC3
vert = np.array([1.0, 0.0, 0.0]); blk = [np.arange(3)]
f2 = lambda X: [1 - (X[1] - X[2]) * (X[1] - X[2]), (X[1] - X[2]) * (X[1] - X[2]), X[0] * 0.0]
r1 = GR.direct_Tk(f2, vert, blk, 1, rng, 300); r2 = GR.direct_Tk(f2, vert, blk, 2, rng, 600)
check("GC2 M_1 is not M_2: (1-(q1-q2)^2, (q1-q2)^2, 0) passes order 1 and fails order 2 at the vertex", r1[0] and not r2[0], f"order 1: {r1[2]}/{r1[1]} bad; order 2: {r2[2]}/{r2[1]} bad")
g = lambda X: X[1] * (X[1] - X[2]) * (X[1] - X[2])
f3 = lambda X: [1 - g(X), g(X), X[0] * 0.0]
s2 = GR.direct_Tk(f3, vert, blk, 2, rng, 600); s3 = GR.direct_Tk(f3, vert, blk, 3, rng, 1500)
check("GC3 M_2 is not M_3: g = x1 (x1 - x2)^2 passes order 2 and fails order 3 at the vertex", s2[0] and not s3[0], f"order 2: {s2[2]}/{s2[1]} bad; order 3: {s3[2]}/{s3[1]} bad")

# GC4
ok = True
for _ in range(500):
    k = random.choice([1, 2, 3, 4]); c = rng.integers(-5, 6, size=2 ** k).astype(float); c[0] = rng.integers(1, 4)
    ok &= TG.in_TH(c, k); inv = HD(c.reshape(-1, 1), k).recip().c[:, 0]; ok &= TG.in_TH(inv, k)
    ok &= np.allclose((HD(c.reshape(-1, 1), k) * HD(inv.reshape(-1, 1), k)).c[:, 0], np.r_[1.0, np.zeros(2 ** k - 1)])
check("GC4 units: a Weil number with positive constant term and its inverse both lie in T^k H", ok)

# GC5
ok = True; tested = []
A = [0, 1, 2]
condf = lambda X: [X[i] * (X[0] + X[1] + X[2]).recip() for i in range(3)]
def bayes_w(X):                                          # X = (w_1, w_2, p_1 (4 cells), p_2 (4 cells)); output pi^A on two sources, A = first two cells
    PA = [X[2] + X[3], X[6] + X[7]]; num = [X[0] * PA[0], X[1] * PA[1]]; Z = (num[0] + num[1]).recip(); return [num[0] * Z, num[1] * Z]
def lab_cond(X):                                          # labelled conditioning on L x A, two sources, A = cells 0, 1
    cells = [X[0] * X[2], X[0] * X[3], X[1] * X[6], X[1] * X[7]]; Z = (cells[0] + cells[1] + cells[2] + cells[3]).recip(); return [c * Z for c in cells]
def bayes_inv(X):                                         # p on 3 cells, K rows (3 x 2); posterior of x given y = 0
    j = [X[i] * X[3 + 2 * i] for i in range(3)]; Z = (j[0] + j[1] + j[2]).recip(); return [v * Z for v in j]
def prodl(X): return [X[i] * X[3 + j] for i in range(3) for j in range(2)]
cases = [(condf, np.array([0.0, 0.3, 0.2, 0.25, 0.25]), [np.arange(5)]),
         (condf, np.array([0.0, 0.0, 0.4, 0.6, 0.0]), [np.arange(5)]),
         (bayes_w, np.array([1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.3, 0.0, 0.3, 0.4]), [np.arange(2), 2 + np.arange(4), 6 + np.arange(4)]),
         (lab_cond, np.array([0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.3, 0.0, 0.3, 0.4]), [np.arange(2), 2 + np.arange(4), 6 + np.arange(4)]),
         (bayes_inv, np.array([0.0, 0.6, 0.4, 0.5, 0.5, 0.0, 1.0, 0.3, 0.7]), [np.arange(3), 3 + np.arange(2), 5 + np.arange(2), 7 + np.arange(2)]),
         (prodl, np.array([0.0, 1.0, 0.0, 0.0, 1.0]), [np.arange(3), 3 + np.arange(2)])]
for f, x, bl in cases:
    for k in (3, 4):
        r = GR.direct_Tk(f, x, bl, k, rng, 150 if k == 3 else 60); ok &= r[0]; tested.append(r[1])
for _ in range(15):                                       # random normalised positive polynomial maps Delta_3 -> Delta_3 at a vertex or edge
    P = [[(float(rng.integers(1, 4)), tuple(int(e) for e in rng.integers(0, 3, size=3))) for _ in range(2)] for _ in range(3)]
    x = random.choice([np.array([1.0, 0, 0]), np.array([0.5, 0.5, 0]), np.array([0.2, 0.3, 0.5])])
    f = GR.sf_map(P)
    try: val = f(HD.var(x, []))
    except ZeroDivisionError: continue
    if any(not np.isfinite(v.c).all() for v in val): continue
    if float(sum(v.c[0] for v in val)) == 0: continue
    r = GR.direct_Tk(f, x, [np.arange(3)], 3, rng, 80); ok &= r[0]; tested.append(r[1])
check("GC5 subtraction-free maps are in every M_k: conditioning, labelled conditioning, Bayes weights, Bayesian inverse, products, random positive maps", ok,
      f"{sum(tested)} T^3/T^4 points over {len(tested)} map-point-order cases")

# GC6
fails = 0; trials = 0
for _ in range(40):
    a, b = rng.integers(1, 3, size=2)
    g6 = lambda X, a=a, b=b: (X[1] * a - X[2] * b) * (X[1] * a - X[2] * b)
    f6 = lambda X, g6=g6: [X[0] + X[1] + X[2] - g6(X) * 0.25, g6(X) * 0.25, X[0] * 0.0]
    r = GR.direct_Tk(f6, vert, blk, 2, rng, 200); trials += 1; fails += not r[0]
check("GC6 contrast: maps carrying the square of a contrast leave M_2 at the vertex (subtraction is where feasibility breaks)", fails == trials, f"{fails}/{trials} fail")
json.dump(OUT, open(os.path.join(HERE, "..", "lab", "graded.json"), "w"))
print(f"{sum(results)}/{len(results)} passed")
