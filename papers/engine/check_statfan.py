"""Checks for Step 4d of "Probability, Context and Differentiation: The Layers".
  ST1  the decorated fan is a sheaf on the faces (exact): in_G(in_F f) = in_G f for a face G of F; in_{w'}(in_w f) = in_{w + eps w'} f for
       small eps; on an edge [alpha, beta] the limit map tends to the vertex limits as c^{beta - alpha} -> 0 or infinity
  ST2  same Newton polyhedron, different decorations (exact): x | y gives limit set [0, 1] for pi^A_1; x + y | x gives [1/2, 1]
  ST3  the cross term (exact and Weil): mu(pi + a, p + u) - mu(pi, p) = sum a_l p_l + sum pi_l u_l + sum a_l u_l; with shares on e_1 and sources
       on e_2 the e_1 e_2 coefficient is sum a_l u_l and the pure second derivatives vanish; for C_A o mu the mixed derivative matches the
       product-rule formula through the Bayes weights
  ST4  Theorem J on the statistical fan: with s = N^(-1/2), pi ~ s^{w_pi}, D ~ s^{w_D}: the source statistic has order w_D + w_pi/2 - 1 and the
       pooled impact (in SEs) order w_pi + w_D - 1 (slopes on log scales); the identity impact = a sqrt(pi/(1-pi)); bounded exposure forces
       impact <= B, and impact -> 0 when the share vanishes; the raw pooled shift pi D -> 0 (no tilt) under bounded exposure
  ST5  Monte Carlo of the three regimes (Gaussian model): exposure rate -> 1 below the line, -> the false-alarm rate above it, and stays at
       Phi(a - c_s) on it (the band), at N = 1e3, 1e4, 1e5
  ST6  the Gaussian validity cone: with P(A) = c N^{-w_A/2}, the nominal 95% Wald interval for pi^A_1 covers ~95% when w_A = 1 (counts grow),
       fails when w_A = 2 (Poisson counts; no event with probability ~ e^{-c}), and the empirical conditional is undefined with probability
       -> 1 when w_A = 3 (the empirical wall)
  ST7  negative control: the same Wald procedure at fixed P(A) (inside the Gaussian cone) covers within Monte Carlo error at every N
"""
import os, sys, json, random, itertools, math
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import fans as FN
from pce import statfan as SF
from pce import tangent_layers as TLM
from pce.tangent import HD
from scipy.stats import norm
random.seed(20261031); rng = np.random.default_rng(20261031); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
def rpoly(d, n=4, mx=4):
    f = {}
    for _ in range(n): f[tuple(Fr(random.randint(0, mx)) for _ in range(d))] = Fr(random.randint(1, 5))
    return f

# ST1
ok = True; nedge = 0
for _ in range(300):
    d = random.choice([2, 3]); f = rpoly(d); w = tuple(Fr(random.randint(1, 4)) for _ in range(d)); w2 = tuple(Fr(random.randint(1, 4)) for _ in range(d))
    lhs = FN.initial(FN.initial(f, w), w2); eps = Fr(1, 10 ** 4); rhs = FN.initial(f, tuple(a + eps * b for a, b in zip(w, w2)))
    ok &= lhs == rhs
    F = set(FN.initial(f, w)); G = set(FN.initial(f, tuple(a + eps * b for a, b in zip(w, w2)))); ok &= G <= F and SF.face_part(SF.face_part(f, F), G) == SF.face_part(f, G)
for _ in range(100):
    cells = {l: rpoly(2, n=2, mx=3) for l in range(3)}; V = FN.newton_vertices_2d(cells)
    for al, be in zip(V[:-1], V[1:]):
        F = {a for f in cells.values() for a in f if (a[0] - al[0]) * (be[1] - al[1]) == (a[1] - al[1]) * (be[0] - al[0]) and min(al[0], be[0]) <= a[0] <= max(al[0], be[0])}
        # c^(beta - alpha) -> 0: take c = (1, tiny) or (tiny, 1) according to the sign of beta - alpha
        dx = be[0] - al[0]; big = Fr(10 ** 9)                       # beta_x > alpha_x and beta_y < alpha_y: c = (1/big, 1) makes c^(beta-alpha) tiny
        c_to_alpha = (Fr(1) / big, Fr(1)) if dx > 0 else (Fr(1), Fr(1) / big)
        la = SF.edge_limit(cells, F, c_to_alpha); va = FN.vertex_limit(cells, al)
        ok &= sum(abs(la.get(k, 0) - va.get(k, 0)) for k in cells) < Fr(1, 10 ** 3); nedge += 1
check("ST1 the decoration is a sheaf on the faces: in_G in_F = in_G, in_w' in_w = in_(w + eps w'); edge limit maps run between vertex limits", ok, f"{nedge} edges")

# ST2
A = {1: {(Fr(1), Fr(0)): Fr(1)}, 2: {(Fr(0), Fr(1)): Fr(1)}}; B = {1: {(Fr(1), Fr(0)): Fr(1), (Fr(0), Fr(1)): Fr(1)}, 2: {(Fr(1), Fr(0)): Fr(1)}}
ok = FN.newton_vertices_2d(A) == FN.newton_vertices_2d(B)
la = [FN.arc_limit(A, (Fr(1), Fr(1)), (Fr(1), c)).get(1, 0) for c in (Fr(1, 10 ** 6), Fr(1), Fr(10 ** 6))]
lb = [FN.arc_limit(B, (Fr(1), Fr(1)), (Fr(1), c)).get(1, 0) for c in (Fr(1, 10 ** 6), Fr(1), Fr(10 ** 6))]
ok &= FN.vertex_limit(A, (Fr(1), Fr(0))) == {1: 1} and FN.vertex_limit(A, (Fr(0), Fr(1))) == {2: 1}
ok &= FN.vertex_limit(B, (Fr(1), Fr(0))) == {1: Fr(1, 2), 2: Fr(1, 2)} and FN.vertex_limit(B, (Fr(0), Fr(1))) == {1: 1}
ok &= la[1] == Fr(1, 2) and lb[1] == Fr(2, 3) and all(Fr(1, 2) <= x <= 1 for x in lb)
check("ST2 one Newton polyhedron, two decorations: limit sets [0, 1] and [1/2, 1] for pi^A_1", ok, f"x|y: {[float(x) for x in la]}; x+y|x: {[round(float(x), 4) for x in lb]}")

# ST3
ok = True
for _ in range(100):
    n, nx = 3, 4; pi = [Fr(random.randint(1, 5)) for _ in range(n)]; s = sum(pi); pi = [x / s for x in pi]
    ps = [[Fr(random.randint(1, 5)) for _ in range(nx)] for _ in range(n)]; ps = [[x / sum(p) for x in p] for p in ps]
    a = [Fr(random.randint(-3, 3)) for _ in range(n)]; a[-1] -= sum(a); us = [[Fr(random.randint(-3, 3)) for _ in range(nx)] for _ in range(n)]
    for u in us: u[-1] -= sum(u)
    mu = lambda w, P: [sum(w[l] * P[l][x] for l in range(n)) for x in range(nx)]
    lhs = [x - y for x, y in zip(mu([p + b for p, b in zip(pi, a)], [[x + y for x, y in zip(P, U)] for P, U in zip(ps, us)]), mu(pi, ps))]
    rhs = [sum(a[l] * ps[l][x] + pi[l] * us[l][x] + a[l] * us[l][x] for l in range(n)) for x in range(nx)]
    ok &= lhs == rhs
    fl = lambda v: np.array([float(z) for z in v])
    W = HD.var(fl(pi), [fl(a), None]); Ps = [HD.var(fl(p), [None, fl(u)]) for p, u in zip(ps, us)]
    pool = [sum((W[l] * Ps[l][x] for l in range(1, n)), W[0] * Ps[0][x]) for x in range(nx)]
    ok &= np.allclose([z.c[3] for z in pool], [float(sum(a[l] * us[l][x] for l in range(n))) for x in range(nx)])
    W2 = HD.var(fl(pi), [fl(a), fl(a)]); pool2 = [sum((W2[l] * ps[l][x] for l in range(1, n)), W2[0] * float(ps[0][x])) for x in range(nx)]
    ok &= np.allclose([z.c[3] for z in pool2], 0)
    # conditioned: mixed derivative of C_A o mu through the Bayes weights, A = {0, 1}
    Am = np.array([1, 1, 0, 0], float); cA = [z * Am[i] for i, z in enumerate(pool)]; ZA = cA[0] + cA[1]
    mixed_hd = np.array([(cA[i] * ZA.recip()).c[3] for i in range(nx)])
    P = np.array([fl(p) for p in ps]); U = np.array([fl(u) for u in us]); av = fl(a); pv = fl(pi)
    Pi = pv * (P @ Am); dPa = av * (P @ Am); dPu = pv * (U @ Am); dPau = av * (U @ Am); Z = Pi.sum()
    w_mixed = dPau / Z - (dPa * dPu.sum() + dPu * dPa.sum()) / Z ** 2 - Pi * dPau.sum() / Z ** 2 + 2 * Pi * dPa.sum() * dPu.sum() / Z ** 3
    w_a = dPa / Z - Pi * dPa.sum() / Z ** 2
    CA = [TLM.cond_vec(P[l], Am) for l in range(n)]; DCA = [TLM.dcond(P[l], Am, U[l]) for l in range(n)]
    formula = sum(w_mixed[l] * CA[l] + w_a[l] * DCA[l] for l in range(n))
    ok &= np.allclose(mixed_hd, formula)
check("ST3 the cross term: exact bilinear identity; e1e2 coefficient = sum a_l u_l, pure second derivatives 0; conditioned mixed derivative via Bayes weights", ok)

# ST4
ok = True; sigma = 1.0; cs = 2.0; beta = 0.05; amax = cs - norm.ppf(1 - beta); slopes = []
for wpi, wD in ((0, 1), (0, 0.5), (0, 1.5), (0.5, 0.75), (1, 0.5), (0.4, 1.2)):
    Ns = np.array([1e4, 1e6, 1e8]); s = Ns ** -0.5; pi = 0.3 * s ** wpi; D = 0.8 * s ** wD
    a = np.array([SF.J_stat(d_, p_, n_) for d_, p_, n_ in zip(D, pi, Ns)]); I = np.array([SF.J_impact(d_, p_, n_) for d_, p_, n_ in zip(D, pi, Ns)])
    ok &= np.allclose(I, a * np.sqrt(pi / (1 - pi)))
    sa = np.polyfit(np.log(s), np.log(a), 1)[0]; si = np.polyfit(np.log(s), np.log(I), 1)[0]
    ok &= abs(sa - (wD + wpi / 2 - 1)) < 0.02 and abs(si - (wpi + wD - 1)) < 0.02; slopes.append((wpi, wD, round(float(sa), 3), round(float(si), 3)))
for N in (1e3, 1e5, 1e7):                                     # bounded exposure: largest shift with a <= amax
    for pi in (0.3, 0.05, 0.3 * N ** -0.25):
        D = amax * sigma / math.sqrt(pi * (1 - pi) * N); I = SF.J_impact(D, pi, N)
        ok &= I <= amax * math.sqrt(pi / (1 - pi)) + 1e-9 and pi * D < 5 / math.sqrt(N)
Bv = amax * math.sqrt(0.3 / 0.7)
check("ST4 J on the statistical fan: exposure order w_D + w_pi/2 - 1, impact order w_pi + w_D - 1; impact = a sqrt(pi/(1-pi)) <= B; raw shift -> 0", ok,
      f"(w_pi, w_D, slope a, slope impact) {slopes[:4]}; B at pi = 0.3: {Bv:.3f} SE")
OUT["B_pi03"] = Bv

# ST5
ok = True; rows = []; reps = 20000
for wD, label in ((0.5, "exposed"), (1.0, "band"), (1.5, "invisible")):
    rates = []
    for N in (1e3, 1e4, 1e5):
        s = N ** -0.5; D = 1.5 * s ** wD; r, imp = SF.sim_J(int(N), 0.3, D, 1.0, cs, reps, rng); rates.append(r)
    rows.append((label, [round(x, 3) for x in rates]))
    if label == "exposed": ok &= rates[-1] > 0.99 and rates[0] < rates[-1]
    if label == "band":
        target = 1 - norm.cdf(cs - 1.5 * math.sqrt(0.3 * 0.7)); ok &= all(abs(x - target) < 0.015 for x in rates)
    if label == "invisible": ok &= abs(rates[-1] - (1 - norm.cdf(cs))) < 0.01
check("ST5 Monte Carlo (Gaussian model): exposure -> 1 below the line, stays at Phi(a - c_s) on it (the band), falls to the false-alarm rate above", ok, f"{rows}")
OUT["ST5"] = rows

# ST6, ST7
ok6 = True; ok7 = True; tab = []; reps = 40000; q = 0.3; c = 3.0
for N in (1e2, 1e4, 1e6):
    row = [int(N)]
    for wA in (1, 2, 3):
        cov, empty = SF.sim_wald(int(N), min(1.0, c * N ** (-wA / 2)), q, reps, rng); row.append((round(cov, 3), round(empty, 3)))
    cov0, empty0 = SF.sim_wald(int(N), 0.2, q, reps, rng); row.append((round(cov0, 3), round(empty0, 3))); tab.append(row)
ok6 &= abs(tab[-1][1][0] - 0.95) < 0.01                          # w_A = 1: counts grow like sqrt(N), coverage nominal at N = 1e6
ok6 &= all(r[2][0] < 0.85 for r in tab) and abs(tab[-1][2][1] - math.exp(-c)) < 0.01   # w_A = 2: Poisson(3), no event w.p. e^-3, coverage poor
ok6 &= tab[-1][3][1] > 0.99                                     # w_A = 3: the empirical wall
ok7 &= all(abs(r[4][0] - 0.95) < 0.01 for r in tab[1:])         # fixed P(A): nominal
check("ST6 the Gaussian validity cone: Wald coverage nominal for w_A < 2, fails on the Poisson ray w_A = 2 (no event w.p. e^-c), empirical wall for w_A > 2", ok6,
      f"(N, (cov, P(no event)) for w_A = 1, 2, 3): {[r[:4] for r in tab]}")
check("ST7 negative control: the same procedure at fixed P(A) = 0.2 covers within Monte Carlo error", ok7, f"{[r[4] for r in tab]}")
OUT["ST6"] = [[str(x) for x in r] for r in tab]
json.dump(OUT, open(os.path.join(HERE, "..", "lab", "statfan.json"), "w"))
print(f"{sum(results)}/{len(results)} passed")
