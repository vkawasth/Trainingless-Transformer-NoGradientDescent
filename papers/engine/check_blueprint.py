"""Checks for Sections "The exclusion filtration and fibre homology" and "A computational blueprint" of "Probability, Context and
Differentiation: The Layers". One synthetic world (pce/blueprint.py): three binary readings, two sources, a flag event.
  BP1  categorical skeleton: the two-type law is a Kleisli composite (latent kernel then product kernel), a probability law; three windows
       taken from one pooled law always glue (any mixing share, 200 random laws): contextuality needs context-dependent provenance
  BP2  the contextuality wall: on a 41 x 33 grid of (theta, kappa) the linear programme agrees with the four Boole (Bell) inequalities; the wall
       theta*(kappa) is decreasing and absent for small kappa; the violated inequality is a certificate (valid on all 8 deterministic records,
       violated by the tables)
  BP3  the exclusion filtration on labelled cells: ranks 4, 7, 14, 16; graded pieces 4, 3, 7, 2 (sum 16); exclusions nested 12 > 9 > 2 > 0;
       the annihilator identity equals the direct rank at each level; the decision-relevant ranks telescope (1 + 1 + 1 + 1 = rank 4)
  BP4  convex exclusions are contractible (segments stay inside, straight-line contraction); the nonlinear two-type fibre is not convex
  BP5  fibre homology: for kappa > 0 the fibre has b0 = 2 (label swap, Jacobian rank 7 at each point: isolated); at kappa = 0 the two points
       land on a segment of exact explanations, b0 = 1 (Betti jump = identifiability fracture), the gap closes linearly and sigma_min -> 0
  BP6  gauge-invariant decisions ignore H0: every function of the law agrees on both components; a latent-indexed act does not (undetermined)
  BP7  two fans: the order fan of P(A) (rays w_t = 2 w_s, w_s = 2 w_t; s^2 t owns no cone), the decision fan of N_ab (one ray w_s = w_t, initial
       form 1 - 7c/4 - c^3, wall c = 1/2), common refinement 4 cones, tie-order map m = v(N) - v(P) positive exactly in the middle cones;
       numerics along arcs agree
  BP9  how much data the decision needs: along arcs in each cone of the refinement, the least N with P(wrong act) = 0.1 (exact, Skellam)
       scales like tau^(-e) with e = 2 v(N) - v_dec (fitted slopes within 0.01), not the fixed-spread exponent 2 v(N) - v(P); on the wall ray
       with t = (s/2)(1 + s^r), e = 3 + 2r; the Skellam formula agrees with Monte Carlo
  BP8  fracture and reach: relabelling the uninformative cell (N + e ts) decides by sign(e) in the middle cones and never in the outer cones;
       on the wall ray the act flips at c = 1/2 (reach |c - 1/2|)
"""
import os, sys, json, itertools
from fractions import Fraction as Fr
import numpy as np
from scipy.linalg import null_space
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import blueprint as BP
rng = np.random.default_rng(20261101); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)

# BP1
ok = True
for k in (0.0, 0.5, 1.0, 1.6):
    P = BP.source1(k); ok &= abs(P.sum() - 1) < 1e-12 and P.min() >= 0
    lat = np.array([0.4, 0.6]); a = [0.5 + k * BP.D_SEP, 0.5 - k * BP.D_SEP]
    kern = np.array([[np.prod([ai[i] if x[i] else 1 - ai[i] for i in range(3)]) for x in BP.CELLS3] for ai in a])
    ok &= np.allclose(lat @ kern, P)                                           # Kleisli composite: latent law pushed through the kernel
bad = 0
for _ in range(200):
    Q = rng.dirichlet(np.full(8, rng.choice([0.2, 1.0, 5.0])))
    bad += not BP.glues({(i, j): BP.pair_table(Q, i, j) for i, j in ((0, 1), (1, 2), (0, 2))})[0]
for th in np.linspace(0, 1, 11):
    for k in (0.6, 1.0, 1.6):
        Q = (1 - th) * BP.source1(k) + th * BP.source2(k); bad += not BP.glues({(i, j): BP.pair_table(Q, i, j) for i, j in ((0, 1), (1, 2), (0, 2))})[0]
ok &= bad == 0 and all(BP.glues(BP.window_tables(0.0, k))[0] for k in (0.3, 1.0, 1.6))
check("BP1 skeleton: Kleisli composite is a law; windows from one pooled law always glue (233 cases)", ok, f"non-gluing pooled cases: {bad}")

# BP2
ths = np.linspace(0, 1, 41); kas = np.linspace(0.05, 1.65, 33); agree = 0; slack = np.zeros((len(kas), len(ths)))
for a_, k in enumerate(kas):
    for b_, th in enumerate(ths):
        T = BP.window_tables(th, k); g, sl = BP.glues(T); v = BP.boole(T)[0]; slack[a_, b_] = v
        agree += g == (v <= 1e-12)
walls = [BP.theta_wall(k) for k in kas]; present = [w for w in walls if w is not None]
mono = all(x >= y - 1e-12 for x, y in zip(present, present[1:])); k0 = kas[[w is not None for w in walls].index(True)]
T = BP.window_tables(1.0, 1.0); v, vs = BP.boole(T); i_ = int(np.argmax(vs))
def boole_on_record(x):
    tabs = {(i, j): np.zeros((2, 2)) for i, j in ((0, 1), (1, 2), (0, 2))}
    for (i, j) in tabs: tabs[(i, j)][x[i], x[j]] = 1.0
    return BP.boole(tabs)[1][i_]
cert = all(boole_on_record(x) <= 1e-12 for x in BP.CELLS3) and vs[i_] > 0
OUT["wall"] = {"kappa": kas.tolist(), "theta": [None if w is None else float(w) for w in walls]}; OUT["slack"] = slack.tolist(); OUT["thetas"] = ths.tolist()
check("BP2 contextuality wall: LP = Boole closed form on the grid; theta*(kappa) decreasing, absent for small kappa; certificate valid",
      agree == slack.size and mono and cert, f"agree {agree}/{slack.size}; wall appears at kappa ~ {k0:.2f}; theta*(1) = {BP.theta_wall(1.0):.4f}; violated inequality #{i_} by {vs[i_]:.4f}")

# BP3
C = BP.cells_lab(); one = np.ones((1, 16))
Ks = [np.vstack([one, BP.marginal_matrix_lab(l)]) for l in range(4)]; rk = [int(np.linalg.matrix_rank(K)) for K in Ks]
ker = [null_space(K) for K in Ks]; dims = [k.shape[1] for k in ker]
nested = all(np.linalg.matrix_rank(np.hstack([ker[l], ker[l + 1]])) == dims[l] for l in range(3))
x = lambda c, i: c[1 + i]; lab = lambda c: 1 if c[0] else -1
rho = np.array([[x(c, 0) for c in C], [x(c, 0) * x(c, 1) for c in C], [lab(c) * x(c, 0) * x(c, 1) for c in C],
                [(-1) ** (c[1] + c[2] + c[3]) for c in C]], float)
def rel_direct(K): N = null_space(K); return np.linalg.matrix_rank(rho @ N) if N.size else 0
def rel_annih(K):
    r = np.linalg.matrix_rank(rho); return r - (r + np.linalg.matrix_rank(K) - np.linalg.matrix_rank(np.vstack([rho, K])))
rel0 = [int(rel_direct(one))] + [int(rel_direct(K)) for K in Ks]; relA = [int(rel_annih(one))] + [int(rel_annih(K)) for K in Ks]
graded = [rk[0] - 1] + [rk[l + 1] - rk[l] for l in range(3)]; resolved = [rel0[l] - rel0[l + 1] for l in range(4)]
OUT["filtration"] = {"rank": rk, "graded": [1] + graded, "exclusion": dims, "relevant": rel0, "resolved": resolved}
check("BP3 exclusion filtration: ranks 4,7,14,16; graded 4,3,7,2 (with mass); exclusions nested; annihilator = direct; resolved dims telescope",
      rk == [4, 7, 14, 16] and dims == [12, 9, 2, 0] and nested and rel0 == relA and resolved == [1, 1, 1, 1] and sum(resolved) == np.linalg.matrix_rank(rho),
      f"rank {rk}, exclusion {dims}, relevant {rel0}, resolved per level {resolved}")

# BP4
ok = True; P16 = 0.9 * np.r_[0.5 * BP.source1(1.0), 0.5 * BP.source2(1.0)] + 0.1 / 16      # an interior labelled law
for l in range(3):
    N = ker[l]; pts = []
    while len(pts) < 20:
        z = P16 + N @ rng.normal(0, 0.02, N.shape[1])
        if z.min() >= 0: pts.append(z)
    for _ in range(200):
        a_, b_ = rng.choice(len(pts), 2, replace=False); t = rng.uniform()
        y = (1 - t) * pts[a_] + t * pts[b_]; ok &= y.min() >= -1e-15 and np.allclose(Ks[l] @ y, Ks[l] @ P16)
        h = (1 - t) * pts[a_] + t * P16; ok &= h.min() >= -1e-15 and np.allclose(Ks[l] @ h, Ks[l] @ P16)   # contraction to P16
_, cl = BP.fibre_points(BP.source1(1.0), 40, rng); mid = 0.5 * (cl[0] + cl[1])
mres = np.max(np.abs(BP.twotype_law(mid[0], mid[1:4], mid[4:7]) - BP.source1(1.0)))
check("BP4 convex exclusions are contractible (segments and straight-line contraction stay inside); the two-type fibre is not convex",
      ok and mres > 1e-3, f"midpoint of the two explanations misses the law by {mres:.4f}")

# BP5
def jac(th, h=1e-6):
    f = lambda z: BP.twotype_law(z[0], z[1:4], z[4:7]); return np.array([(f(th + h * e) - f(th - h * e)) / (2 * h) for e in np.eye(7)]).T
ok = True; rows = []
for k in (0.1, 0.4, 0.8, 1.2, 1.6):
    _, cl = BP.fibre_points(BP.source1(k), 60, rng); cl = sorted(cl, key=lambda z: z[0])
    swap = lambda z: np.r_[1 - z[0], z[4:7], z[1:4]]
    sv = [np.linalg.svd(jac(z), compute_uv=False) for z in cl]; smin = min(s[6] for s in sv)
    gap = float(np.max(np.abs(cl[0] - cl[1]))) if len(cl) == 2 else 0.0
    ok &= len(cl) == 2 and np.allclose(swap(cl[0]), cl[1], atol=1e-5) and all(np.sum(s > 1e-8) == 7 for s in sv)
    rows.append((k, len(cl), round(gap, 4), float(smin)))
seg = all(np.allclose(BP.twotype_law(w, np.full(3, .5), np.full(3, .5)), BP.source1(0.0)) for w in np.linspace(0, 1, 11))
r0 = np.sum(np.linalg.svd(jac(np.r_[0.4, np.full(6, .5)]), compute_uv=False) > 1e-8)
OUT["fibre"] = rows
check("BP5 fibre homology: b0 = 2 (swap pair, isolated) for kappa > 0; at kappa = 0 one segment of explanations, b0 = 1, Jacobian rank drops",
      ok and seg and r0 < 7 and rows[0][3] < rows[-1][3], f"(kappa, b0, gap, sigma_min) {[(r[0], r[1], r[2], f'{r[3]:.2e}') for r in rows]}; rank at collision {r0}")

# BP6
_, cl = BP.fibre_points(BP.source1(1.0), 40, rng); L = [BP.twotype_law(z[0], z[1:4], z[4:7]) for z in cl]
inv = [lambda Pl: Pl @ np.array([x[0] for x in BP.CELLS3]), lambda Pl: Pl[7] / (Pl[7] + Pl[6]), lambda Pl: -np.sum(Pl * np.log(Pl))]
latent = lambda z: "type 1 is the high-X0 type" if z[1] > 0.5 else "type 1 is the low-X0 type"
check("BP6 gauge-invariant decisions ignore H0: functions of the law agree on both components; a latent-indexed act differs (undetermined)",
      all(abs(f(L[0]) - f(L[1])) < 1e-9 for f in inv) and latent(cl[0]) != latent(cl[1]), f"{[latent(z) for z in cl]}")

# BP7
def val(poly, w): return min(a * w[0] + b * w[1] for (a, b) in poly)
def owners(poly, w): m = val(poly, w); return {(a, b) for (a, b) in poly if a * w[0] + b * w[1] == m}
PA = {}; N = {}
for key, poly in BP.FLAG_CELLS.items():
    for e, c in poly.items(): PA[e] = PA.get(e, 0) + c; N[e] = N.get(e, 0) + BP.FLAG_GAP[key] * c
N = {e: c for e, c in N.items() if c != 0}
W = [(Fr(1), Fr(q, 8)) for q in range(1, 40)]                                    # w = (1, slope) sweeps the open quadrant
ownP = [owners(PA, w) for w in W]; ownN = [owners(N, w) for w in W]
raysP = [w[1] for w, o in zip(W, ownP) if len(o) > 1]; raysN = [w[1] for w, o in zip(W, ownN) if len(o) > 1]
never = (Fr(2), Fr(1)) not in set().union(*[o for o in ownP if len(o) == 1])
ini = lambda c: sum(co * c ** b for (a, b), co in N.items())                    # initial form on the ray w_s = w_t (all of N is on the edge)
m = lambda w: val(N, w) - val(PA, w)
cones = sorted(set(raysP) | set(raysN))
tie = {str(w[1]): str(m(w)) for w in W if w[1] in (Fr(1, 4), Fr(3, 4), Fr(5, 4), Fr(3, 2), Fr(3), Fr(4))}
num = True
for w in [(1.0, 0.25), (1.0, 0.75), (1.0, 1.5), (1.0, 3.0)]:
    ev = lambda poly, tau: sum(float(c) * np.exp(-tau * (float(a) * w[0] + float(b) * w[1])) for (a, b), c in poly.items())
    t1, t2 = 30.0, 40.0; slope = (np.log(abs(ev(N, t2) / ev(PA, t2))) - np.log(abs(ev(N, t1) / ev(PA, t1)))) / (t1 - t2)
    num &= abs(slope - float(m((Fr(w[0]).limit_denominator(), Fr(w[1]).limit_denominator())))) < 1e-3
OUT["fans"] = {"raysP": [str(r) for r in raysP], "raysN": [str(r) for r in raysN], "tie": tie}
check("BP7 two fans: P(A) rays at slopes 1/2, 2 (s^2 t owns no cone); N_ab one ray (slope 1) with wall c = 1/2; refinement 4 cones; m > 0 in the middle",
      raysP == [Fr(1, 2), Fr(2)] and raysN == [Fr(1)] and never and ini(Fr(1, 2)) == 0 and ini(Fr(1, 4)) > 0 and ini(Fr(1)) < 0 and len(cones) + 1 == 4
      and all((Fr(1, 2) < w[1] < Fr(2)) == (m(w) > 0) for w in W) and num, f"tie order by slope w_t/w_s: {tie}")

# BP8
ok = True; rows = []
for slope in (0.25, 0.75, 1.5, 3.0):
    w = (1.0, slope); dec = []
    for e in (1e-3, -1e-3):
        Ne = dict(N); Ne[(Fr(1), Fr(1))] = Ne.get((Fr(1), Fr(1)), 0) + Fr(e).limit_denominator(10 ** 6)
        tau = 60.0; s, t = np.exp(-tau * w[0]), np.exp(-tau * w[1]); vN = sum(float(c) * s ** float(a) * t ** float(b) for (a, b), c in Ne.items())
        dec.append(np.sign(vN))
    middle = 0.5 < slope < 2; ok &= (dec[0] != dec[1]) == middle; rows.append((slope, "flips with sign(e)" if dec[0] != dec[1] else "stable"))
for c in (0.3, 0.49, 0.51, 0.8):
    s = 1e-3; vN = sum(float(co) * s ** float(a) * (c * s) ** float(b) for (a, b), co in N.items()); ok &= (vN > 0) == (c < 0.5)
check("BP8 fracture and reach: relabelling the uninformative cell decides by sign(e) exactly in the middle cones; on the ray the act flips at c = 1/2",
      ok, f"{rows}")
# BP9
ok = True; rows = []; taus = np.geomspace(1e-6, 1e-14, 6)
for w in [(1, 3), (1, 0.4), (1, 0.75), (1, 1.25), (1, 1.5), (1, 1)]:
    ex = BP.sample_exponents(w); Ns = [BP.n_star(t ** w[0], t ** w[1]) for t in taus]
    sl = float(np.polyfit(np.log(1 / taus), np.log(Ns), 1)[0]); ok &= abs(sl - ex["e"]) < 0.01
    rows.append((w, round(ex["vP"], 3), round(ex["m"], 3), round(ex["e"], 3), round(ex["fixed_spread"], 3), round(sl, 3)))
r = 0.5; tw = np.geomspace(1e-6, 1e-14, 6); Nw = [BP.n_star(t, t / 2 * (1 + t ** r)) for t in tw]
slw = float(np.polyfit(np.log(1 / tw), np.log(Nw), 1)[0]); ok &= abs(slw - (3 + 2 * r)) < 0.01
mc = []
for (w, tau, N) in [((1, 1.5), 1e-2, None), ((1, 0.75), 1e-2, None)]:
    s_, t_ = tau ** w[0], tau ** w[1]; N = BP.n_star(s_, t_); m = N * BP.flag_masses(s_, t_)
    n = rng.poisson(m, size=(400000, 4)); g = n @ BP.GAP_VEC; truth = np.sign(m @ BP.GAP_VEC)
    pw = np.mean(np.where(g == 0, 0.5, (np.sign(g) != truth).astype(float))); mc.append(round(float(pw), 4)); ok &= abs(pw - 0.1) < 0.003
OUT["bands"] = {"rows": [list(map(str, r_)) for r_ in rows], "wall": [r, slw]}
check("BP9 data needed per cone: N* ~ tau^-(2v(N) - v_dec), not the fixed-spread 2v(N) - v(P); wall ray 3 + 2r; Skellam = Monte Carlo", ok,
      f"(w, v(P), m, e, fixed-spread, fitted) {rows}; wall r = 0.5: {slw:.3f}; Monte Carlo P(wrong) at N*: {mc}")
json.dump(OUT, open(os.path.join(HERE, "..", "lab", "blueprint.json"), "w"), default=str)
print(f"{sum(results)}/{len(results)} passed")
