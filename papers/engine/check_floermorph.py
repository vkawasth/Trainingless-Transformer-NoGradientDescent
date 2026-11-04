"""Checks for section 12 of the companion note, objects, morphisms and the categorical comparison (module pce/floermorph.py). All on the
B-side data of W (critical points, values, Hessians); the A-side reading is cited.
  MO1  the objects of F_1 (T = 10^-24): four critical points = chi(F_1); alpha = 1/5: the positive one and two complex ones (holonomies (zeta, zeta),
       zeta^3 = 1) over the barycentre, a real one with holonomy (-1, 1) over (1/5, 3/5); alpha = 3/5: all four over (2/5, 1/5), holonomies (1, 1),
       (-1, 1), (i, -1), (-i, -1)
  MO2  the objects of the 2x3 example and of its one-count neighbour (exact resultants, T = 10^-200): five = chi(Bl_2 CP^2); fibres 3 + 1 + 1 before,
       4 + 1 after
  MO3  morphisms vanish between distinct objects: objects over distinct fibres have disjoint fibres (distinct valuations); objects over one fibre
       have distinct holonomies (distinct summands); all critical values are distinct
  MO4  the Hessian of the positive object is W Cov_w(v), w the law of the terms (mean exponent 0): real, positive definite; every object is Morse
  MO5  the Clifford relations e_i e_j + e_j e_i = Q_ij hold for explicit 2 x 2 generators, which generate all of M_2 (dimension 4: semisimple, so separable)
  MO6  the Clifford shadow of selection: Q/W of the positive object reduces mod T to Cov_q*(v), of rank dim sigma_0 = n - dim(candidates): F_1 rank 2
       at alpha = 1/5 and 1/3, rank 1 at 3/5 and 4/5 with small eigenvalue of valuation (3 alpha - 1)/4; 2x3 example rank 2; one count rank 1 with
       valuation ~1/200; the column-binding example rank 1 with valuation ~1/20 (Q/W converges to Cov_q*(v), the reduction, with q* from the balance
       programme; small-eigenvalue valuation from the last two T, to 25%)
  MO7  the comparison along F_1, alpha from 1/5 to 3/5 (T = 10^-24): the four critical points never collide and stay Morse, so the summands continue;
       explicit real isomorphisms A^T Q(s') A = Q(s) between the positive object's Clifford forms
  MO8  Stokes walls accumulate at the outcome wall: for phases theta = 0.3, 1.0, pi/2, every Stokes event on alpha in [1/5, 1/2] lies within 4 / log(1/T)
       of 1/3 (T = 10^-12, 10^-24, 10^-48); some involve the real-signed object; on [1/20, 1/5] and [1/2, 19/20] there are none (T = 10^-24)
  MO9  the rate: log(1/T) (alpha_theta - 1/3) is constant to 5% across the three values of T, for every event
"""
import os, sys, math
from fractions import Fraction as Fr
import numpy as np
import mpmath as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import floermorph as FM, kahlerbridge as KB, couplingfibre as CF
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
def f1_obj(a, T, dps=200):
    v, lam = KB.F1(a); return FM.objects(v, [mp.mpf(Fr(x).limit_denominator(1000).numerator) / Fr(x).limit_denominator(1000).denominator for x in lam], T, FM.f1_critical(a, T, dps))
def groups(ob, tol=2e-2):
    G = []
    for o in ob:
        for g_ in G:
            if np.abs(g_[0]["u"] - o["u"]).max() < tol: g_.append(o); break
        else: G.append([o])
    return G
T24 = mp.mpf(10) ** -24

# MO1
ok = True; ob15 = f1_obj(0.2, T24); ob35 = f1_obj(0.6, T24); z = np.exp(2j * np.pi / 3)
ok &= len(ob15) == 4 and len(ob35) == 4
bary = [o for o in ob15 if np.allclose(o["u"], 1 / 3, atol=1e-2)]; other = [o for o in ob15 if np.allclose(o["u"], (0.2, 0.6), atol=1e-2)]
ok &= len(bary) == 3 and len(other) == 1 and other[0]["type"] == "real-signed" and np.allclose(other[0]["hol"], (-1, 1), atol=1e-3)
ok &= sorted(o["type"] for o in bary) == ["complex", "complex", "positive"]
ok &= all(any(np.allclose(o["hol"], (zz, zz), atol=1e-3) for zz in (1, z, z * z)) for o in bary)
ok &= all(np.allclose(o["u"], (0.4, 0.2), atol=1e-2) for o in ob35)
H35 = sorted([tuple(np.round(o["hol"], 3)) for o in ob35], key=lambda h: (h[0].real, h[0].imag))
ok &= all(any(np.allclose(o["hol"], h, atol=1e-3) for o in ob35) for h in ((1, 1), (-1, 1), (1j, -1), (-1j, -1)))
check("MO1 F_1 objects: alpha = 1/5 -> 3 over the barycentre (holonomies (z, z)) + 1 signed (-1, 1); alpha = 3/5 -> all 4 over (2/5, 1/5)", ok)

# MO2
def cpl_obj(f, g, tau):
    pts, D = CF.critical_points(f, g, tau, dps=400); ff = [float(x) for x in f]; gf = [float(x) for x in g]; F = CF.facet_cells(ff, gf)
    V, L = CF.cell_functions(Fr(f[0]), Fr(g[0]), Fr(g[1])); T = (mp.mpf(Fr(tau).numerator) / Fr(tau).denominator) ** D
    mp.mp.dps = 400
    return FM.objects([V[c] for c in F], [mp.mpf(Fr(L[c]).numerator) / Fr(L[c]).denominator for c in F], T, pts), T
obA, TA = cpl_obj(CF.EX_F, CF.EX_G, "1/100"); obB, TB = cpl_obj(CF.EX_F, (Fr(30, 100), Fr(44, 100), Fr(26, 100)), "1/100")
FIBA = [(2 / 15, 2 / 15), (0.13, 0.13), (0.2, 0.1)]; FIBB = [(0.135, 0.135), (0.2, 0.1)]
def assign(ob, fib):
    G = [[] for _ in fib]
    for o in ob: G[int(np.argmin([np.abs(o["u"] - np.array(f_)).max() for f_ in fib]))].append(o)
    return G
GA = assign(obA, FIBA); GB = assign(obB, FIBB)
gA = [len(x) for x in GA]; gB = [len(x) for x in GB]
ok = len(obA) == 5 and len(obB) == 5 and gA == [3, 1, 1] and gB == [4, 1]
ok &= all(np.abs(o["u"] - np.array(f_)).max() < 1e-3 for G, fib in ((GA, FIBA), (GB, FIBB)) for g_, f_ in zip(G, fib) for o in g_)
check("MO2 2x3 objects: five = chi(Bl_2 CP^2); fibres 3 + 1 + 1 before the one-count move, 4 + 1 after", ok, f"{gA} -> {gB}")

# MO3
ok = True
for ob, G in ((ob15, groups(ob15)), (ob35, groups(ob35)), (obA, GA), (obB, GB)):
    for g_ in G:
        for i in range(len(g_)):
            for j in range(i + 1, len(g_)): ok &= np.abs(g_[i]["hol"] - g_[j]["hol"]).max() > 1e-2
    W = [complex(o["W"]) for o in ob]; ok &= min(abs(W[i] - W[j]) for i in range(len(W)) for j in range(i + 1, len(W))) > 0
check("MO3 Hom = 0 between distinct objects: distinct fibres are disjoint; on one fibre the holonomies differ; critical values distinct", ok)

# MO4
ok = True
for ob, v in ((ob15, KB.F1(0.2)[0]), (ob35, KB.F1(0.6)[0])):
    p = [o for o in ob if o["type"] == "positive"][0]; V = np.array(v, float)
    c = np.array([complex(x) for x in [p["W"] * 0]]) if False else None
    Qn = p["Qn"]; ok &= np.allclose(Qn.imag, 0, atol=1e-25) and np.all(np.linalg.eigvalsh(Qn.real) > 0)
    for o in ob: ok &= abs(np.linalg.det(o["Qn"])) > 1e-12
for ob in (obA, obB):
    p = [o for o in ob if o["type"] == "positive"][0]; ok &= np.all(np.linalg.eigvalsh(p["Qn"].real) > 0)
    for o in ob: ok &= abs(np.linalg.det(o["Qn"])) > 1e-30
# the identity Q = W Cov_w(v) for the positive object, from the term law
for a in (0.2, 0.6):
    v, lam = KB.F1(a); b, w = KB.positive_crit(v, lam, 1e-12); V = np.array(v, float); C = (V.T * w) @ V
    y = np.exp(b); terms = np.array([1e-12 ** (-l) * np.prod(y ** np.array(vk)) for vk, l in zip(v, lam)]); Q = (V.T * terms) @ V
    ok &= np.allclose(Q / terms.sum(), C, rtol=1e-9) and np.allclose(w @ V, 0, atol=1e-8)
check("MO4 positive object: Q = W Cov_w(v), real positive definite (the Fisher information of the conditioned law); every object Morse", ok)

# MO5
ok = True
for ob in (ob15, ob35, obA, obB):
    for o in ob:
        Q = o["Qn"]; e, dim = FM.clifford_generators(Q)
        ok &= dim == 4 and all(np.abs(e[i] @ e[j] + e[j] @ e[i] - Q[i, j] * np.eye(2)).max() < 1e-9 * max(1, np.abs(Q).max()) for i in range(2) for j in range(2))
check("MO5 Clifford relations e_i e_j + e_j e_i = Q_ij by explicit generators; they generate M_2 (dimension 4) for all 18 objects", ok)

# MO6
ok = True; rows = []
Ts = [mp.mpf(10) ** -k for k in (24, 48, 96)]
for a, want_rank, want_val in ((0.2, 2, None), (1 / 3, 2, None), (0.6, 1, 0.2), (0.8, 1, 0.35)):
    Qs = []
    for T in Ts:
        ob = f1_obj(a, T, dps=260); Qs.append([o for o in ob if o["type"] == "positive"][0]["Qn"].real)
    vals, rk = FM.reduced_rank(Qs, Ts); ok &= rk == want_rank and (want_val is None or abs(max(vals) - want_val) < 5e-3); rows.append((round(a, 3), rk, round(max(vals), 3)))
from pce import datawalls as DW
for (f, g), want_rank, want_val in (((CF.EX_F, CF.EX_G), 2, None), ((CF.EX_F, (Fr(30, 100), Fr(44, 100), Fr(26, 100))), 1, 1 / 200),
                                     (((Fr(2, 5), Fr(3, 5)), (Fr(1, 2), Fr(3, 10), Fr(1, 5))), 1, 1 / 20)):
    ff = [float(x) for x in f]; gf = [float(x) for x in g]; F = CF.facet_cells(ff, gf); V, Lc = CF.cell_functions(ff[0], gf[0], gf[1])
    vk = np.array([V[c] for c in F], float); h, qs, K = DW.balance_lp([V[c] for c in F], [Lc[c] for c in F]); Cs = (vk.T * qs) @ vk   # the reduction: Cov of q*
    rk = int(np.linalg.matrix_rank(Cs, tol=1e-9)); errs = []; Qs = []; Tl = []
    for tau in ("1/100", "1/1000", "1/10000", "1/100000"):
        ob, T = cpl_obj(f, g, tau); Q = [o for o in ob if o["type"] == "positive"][0]["Qn"].real; Qs.append(Q); Tl.append(T); errs.append(np.abs(Q - Cs).max())
    ok &= rk == want_rank and all(errs[i + 1] < errs[i] for i in range(3))
    if want_val is not None:
        lT = [float(mp.log(t)) for t in Tl[-2:]]; ev = [np.sort(np.abs(np.linalg.eigvals(Q)))[0] for Q in Qs[-2:]]
        val = (np.log(ev[1]) - np.log(ev[0])) / (lT[1] - lT[0]); ok &= abs(val - want_val) < 0.25 * want_val
    else: val = 0.0
    rows.append((str(g[2]), rk, round(val, 4), f"{errs[-1]:.1e}"))
check("MO6 the Clifford shadow: rank of Q/W mod T = dim sigma_0; F_1 2, 2, 1, 1 (valuation (3 alpha - 1)/4); 2x3: 2, then 1 after one count (1/200); column example 1 (1/20)",
      ok, f"{rows}")

# MO7
ok = True; al = np.linspace(0.2, 0.6, 81)
def solver(a):
    ob = f1_obj(a, T24, dps=150); return [o["y"] for o in ob], [o["W"] for o in ob]
P, Wt = FM.track(al, solver); ok &= np.min([np.min(np.abs(Wt[:, i] - Wt[:, j]) / np.max(np.abs(Wt[:, [i, j]]), 1)) for i in range(4) for j in range(i + 1, 4)]) > 1e-8
dets = [min(abs(np.linalg.det(o["Qn"])) for o in f1_obj(a, T24, dps=150)) for a in al[::8]]; ok &= min(dets) > 1e-8
import scipy.linalg as sl
Qa = [[o for o in f1_obj(a, T24) if o["type"] == "positive"][0]["Qn"].real for a in (0.2, 0.6)]
A = sl.sqrtm(np.linalg.inv(Qa[1])) @ sl.sqrtm(Qa[0]); ok &= np.allclose(A.T @ Qa[1] @ A, Qa[0], rtol=1e-8)
check("MO7 along alpha in [1/5, 3/5]: no collision, Morse throughout; an explicit real isomorphism of the positive object's Clifford forms", ok,
      f"min |det Q/W| = {min(dets):.1e}")

# MO8, MO9
ok8 = ok9 = True; table = {}
for k in (12, 24, 48):
    T = mp.mpf(10) ** -k; Lg = k * math.log(10); al = np.linspace(0.2, 0.5, 601)
    def solv(a):
        ob = f1_obj(a, T, dps=150); return [o["y"] for o in ob], [o["W"] for o in ob]
    P, Wt = FM.track(al, solv); types = [o["type"] for o in f1_obj(0.2, T, dps=150)]
    ob0 = f1_obj(0.2, T, dps=150); P0 = np.array([[complex(x) for x in o["y"]] for o in ob0])
    from scipy.optimize import linear_sum_assignment
    _, col = linear_sum_assignment(sum(np.abs(P[0][None, :, i] - P0[:, None, i]) / np.abs(P0[:, None, i]) for i in range(2)))
    lab = {col[r]: types[r] for r in range(4)}
    for th in (0.3, 1.0, math.pi / 2):
        ev = FM.stokes_events(Wt, al, th); table[(k, round(th, 2))] = [(Lg * (x - 1 / 3), i, j) for x, i, j in ev]
        ok8 &= len(ev) > 0 and all(abs(x - 1 / 3) < 4 / Lg for x, i, j in ev) and any("real-signed" in (lab[i], lab[j]) for x, i, j in ev)
        if k == 24:                                     # away from the wall there are no Stokes events at these phases
            for lo, hi in ((0.05, 0.2), (0.5, 0.95)):
                al2 = np.linspace(lo, hi, 151); P2, W2 = FM.track(al2, solv); ok8 &= len(FM.stokes_events(W2, al2, th)) == 0
for th in (0.3, 1.0, 1.57):
    ref = sorted(x for x, _, _ in table[(48, th)]);
    for k in (12, 24):
        cur = sorted(x for x, _, _ in table[(k, th)]); ok9 &= len(cur) == len(ref) and all(abs(c - r) <= 0.05 * abs(r) + 0.05 for c, r in zip(cur, ref))
check("MO8 Stokes walls (phases 0.3, 1.0, pi/2) lie within 4/log(1/T) of the outcome wall alpha = 1/3, none far from it", ok8)
check("MO9 log(1/T) (alpha_theta - 1/3) is constant across T = 1e-12, 1e-24, 1e-48: Stokes walls converge to the outcome wall at rate 1/log(1/T)", ok9,
      f"{ {k: [round(x, 2) for x, _, _ in v] for k, v in table.items() if k[0] == 48} }")
print(f"{sum(results)}/{len(results)} passed")
