"""Checks for section 12.21 of the companion note (module pce/poisonflip.py): the geometry of poisoning.
IDENTITY = an exact or numerical instance of a statement proved in the text; CERTIFICATE = an exact computation reported as a result;
NUMERICAL = floating-point (linear programs, iterative proportional fitting), with tolerances stated.
  PG1  CERTIFICATE  Lipschitz constants of L and U in the edit cost, max over dual vertices and unit circuits of |y.c|: 1 and 1 for
                    P(A0 = A1), 1/2 and 1/2 for P(A0 = 0, A1 = 0)
  PG2  NUMERICAL    certified radii: c_some >= (L - tau) / Lam_L and c_all >= (U - tau) / Lam_U at random and face points; how tight
  PG3  NUMERICAL    the width U - L is Lipschitz with constant Lam_L + Lam_U, while the fibre dimension jumps (0 to 7 for an edit of cost 1e-3)
  PG4  NUMERICAL    the distance to the contextuality boundary has a closed form (min over CH facets of slack / largest circuit rate) and
                    equals the linear-programming distance
  PG5  NUMERICAL    poisoning in stages from a rigid point: on the facet the fibre stays a point; off the facet it opens, some rule flips at
                    the same share but every rule needs about five times as much; a PR-box overreach makes the data contextual at once,
                    and from an interior point exactly where the facet slacks predict; the fibre is empty exactly when the contextual
                    fraction is positive
  PG6  NUMERICAL    continuity: on straight data paths the maximum-entropy law and a fixed-objective vertex rule (unique optimum) both move
                    in small steps; a selected law jumps only with ties or a data-dependent rule (Berge), and L, U never jump
"""
import os, sys, random, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import margsec as G
from pce import poisonflip as P
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
CELLS = G.CELLS; M4 = np.array(G.M, float); W = np.array([1.0 if c[0] == c[1] else 0.0 for c in CELLS]); V2 = np.array([1.0 if c[2] == c[3] else 0.0 for c in CELLS])
E4, _ = P.entries_chsh(); SU = P.signed_unit(E4, P.circuits(E4)); F = G.facets_D(); DVU = P.dual_vertices(W); DVL = P.dual_lower(W)

# PG1
lam = (P.lipschitz(DVL, SU), P.lipschitz(DVU, SU)); lam2 = (P.lipschitz(P.dual_lower(G.W_A0A1), SU), P.lipschitz(P.dual_vertices(G.W_A0A1), SU))
check("PG1 CERTIFICATE Lipschitz constants of the identified interval's ends in the edit cost (dual vertices x unit circuits)",
      abs(lam[0] - 1) < 1e-12 and abs(lam[1] - 1) < 1e-12 and abs(lam2[0] - 0.5) < 1e-12 and abs(lam2[1] - 0.5) < 1e-12,
      f"P(A0=A1): Lam_L = {lam[0]}, Lam_U = {lam[1]} ({len(DVL)}, {len(DVU)} dual vertices, {len(SU)} unit circuits); P(A0=0,A1=0): {lam2}")

# PG2
rng = np.random.default_rng(11); ok = True; rs, ra = [], []
faces_ = sorted(G.faces_D(F), key=lambda V: sorted(V)); rr = random.Random(20261108)
fam = {nm: rr.sample([V for V in faces_ if len(V) == 8 and G.face_dim(V) == dm and G.fibre_dim_face(V) == fb], 3) for nm, (dm, fb) in (("F0", (7, 0)), ("F1", (6, 1)), ("F2", (5, 2)))}
rr = random.Random(20261108); pts = []
for nm in ("F0", "F1", "F2"):
    for V in fam[nm]:
        wv = [rr.randint(5, 25) for _ in V]; t = sum(wv); wv = [x * 240 // t for x in wv]; wv[0] += 240 - sum(wv); x = np.zeros(16)
        for k, v in zip(sorted(V), wv): x[k] = v / 240
        pts.append(x)
pts += [rng.dirichlet(np.ones(16)) for _ in range(30)]
for x in pts:
    d = M4 @ x; L, U = P.lower_upper(d, W)
    for g in (0.02, 0.05):
        tau = L - g
        if tau <= 0: continue
        cs = P.flip_some(d, W, tau, E4)[0]; ca = P.flip_all(d, W, tau, E4, DVU)[0]
        ok &= cs >= (L - tau) / lam[0] - 1e-9 and ca >= (U - tau) / lam[1] - 1e-9; rs.append(cs / ((L - tau) / lam[0])); ra.append(ca / ((U - tau) / lam[1]))
check("PG2 NUMERICAL certified radii c_some >= (L - tau)/Lam_L and c_all >= (U - tau)/Lam_U (gaps 0.02, 0.05; 39 points)", ok,
      f"{len(rs)} cases; actual / bound for some rule: min {min(rs):.3f}, median {np.median(rs):.3f}, max {max(rs):.3f}; tight (ratio < 1.001) in "
      f"{sum(r_ < 1.001 for r_ in rs)}; for every rule: min {min(ra):.3f}, median {np.median(ra):.3f}, max {max(ra):.3f}")

# the rigid start and the two routes (used by PG3, PG5, PG6)
V0 = fam["F0"][0]; x0 = pts[0]; d0 = M4 @ x0; TAU = 0.45; N = 12000
qA, RA = P.poison_mix(d0, W, TAU, DVU); jA = int(np.argmax(RA)); jB = 10; pA = M4[:, jA]; pB = M4[:, jB]
fac = [f for f in F if set(V0) == set(f[2])][0]; h = np.array([float(v) for v in fac[0]]); h0 = float(fac[1])
sg = min((s_ for s_ in itertools.product((1, -1), repeat=4) if np.prod(s_) == -1), key=lambda s_: h @ P.pr_box(s_) + h0); PR = P.pr_box(sg)

# PG3
ok = True; worst = 0.0
for _ in range(40):
    xa, xb = rng.dirichlet(np.ones(16)), rng.dirichlet(np.ones(16)); da = M4 @ xa; db = da + rng.uniform(0.01, 0.3) * (M4 @ xb - da)
    La, Ua = P.lower_upper(da, W); Lb, Ub = P.lower_upper(db, W); c = P.cost(E4, db - da)
    ok &= abs((Ub - Lb) - (Ua - La)) <= (lam[0] + lam[1]) * c + 1e-9; worst = max(worst, abs((Ub - Lb) - (Ua - La)) / ((lam[0] + lam[1]) * c))
dj = (1 - 1e-3) * d0 + 1e-3 * pB; cj = P.cost(E4, dj - d0); Lj, Uj = P.lower_upper(dj, W)
ok &= G.fibre_dim(d0) == 0 and G.fibre_dim(dj) == 7 and Uj - Lj <= 2 * cj + 1e-12
check("PG3 NUMERICAL the width is Lipschitz (constant Lam_L + Lam_U = 2) while the fibre dimension jumps", ok,
      f"40 random pairs, largest ratio to the bound {worst:.3f}; from the rigid start, an edit of cost {cj:.2e} toward reading {CELLS[jB]}: fibre dimension 0 -> 7, width {Uj - Lj:.2e}")

# PG4
ok = True; dev = 0.0
from scipy.optimize import linprog
for _ in range(25):
    d = M4 @ rng.dirichlet(np.ones(16)); best = np.inf
    for c, c0, V in F:
        if len(V) != 8: continue
        hh = np.array([float(v) for v in c]); sl = hh @ d + float(c0)
        r = linprog(np.r_[np.zeros(8), 0.5 * np.ones(16)], A_ub=np.vstack([np.c_[E4, -np.eye(16)], np.c_[-E4, -np.eye(16)], np.r_[hh, np.zeros(16)][None]]),
                    b_ub=np.r_[np.zeros(32), -sl], bounds=[(None, None)] * 8 + [(0, None)] * 16, method="highs")
        best = min(best, r.fun)
    cf = P.ctx_radius(d, F, SU); dev = max(dev, abs(cf - best)); ok &= abs(cf - best) < 1e-8
check("PG4 NUMERICAL distance to the contextuality boundary: closed form = linear program", ok, f"25 random interior points, largest difference {dev:.1e}")

# PG5
stagesA = [0, 600, 1200, 1800]; stagesB = [0, 600, 1200, 2400, 4800, 9600, 12000]
def stage(p, k):
    q = k / (N + k); d = (1 - q) * d0 + q * p; L, U = P.lower_upper(d, W)
    return dict(k=k, q=q, L=L, U=U, dim=G.fibre_dim(d), me=float(W @ G.maxent(d)), rad=P.ctx_radius(d, F, SU), cost=P.cost(E4, d - d0))
SA = [stage(pA, k) for k in stagesA]; SB = [stage(pB, k) for k in stagesB]
def first_q(p, which):
    lo, hi = 0.0, 0.9
    f = lambda q: P.lower_upper((1 - q) * d0 + q * p, W)[0 if which == "some" else 1] <= TAU
    if not f(hi): return None
    for _ in range(60):
        mid = (lo + hi) / 2; lo, hi = (lo, mid) if f(mid) else (mid, hi)
    return hi
qAs, qAa, qBs, qBa = first_q(pA, "some"), first_q(pA, "all"), first_q(pB, "some"), first_q(pB, "all")
ok = all(s["dim"] == 0 and s["U"] - s["L"] < 1e-9 for s in SA) and all(s["dim"] == 7 for s in SB[1:]) and abs(qAs - qAa) < 1e-6 and qBa > 4 * qBs
# overreach: mixing the PR box that violates this facet
emp0 = []; agree = 0; tot = 0
for q in np.linspace(0, 0.3, 31):
    d = (1 - q) * d0 + q * PR; cf = P.contextual_fraction(d, E4); empty = P.lower_upper(d, W) is None; agree += (cf > 1e-9) == empty; tot += 1
    if q > 0: emp0.append(empty)
dI = SB[3]; dIv = (1 - dI["q"]) * d0 + dI["q"] * pB
pred = P.line_cross(dIv, PR - dIv, F); obs = None
for q in np.linspace(0, 0.3, 3001):
    if P.contextual_fraction(dIv + q * (PR - dIv), E4) > 1e-9: obs = q; break
ok &= all(emp0) and agree == tot and obs is not None and abs(obs - pred) < 2e-4
check("PG5 NUMERICAL poisoning in stages from a rigid point (route A on the facet, route B off it, then a PR-box overreach)", ok,
      "route A (reading " + str(CELLS[jA]) + "): " + "; ".join(f"k={s['k']}: L=U={s['L']:.3f}, dim {s['dim']}" for s in SA)
      + " | route B (reading " + str(CELLS[jB]) + "): " + "; ".join(f"k={s['k']}: [{s['L']:.3f}, {s['U']:.3f}], dim {s['dim']}, maxent {s['me']:.3f}, radius {s['rad']:.3f}" for s in SB)
      + f" | share for some / every rule to flip: route A {qAs:.4f} / {qAa:.4f}, route B {qBs:.4f} / {qBa:.4f}"
      + f" | PR overreach from the rigid start: fibre empty for every q > 0 ({sum(emp0)}/{len(emp0)}), empty iff contextual fraction > 0 at {agree}/{tot};"
      + f" from route B at k = {dI['k']}: contextual from q = {obs:.4f}, facet slacks predict {pred:.4f}")

# PG6
qs = np.linspace(0, 1, 101); cvec = np.random.default_rng(5).normal(size=16); rng6 = np.random.default_rng(6); JM, JV = [], []
paths = [(d0, pB)] + [(M4 @ rng6.dirichlet(np.ones(16)), M4 @ rng6.dirichlet(np.ones(16))) for _ in range(19)]
for da, db in paths:
    prevm = prevv = None; jm = jv = 0.0
    for q in qs:
        d = (1 - q) * da + q * db; xm = G.maxent(d); xv = P.vertex_rule(d, cvec)
        if prevm is not None: jm = max(jm, 0.5 * np.abs(xm - prevm).sum()); jv = max(jv, 0.5 * np.abs(xv - prevv).sum())
        prevm, prevv = xm, xv
    JM.append(jm); JV.append(jv)
check("PG6 NUMERICAL continuity of rules along straight data paths (101 steps): maximum entropy and a fixed-objective vertex rule (unique "
      "optimum) both move in small steps, so a selected law cannot jump without ties or a data-dependent rule", max(JM) < 0.05 and max(JV) < 0.05,
      f"20 paths (route B and 19 random): largest one-step change (TV of the joint law) for maximum entropy {max(JM):.4f}, for the vertex rule {max(JV):.4f}")
print(f"{sum(results)}/{len(results)} passed")
