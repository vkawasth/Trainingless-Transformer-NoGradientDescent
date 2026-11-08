"""Checks for section 12.23 of the companion note (module pce/voigeom.py): the geometry of decisions and of the value of information.
IDENTITY = an exact or numerical instance of a statement proved in the text; CERTIFICATE = an exact computation reported as a result;
NUMERICAL = floating-point (linear programs, iterative proportional fitting), with tolerances stated.
  VG1  NUMERICAL    a decision is identified exactly when the value of resolving the fibre is 0 (3 random actions, 40 points); for act /
                    abstain exactly when tau is outside (L, U)
  VG2  NUMERICAL    the value of identification R: at most the largest width of a utility-difference interval against an action optimal
                    somewhere on the fibre; Lipschitz with constant 2 for act / abstain; never larger after observing more contexts; 0 on
                    rigid faces for any utilities
  VG3  NUMERICAL    act / abstain: R = 2 min((tau - L)+, (U - tau)+) by linear programming, and the randomisation gap is R / 2
  VG4  NUMERICAL    instance information: VOI = sum_z |a_z| - |sum_z a_z| >= 0, equal to the direct Bayes computation, 0 exactly when the
                    signal cannot move the posterior across tau; the committed robust value is >= 0 with pure or mixed policies, and 0 for an
                    uninformative signal when mixed policies are allowed; mutual information is positive while
                    VOI is 0 below the critical accuracy; VOI is largest when the threshold sits at the selected value
  VG5  NUMERICAL    mixing: L is convex, U and the width are concave along segments of data; R is concave where positive along route B
  VG6  NUMERICAL    the staged routes of section 12.21 read through the value of information
  VG7  NUMERICAL    stability: Hausdorff distance (total variation) between fibres along route B against the edit cost, while the
                    dimension jumps; dilation (observing B0 widens the interval of P(A0 = A1) for some outcome, sometimes for every
                    outcome; never on a rigid face); the robust value against the envelope of classical values: with pure policies it can
                    exceed the envelope, by the value of randomising; with mixed policies it lies inside
  VG8  NUMERICAL    Das's square for E = {B0 = z}: committing to the selected law and conditioning it differs from re-selecting after
                    conditioning, because only the contexts containing B0 stay observable; also on rigid faces
"""
import os, sys, random, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import margsec as G
from pce import poisonflip as P
from pce import voigeom as V
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
CELLS = G.CELLS; M4 = np.array(G.M, float); W = np.array([1.0 if c[0] == c[1] else 0.0 for c in CELLS]); TAU = 0.45
E4, _ = P.entries_chsh(); SU = P.signed_unit(E4, P.circuits(E4)); F = G.facets_D()
rng = np.random.default_rng(20261107); UA = V.act_abstain_utilities(TAU)

# VG1
ok = True; nid = 0; n2 = 0
for s in range(40):
    d = M4 @ rng.dirichlet(np.ones(16)); Us = [rng.normal(size=16) * 0.3 + (0.6 if j == 0 else 0) * W for j in range(3)]
    idf = V.identified(d, Us); R = V.regret_value(d, Us); ok &= idf == (R <= 1e-12); nid += idf
    L, U = P.lower_upper(d, W); tau = rng.uniform(L - 0.05, U + 0.05); Ua = V.act_abstain_utilities(tau)
    ok &= V.identified(d, Ua) == (not (L < tau < U)); n2 += not (L < tau < U)
check("VG1 NUMERICAL identified exactly when the value of resolving the fibre is 0; act / abstain identified exactly when tau is outside (L, U)",
      ok, f"40 points with 3 random actions ({nid} identified); 40 random thresholds ({n2} outside the interval)")

# VG2
ok = True; worst = 0.0; mono = 0; nm = 0
for s in range(30):
    d = M4 @ rng.dirichlet(np.ones(16)); Us = [rng.normal(size=16) * 0.3 for _ in range(3)]
    x = G.maxent(d); a0 = int(np.argmax([u @ x for u in Us])); R = V.regret_value(d, Us)
    wid = max(np.subtract(*V.utility_interval(d, Us[a0], Us[b])[::-1]) for b in range(3) if b != a0); ok &= R <= wid + 1e-9
    for C in itertools.combinations(range(4), 3):
        Rs = V.regret_value(d, Us, rows=P.chsh_rows(C)); ok &= Rs >= R - 1e-9; nm += 1; mono += Rs > R + 1e-9
for s in range(40):
    xa, xb = rng.dirichlet(np.ones(16)), rng.dirichlet(np.ones(16)); da = M4 @ xa; db = da + rng.uniform(0.01, 0.4) * (M4 @ xb - da)
    Ra, Rb = V.regret_value(da, UA), V.regret_value(db, UA); c = P.cost(E4, db - da); ok &= abs(Ra - Rb) <= 2 * c + 1e-9; worst = max(worst, abs(Ra - Rb) / (2 * c))
faces_ = sorted(G.faces_D(F), key=lambda V_: sorted(V_)); rr = random.Random(20261108)
F0 = rr.sample([V_ for V_ in faces_ if len(V_) == 8 and G.face_dim(V_) == 7 and G.fibre_dim_face(V_) == 0], 3); rr = random.Random(20261108); rig = []
for V_ in F0:
    wv = [rr.randint(5, 25) for _ in V_]; t = sum(wv); wv = [x * 240 // t for x in wv]; wv[0] += 240 - sum(wv); x = np.zeros(16)
    for k, v in zip(sorted(V_), wv): x[k] = v / 240
    rig.append(x)
for x in rig:
    for s in range(10): ok &= V.regret_value(M4 @ x, [rng.normal(size=16) for _ in range(3)]) <= 1e-9
check("VG2 NUMERICAL R <= the largest width against an action optimal on the fibre; |dR| <= 2 x cost; more contexts never raise R; R = 0 on rigid faces",
      ok, f"30 points x 3 random actions; Lipschitz ratio at most {worst:.3f} over 40 pairs; sub-covers: R larger without a context in {mono}/{nm}; 3 CH-facet laws x 10 utilities: R = 0")

# VG3
ok = True; dev = 0.0
for s in range(40):
    d = M4 @ rng.dirichlet(np.ones(16)); L, U = P.lower_upper(d, W); tau = rng.uniform(L - 0.05, U + 0.05); aa = V.act_abstain(L, U, tau)
    R = V.regret_value(d, V.act_abstain_utilities(tau)); ok &= abs(R - aa["R"]) < 1e-9
    gap = V.mixed_maxmin(L, U, tau) - aa["pure"]; dev = max(dev, abs(gap - aa["R"] / 2)); ok &= abs(gap - aa["R"] / 2) < 1e-3
check("VG3 NUMERICAL act / abstain: R = 2 min((tau - L)+, (U - tau)+) (LP, 1e-9) and the randomisation gap is R / 2 (grid of 2001 mixtures)", ok,
      f"40 random points and thresholds; largest gap deviation {dev:.1e}")

# VG4
ok = True; below = []
for s in range(200):
    phi, tau, al = rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95), rng.uniform(0.5, 1.0)
    a1, a0 = V._a(phi, tau, al); v = V.signal_voi(phi, tau, al)
    pz1 = al * phi + (1 - al) * (1 - phi); post = [al * phi / pz1, (1 - al) * phi / (1 - pz1)]                       # direct Bayes computation
    direct = pz1 * abs(post[0] - tau) + (1 - pz1) * abs(post[1] - tau) - abs(phi - tau)
    ok &= abs(v - direct) < 1e-12 and v >= -1e-15 and ((v > 1e-12) == ((post[0] - tau) * (post[1] - tau) < 0))
    L_, U_ = sorted(rng.uniform(0.05, 0.95, 2)); ok &= V.robust_signal_voi(L_, U_, tau, al) >= -1e-12 and V.robust_signal_voi_mixed(L_, U_, tau, al, 401) >= -1e-12
    if s < 40: ok &= abs(V.robust_signal_voi_mixed(L_, U_, tau, 0.5, 401)) < 1e-12
for phi, tau in ((0.3, 0.45), (0.516, 0.45), (0.7, 0.45)):
    a_ = V.alpha_star(phi, tau); al = (0.5 + a_) / 2; below.append((phi, round(a_, 4), round(V.mutual_info(phi, al), 4), V.signal_voi(phi, tau, al)))
    ok &= V.mutual_info(phi, al) > 0 and V.signal_voi(phi, tau, al) <= 1e-12
    taus = np.linspace(0.01, 0.99, 981); best = taus[int(np.argmax([V.signal_voi(phi, t_, 0.8) for t_ in taus]))]; ok &= abs(best - phi) < 0.002
check("VG4 NUMERICAL instance VOI = sum|a_z| - |sum a_z| >= 0 (= direct Bayes), 0 exactly when no posterior crosses tau; committed robust VOI >= 0; "
      "mutual information > 0 while VOI = 0 below the critical accuracy; VOI peaks with the threshold at the selected value", ok,
      "200 random (phi, tau, alpha); (phi, critical accuracy, mutual information halfway below it, VOI there): " + "; ".join(str(b) for b in below))

# VG5 and VG6: the staged routes
d0 = M4 @ rig[0]; pA = M4[:, 6]; pB = M4[:, 10]; N = 12000
ok = True; viol = 0
for s in range(40):
    da, db = M4 @ rng.dirichlet(np.ones(16)), M4 @ rng.dirichlet(np.ones(16)); lam = rng.uniform(0.1, 0.9); dm = lam * da + (1 - lam) * db
    (La, Ua_), (Lb, Ub), (Lm, Um) = P.lower_upper(da, W), P.lower_upper(db, W), P.lower_upper(dm, W)
    c1 = Lm <= lam * La + (1 - lam) * Lb + 1e-9; c2 = Um >= lam * Ua_ + (1 - lam) * Ub - 1e-9; c3 = (Um - Lm) >= lam * (Ua_ - La) + (1 - lam) * (Ub - Lb) - 1e-9
    ok &= c1 and c2 and c3; viol += not (c1 and c2 and c3)
qs = np.linspace(0, 0.6, 241); Rq = [V.act_abstain(*P.lower_upper((1 - q) * d0 + q * pB, W), TAU)["R"] for q in qs]
sec = [Rq[i - 1] - 2 * Rq[i] + Rq[i + 1] for i in range(1, len(qs) - 1) if Rq[i - 1] > 1e-9 and Rq[i] > 1e-9 and Rq[i + 1] > 1e-9]; ok &= max(sec) <= 1e-9
check("VG5 NUMERICAL L convex, U and the width concave along segments of data (40 random segments); R concave where positive along route B",
      ok, f"{viol} violations; largest second difference of R on its support {max(sec):.1e} (over {len(sec)} interior points)")
ks = np.r_[0, np.linspace(50, 13000, 260)]; rowsA, rowsB = [], []
for k in ks:
    q = k / (N + k)
    for p, rows in ((pA, rowsA), (pB, rowsB)):
        d = (1 - q) * d0 + q * p; L, U = P.lower_upper(d, W); me = float(W @ G.maxent(d)); aa = V.act_abstain(L, U, TAU)
        rows.append((k, L, U, aa["R"], me, V.signal_voi(me, TAU, 0.8), V.robust_signal_voi(L, U, TAU, 0.8)))
rA, rB = np.array(rowsA), np.array(rowsB)
flipA = rA[np.argmax(rA[:, 4] < TAU), 0]; supB = rB[rB[:, 3] > 1e-9, 0]; pk = rB[np.argmax(rB[:, 3])]
ok = rA[:, 3].max() <= 1e-12 and supB.min() > 1222 - 60 and supB.max() < 11890 + 60
iA = int(np.argmax(rA[:, 5])); iB = int(np.argmax(rB[:, 5]))
check("VG6 NUMERICAL the staged routes read through the value of information", ok,
      f"route A: R = 0 at all {len(ks)} stages although the decision flips at about {flipA:.0f} units; instance VOI (accuracy 0.8) peaks at "
      f"{rA[iA, 5]:.3f} at {rA[iA, 0]:.0f} units (maxent value {rA[iA, 4]:.3f}) | route B: R > 0 from {supB.min():.0f} to {supB.max():.0f} units, "
      f"peak {pk[3]:.3f} at {pk[0]:.0f} units; instance VOI peaks at {rB[iB, 5]:.3f} at {rB[iB, 0]:.0f} units")
# VG7
hd = []
for k in (25, 100, 600, 2400):
    q = k / (N + k); d = (1 - q) * d0 + q * pB; hd.append((k, round(V.hausdorff_tv(d0, d), 4), round(P.cost(E4, d - d0), 4), G.fibre_dim(d)))
okh = all(h <= 2 * c + 1e-9 for _, h, c, _ in hd) and hd[0][3] == 7
nd = strict = 0; ex = None; rngd = np.random.default_rng(3)
B0 = lambda z: (np.array([1.0 if (c[0] == c[1] and c[2] == z) else 0.0 for c in CELLS]), np.array([1.0 if c[2] == z else 0.0 for c in CELLS]))
for s in range(300):
    x = rngd.dirichlet(np.ones(16) * rngd.choice([0.3, 1, 3])); d = M4 @ x; L, U = P.lower_upper(d, W)
    ci = [V.conditional_interval(d, B0(z)[0], B0(z)[1], float(B0(z)[1] @ x)) for z in (0, 1)]
    nd += any(hi - lo > U - L + 1e-6 for lo, hi in ci)
    sd = all(lo <= L + 1e-9 and hi >= U - 1e-9 for lo, hi in ci) and any(lo < L - 1e-6 or hi > U + 1e-6 for lo, hi in ci); strict += sd
    if sd and ex is None: ex = (round(L, 3), round(U, 3), [(round(a_, 3), round(b_, 3)) for a_, b_ in ci])
rig_ci = []
for x in rig:
    d = M4 @ x; rig_ci += [np.subtract(*V.conditional_interval(d, B0(z)[0], B0(z)[1], float(B0(z)[1] @ x))[::-1]) for z in (0, 1)]
env = []
for k in (600, 2400, 6000, 11000):
    q = k / (N + k); d = (1 - q) * d0 + q * pB; L, U = P.lower_upper(d, W); lo, hi = V.signal_envelope(L, U, TAU, 0.8)
    pure, mixd, gap0 = V.robust_signal_voi(L, U, TAU, 0.8), V.robust_signal_voi_mixed(L, U, TAU, 0.8), V.act_abstain(L, U, TAU)["gap"]
    env.append((k, round(lo, 4), round(hi, 4), round(pure, 4), round(mixd, 4), round(gap0, 4)))
okd = nd == 300 and strict >= 1 and max(rig_ci) < 1e-9 and any(r_[3] > r_[2] + 1e-6 for r_ in env)
okd &= all(r_[1] - 1e-3 <= r_[4] <= r_[2] + 1e-3 and abs(r_[3] - r_[4] - r_[5]) < 1e-3 for r_ in env)
check("VG7 NUMERICAL stability profile: fibres move little in Hausdorff distance while their dimension jumps; dilation; robust value outside the envelope",
      okh and okd, f"route B (units, Hausdorff TV to the start, edit cost, fibre dim): {hd}; observing B0 widens the interval for some outcome at "
      f"{nd}/300 random points and for every outcome (strict dilation) at {strict}/300, e.g. prior {ex[:2]} -> {ex[2]}; on the 3 rigid faces the conditional "
      f"intervals have width {max(rig_ci):.0e}; (units, envelope low, envelope high, pure-policy robust value, mixed-policy robust value, "
      f"value of randomising without the signal): {env}; the pure-policy value leaves the envelope by exactly the value of randomising, the mixed one stays inside")
# VG8
gaps = []; rg = np.random.default_rng(4)
for s in range(20):
    d = M4 @ rg.dirichlet(np.ones(16)); gaps += [V.das_gap(d, z) for z in (0, 1)]
rgaps = [V.das_gap(M4 @ x, z, sel=lambda d_, x_=x: x_) for x in rig for z in (0, 1)]
check("VG8 NUMERICAL Das's square fails for re-selection after conditioning on B0 = z (commit-then-condition differs), also on rigid faces",
      min(g[0] for g in gaps) > 1e-3 and min(g[0] for g in rgaps) > 1e-3,
      f"20 random points x 2 outcomes: TV between committed and re-selected laws {min(g[0] for g in gaps):.3f}-{max(g[0] for g in gaps):.3f}, "
      f"decision value differs by up to {max(abs(g[1] - g[2]) for g in gaps):.3f}; rigid faces (the unique law committed): TV {min(g[0] for g in rgaps):.3f}-{max(g[0] for g in rgaps):.3f}, "
      f"decision value differs by up to {max(abs(g[1] - g[2]) for g in rgaps):.3f}")
print(f"{sum(results)}/{len(results)} passed")
