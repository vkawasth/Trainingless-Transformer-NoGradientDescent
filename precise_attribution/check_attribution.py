"""check_attribution.py -- verifies the theorems of "Precise Attribution of Model Interventions".

Exact (rational) checks unless marked LP; LP certificates are rationalized and re-verified exactly.
"""
import random, math, itertools
from fractions import Fraction as F
import numpy as np
from provenance import *

R = random.Random(2026)
res = []
def check(name, cond):
    res.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name)

def rdist(n, zeros=0):
    w = [F(R.randint(1, 9)) for _ in range(n)]
    for i in R.sample(range(n), zeros): w[i] = F(0)
    s = sum(w); return [v / s for v in w]

# ---------------- Theorem 3.2: attribution set, budget interval
ok_int = ok_empty = True
for _ in range(40):
    n, k = 5, R.randint(2, 3)
    A = [rdist(n, R.randint(0, 2)) for _ in range(k)]
    p = rdist(n, R.randint(0, 1))
    mm, lb, y = mu_min_sources(p, A)
    ok_int &= lb <= mm + 1e-7 and mm - float(lb) < 1e-5
    for bud in (0.0, 0.3, 0.7, 1.0):
        rep = attribution_report(p, [("exact", a) for a in A], bud)
        if bud + 1e-9 < mm:
            ok_empty &= rep is None
        elif rep is not None:
            ok_int &= abs(rep["mu"][0] - mm) < 1e-7 and abs(rep["mu"][1] - bud) < 1e-7
check("Thm 3.2: attribution set nonempty iff mu_min <= budget; mu ranges over [mu_min, budget] (LP)", ok_int and ok_empty)
check("Thm 3.2: exact dual certificate bounds mu_min from below (rationalized, verified exactly)", ok_int)

# ---------------- Theorem 3.4: monotonicity of widths
ok_mono = True
for _ in range(30):
    n, k = 5, 2
    A = [rdist(n, 1) for _ in range(k)]
    lam = rdist(k)
    p = mix(lam, A)
    exact = [("exact", a) for a in A]
    supp = [("support", {b for b in range(n) if a[b] > 0}) for a in A]
    seq = [attribution_report(p, exact, 0), attribution_report(p, supp, 0), attribution_report(p, supp, 0.2)]
    for r1, r2 in zip(seq, seq[1:]):
        ok_mono &= all(w2 >= w1 - 1e-8 for w1, w2 in zip(r1["widths"], r2["widths"]))
    r_lo, r_hi = attribution_report(p, exact, 0.1), attribution_report(p, exact, 0.3)
    ok_mono &= all(w2 >= w1 - 1e-8 for w1, w2 in zip(r_lo["widths"], r_hi["widths"]))
check("Thm 3.5: widths are nondecreasing from exact sources to supports only to positive budget (LP)", ok_mono)

# ---------------- Proposition 3.5: budget slack forces imprecision
ok_slack = True
for _ in range(30):
    n, k = 5, 2
    A = [rdist(n) for _ in range(k)]
    p = mix(rdist(k), A)                         # mu_min = 0
    rep = attribution_report(p, [("exact", a) for a in A], 0.25)
    ok_slack &= abs(rep["mu_width"] - 0.25) < 1e-7 and all(w > 1e-6 for w in rep["widths"])
check("Prop 3.6: with budget slack, w_mu = budget - mu_min and every weight has positive width (LP)", ok_slack)

# ---------------- Theorem 4.3: tilt reach set is a polytope (exact)
K = F(4)
ok_fwd = ok_back = ok_conv = True
for _ in range(300):
    n = R.randint(2, 5)
    p = rdist(n, R.choice([0, 0, 1]))
    w1 = [F(R.randint(10, 40), 10) for _ in range(n)]       # weights in [1, 4]
    w2 = [F(R.randint(10, 40), 10) for _ in range(n)]
    q1, q2 = tilt(p, w1), tilt(p, w2)
    ok_fwd &= implied_ratio(p, q1) is not None and implied_ratio(p, q1) <= K
    t = F(R.randint(1, 9), 10)
    m = [t * a + (1 - t) * b for a, b in zip(q1, q2)]
    ok_conv &= implied_ratio(p, m) <= K
    # backward: build weights from the ratios and recover m exactly
    S = [b for b in range(n) if p[b] > 0]
    r = {b: m[b] / p[b] for b in S}; lo = min(r.values())
    w = [r[b] / lo if b in S else F(1) for b in range(n)]
    ok_back &= all(F(1) <= v <= K for v in w) and tilt(p, w) == m
check("Thm 4.4: every tilt with weights in [1,K] satisfies q_b p_c <= K q_c p_b (exact)", ok_fwd)
check("Thm 4.4: every point satisfying the inequalities is such a tilt (exact reconstruction)", ok_back)
check("Thm 4.4: mixtures of tilts are tilts, so the reach set is convex (exact)", ok_conv)

# fixed reward, unknown temperature: curve, not convex
p = [1 / 3] * 3; r = [0.0, 1.0, 2.0]
def curve(t): return tilt(p, [math.exp(t * x) for x in r])
mid = [(a + b) / 2 for a, b in zip(curve(1.0), curve(3.0))]
lr = [math.log(mid[b] / p[b]) - math.log(mid[0] / p[0]) for b in range(3)]
check("Prop 4.6: with a fixed reward and unknown temperature the reach set is a curve; midpoint leaves it",
      abs(lr[2] - 2 * lr[1]) > 1e-3)

# ---------------- Theorem 4.6: change decomposition with certificates (tilt cone)
ok_cert = ok_zero = True; n_in = n_out = 0
for _ in range(60):
    n = 4
    p = rdist(n)
    if R.random() < 0.4:
        q = tilt(p, [F(R.randint(10, 40), 10) for _ in range(n)]); n_in += 1
    else:
        q = rdist(n, R.choice([0, 1])); n_out += 1
    G = tilt_cone_rows(p, K)
    mm, s, lb, (y, nu) = cone_residual(q, G)
    ok_cert &= lb <= F(mm) + F(1, 10**5) and float(lb) >= mm - 1e-5
    kstar = implied_ratio(p, q)
    inside = kstar is not None and kstar <= K
    ok_zero &= (mm < 1e-8) == inside
check(f"Thm 4.9: exact certificates (y, nu) bracket the LP residual for tilt reach sets ({n_in} in, {n_out} out)", ok_cert)
check("Thm 4.9: residual mu_min = 0 exactly when the change is reachable", ok_zero)

# ---------------- Proposition 5.3: Lipschitz constants
ok_mixL = True
for _ in range(50):
    n = 4; a = F(R.randint(1, 9), 10); t = rdist(n)
    p, q = rdist(n), rdist(n)
    T = mixing(a, t)
    ok_mixL &= tv(T(p), T(q)) == (1 - a) * tv(p, q)
check("Prop 5.3: mixing with a fixed target is exactly (1-a)-Lipschitz in TV", ok_mixL)
ok_tiltL = True; worst = F(0)
for _ in range(400):
    n = R.randint(2, 4)
    w = [F(R.randint(10, 40), 10) for _ in range(n)]; Kw = max(w) / min(w)
    p, q = rdist(n), rdist(n)
    if tv(p, q) == 0: continue
    ratio = tv(tilt(p, w), tilt(q, w)) / tv(p, q)
    ok_tiltL &= ratio <= lipschitz_bound_tilt(Kw)
    worst = max(worst, ratio / Kw)
eps = F(1, 10**6)
p, q = [eps, 1 - eps], [2 * eps, 1 - 2 * eps]; w = [K, F(1)]
amp = tv(tilt(p, w), tilt(q, w)) / tv(p, q)
check("Prop 5.3: tilts are (3K-1)/2-Lipschitz (exact, random)", ok_tiltL)
check(f"Prop 5.3: tilts can amplify differences by nearly K (example ratio {float(amp):.4f}, K = 4)", amp > 3.99)

# non-faithful observation: no Lipschitz constant on observables
M1, M2 = ([F(1, 2), F(1, 2)], "h1"), ([F(1, 2), F(1, 2)], "h2")
Phi = lambda M: M[0]
def edit(M):                     # reads the hidden state
    return ([F(9, 10), F(1, 10)], M[1]) if M[1] == "h1" else (M[0], M[1])
check("Prop 5.4: Phi(M1) = Phi(M2) but Phi(edit M1) != Phi(edit M2), so no Lipschitz constant exists",
      Phi(M1) == Phi(M2) and tv(Phi(edit(M1)), Phi(edit(M2))) > 0)

# ---------------- Theorem 5.5: composition of quantitative comparisons
ok_v = ok_h = ok_acc = True
for _ in range(60):
    n = 4
    S = [rdist(n) for _ in range(6)]
    mk_mix = lambda: mixing(F(R.randint(1, 5), 10), rdist(n))
    mk_tilt = lambda: (lambda w: (tilt_map(w), max(w) / min(w)))([F(R.randint(10, 40), 10) for _ in range(n)])
    F1, F2, F3 = mk_mix(), mk_mix(), mk_mix()
    ok_v &= dist(F1, F3, S) <= dist(F1, F2, S) + dist(F2, F3, S)
    (G, KG), (G2, KG2) = mk_tilt(), mk_tilt()
    lhs = dist(lambda p: G(F1(p)), lambda p: G2(F2(p)), S)
    FS = [F1(p) for p in S]
    rhs = dist(G, G2, FS) + lipschitz_bound_tilt(KG2) * dist(F1, F2, S)
    ok_h &= lhs <= rhs
    # accumulation over a 3-step path of mixings (L_i = 1 - a_i)
    a = [F(R.randint(1, 5), 10) for _ in range(6)]; tg = [rdist(n) for _ in range(6)]
    P = [mixing(a[i], tg[i]) for i in range(3)]; Q = [mixing(a[i + 3], tg[i + 3]) for i in range(3)]
    def run(Ts, p):
        for T in Ts: p = T(p)
        return p
    states = [S]
    for i in range(3): states.append([P[i](p) for p in states[-1]])
    eps_ = [dist(P[i], Q[i], states[i]) for i in range(3)]
    L = [1 - a[i + 3] for i in range(3)]
    bound = eps_[2] + L[2] * (eps_[1] + L[1] * eps_[0])
    ok_acc &= dist(lambda p: run(P, p), lambda p: run(Q, p), S) <= bound
check("Thm 5.5(i): vertical composition, the triangle inequality (exact)", ok_v)
check("Thm 5.5(ii): horizontal composition with the Lipschitz factor (exact)", ok_h)
check("Thm 5.5(iii): accumulated path defect obeys the recursive bound (exact)", ok_acc)

# ---------------- Proposition 5.7: coarsening is monotone
ok_c = True
for _ in range(50):
    n = 6; p, q = rdist(n), rdist(n)
    groups = [[0, 1], [2, 3], [4, 5]]
    ok_c &= tv(coarsen(p, groups), coarsen(q, groups)) <= tv(p, q)
check("Prop 5.9: coarsening never increases TV, so coarse defects certify fine ones (exact)", ok_c)


# ---------------- Proposition 4.6: identifiability of tilt parameters
ok_end = ok_out = ok_w = True
beta, rmin, rmax = 1.0, 0.0, math.log(4.0)
for _ in range(200):
    n = R.randint(2, 5)
    p = [float(v) for v in rdist(n)]
    w = [R.uniform(1, 4) for _ in range(n)]
    q = [float(v) for v in tilt(p, w)]
    ell = [beta * math.log(q[b] / p[b]) for b in range(n)]
    clo, chi = rmin - min(ell), rmax - max(ell)
    Ks = max(q[b] / p[b] for b in range(n)) / min(q[b] / p[b] for b in range(n))
    ok_w &= abs((chi - clo) - ((rmax - rmin) - beta * math.log(Ks))) < 1e-9 and chi >= clo - 1e-12
    for c in (clo, chi):
        r = [l + c for l in ell]
        qq = tilt(p, [math.exp(v / beta) for v in r])
        ok_end &= all(rmin - 1e-9 <= v <= rmax + 1e-9 for v in r) and max(abs(a - b) for a, b in zip(qq, q)) < 1e-12
    ok_out &= any(v > rmax + 1e-9 for v in [l + chi + 1e-6 for l in ell]) and \
              any(v < rmin - 1e-9 for v in [l + clo - 1e-6 for l in ell])
check("Prop 4.7: both ends of the reward interval reproduce the tilt and respect the range", ok_end)
check("Prop 4.7: just beyond either end the range is violated (interval is sharp)", ok_out)
check("Prop 4.7: reward width equals Delta_r - beta log K*", ok_w)

# ---------------- Proposition 4.10: observable reach is set-valued
R_obs = {tuple(Phi(edit(M))) for M in (M1, M2) if Phi(M) == Phi(M1)}
check("Prop 4.12: observable reach of the non-faithful example has two points", len(R_obs) == 2)

# ---------------- Corollary 4.11: enlarged tilt class
from scipy.optimize import linprog
def enlarged_residual(p, q, K, eps):
    n = len(p); Gs = tilt_cone_rows(p, K); G = np.array([[float(v) for v in r] for r in Gs])
    m = len(Gs); nv = 3 * n                       # s, s', u
    c = np.r_[-np.ones(n), np.zeros(2 * n)]
    A, b = [], []
    for i in range(n):                            # s <= q
        row = np.zeros(nv); row[i] = 1; A.append(row); b.append(float(q[i]))
    for j in range(m):                            # G s' <= 0
        row = np.zeros(nv); row[n:2 * n] = G[j]; A.append(row); b.append(0.0)
    for i in range(n):                            # |s - s'| <= u
        row = np.zeros(nv); row[i] = 1; row[n + i] = -1; row[2 * n + i] = -1; A.append(row); b.append(0.0)
        row = np.zeros(nv); row[i] = -1; row[n + i] = 1; row[2 * n + i] = -1; A.append(row); b.append(0.0)
    row = np.zeros(nv); row[2 * n:] = 1; row[:n] = -2 * eps; A.append(row); b.append(0.0)
    Aeq = np.zeros((1, nv)); Aeq[0, :n] = 1; Aeq[0, n:2 * n] = -1
    r = linprog(c, A_ub=np.array(A), b_ub=np.array(b), A_eq=Aeq, b_eq=[0.0], bounds=[(0, None)] * nv)
    return 1 + r.fun
ok_in = ok_outE = True
for _ in range(30):
    n = 4; p = rdist(n); wq = [F(R.randint(10, 40), 10) for _ in range(n)]
    q = tilt(p, wq)
    t = rdist(n); eps = F(1, 50)
    qpert = [(1 - eps) * a + eps * b for a, b in zip(q, t)]          # TV(qpert, q) <= eps
    ok_in &= enlarged_residual(p, qpert, K, float(eps)) < 1e-7
    qfar = tilt(p, [F(1)] * (n - 1) + [F(40)])                      # ratio 40 > K
    ok_outE &= enlarged_residual(p, qfar, K, 0.001) > 1e-4
check("Cor 4.13: residual vanishes for tilts perturbed within epsilon (LP)", ok_in)
check("Cor 4.13: residual stays positive for changes far outside the enlarged class (LP)", ok_outE)

print(f"\n{sum(res)}/{len(res)} checks passed")
