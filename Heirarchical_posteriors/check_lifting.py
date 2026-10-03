"""Checks for Section 10 (coherent lifting and obstruction certificates).

Admissibility data: finite S1 (admissible posteriors) in D B, and S2 in D(S1) with eta(S1) in S2.
Admissible sets: L1 = conv S1, L2 = D(S1), L3 = D(S2).
Checks (exact arithmetic unless noted):
  * mu(L3) = L2 and mu(L2) = L1 on random elements (surjectivity via explicit lifts)
  * monotonicity of the failure level
  * worked examples of first failure at levels 1, 2 and 3, with certificates
  * level-2/3 distance = 2 * (mass outside), and it is affine under pooling
  * level-1 distance (LP, floating point) equals the dual certificate margin (minimax)
  * level-1 obstructions can cancel under pooling; level-2/3 obstructions cannot
  * canonical contexts sigma(D(H)) realize exactly L1 x L2 x L3 in the committed regime
"""
import contextlib, io, random, itertools
from fractions import Fraction as F
import numpy as np
from scipy.optimize import linprog
with contextlib.redirect_stdout(io.StringIO()):
    from check_tower import Dist, delta, Dmap, mu, mix, rdist

R = random.Random(5)
res = []
def check(n, c): res.append(bool(c)); print(("PASS " if c else "FAIL ") + n)

def vec(rho, B): return [rho.m.get(b, F(0)) for b in B]
def P(*ps, B=(0, 1)): return Dist([(b, F(p)) for b, p in zip(B, ps)])

# ---------- level-1 membership and certificates via LP (floating point) ----------
def l1_distance(x, S1, B):
    """min_w ||x - sum_h w_h rho_h||_1 over the simplex (LP)."""
    n, m = len(S1), len(B)
    # variables: w (n), t (m);  min sum t  s.t.  -t <= x - M w <= t, sum w = 1, w,t >= 0
    M = np.array([[float(vec(r, B)[j]) for r in S1] for j in range(m)])
    xv = np.array([float(v) for v in vec(x, B)])
    c = np.r_[np.zeros(n), np.ones(m)]
    A = np.r_[np.c_[-M, -np.eye(m)], np.c_[M, -np.eye(m)]]
    b = np.r_[-xv, xv]
    r = linprog(c, A_ub=A, b_ub=b, A_eq=np.r_[np.ones(n), np.zeros(m)][None], b_eq=[1],
                bounds=[(0, None)] * (n + m))
    return r.fun
def dual_margin(x, S1, B):
    """max_{||f||_inf <= 1} f.x - max_h f.rho_h   (LP in (f, s))."""
    m = len(B)
    xv = np.array([float(v) for v in vec(x, B)])
    c = np.r_[-xv, 1.0]                       # minimize -(f.x - s)
    A = np.array([np.r_[[float(v) for v in vec(r, B)], -1.0] for r in S1])  # f.rho_h - s <= 0
    r = linprog(c, A_ub=A, b_ub=np.zeros(len(S1)), bounds=[(-1, 1)] * m + [(None, None)])
    return -r.fun

# ---------- admissibility ----------
def in_L2(Pi, S1): return all(r in S1 for r in Pi.m)
def in_L3(Om, S2): return all(p in S2 for p in Om.m)
def mass_out(X, S): return sum((p for v, p in X.items() if v not in S), F(0))
def l1(X, Y): return sum(abs(X.m.get(k, 0) - Y.m.get(k, 0)) for k in set(X.m) | set(Y.m))
def project(X, S):
    """closest point of D(S) in l1: keep mass inside S, move the outside mass to one point of S."""
    inside = [(v, p) for v, p in X.items() if v in S]
    out = mass_out(X, S)
    s0 = next(iter(S))
    return Dist(inside + [(s0, out)])

# ---------- worked examples:  B = {0,1},  S1 = {d0, (1/2,1/2)},  S2 = eta(S1) ----------
B = (0, 1)
r0, rh = P(1, 0), P(F(1, 2), F(1, 2))
S1 = [r0, rh]
S2 = [delta(r) for r in S1]

# level-1 failure: claim d1
x1 = P(0, 1)
f = (F(-1), F(1))
margin = sum(a * b for a, b in zip(f, vec(x1, B))) - max(sum(a * b for a, b in zip(f, vec(r, B))) for r in S1)
check("Ex level 1: certificate f=(-1,1) has margin 1", margin == 1)
check("Ex level 1: LP distance = 1 = dual margin",
      abs(l1_distance(x1, S1, B) - 1) < 1e-9 and abs(dual_margin(x1, S1, B) - 1) < 1e-9)

# level-2 first failure: Pi = delta_{(3/4,1/4)}; its barycenter is admissible
r34 = P(F(3, 4), F(1, 4))
x2 = delta(r34)
check("Ex level 2: barycenter (3/4,1/4) in conv S1", abs(l1_distance(mu(x2), S1, B)) < 1e-9)
check("Ex level 2: Pi not in D(S1); witness point (3/4,1/4)", not in_L2(x2, S1) and r34 not in S1)
alt = Dist([(r0, F(1, 2)), (rh, F(1, 2))])
check("Ex level 2: admissible Pi' with same barycenter exists", in_L2(alt, S1) and mu(alt) == mu(x2))

# level-3 first failure: Omega = delta_{Pi'} with Pi' = 1/2 d_{r0} + 1/2 d_{rh}
x3 = delta(alt)
check("Ex level 3: mu(Omega) in L2", in_L2(mu(x3), S1))
check("Ex level 3: Omega not in D(S2) (committed regime)", not in_L3(x3, S2))
x3ok = Dist([(delta(r0), F(1, 2)), (delta(rh), F(1, 2))])
check("Ex level 3: admissible Omega' with same mu exists", in_L3(x3ok, S2) and mu(x3ok) == mu(x3))
check("Ex level 3: unrestricted S2 = D(S1) admits it (inherited regime)", in_L2(mu(x3), S1))

# ---------- random structural checks ----------
def rand_S1(k=3):
    out = []
    while len(out) < k:
        r = rdist(list(B), 2)
        if r not in out: out.append(r)
    return out
ok_surj = ok_mono = ok_dist = ok_aff = ok_minimax = ok_sigma = True
for _ in range(60):
    S1r = rand_S1()
    S2r = [delta(r) for r in S1r] + [rdist(S1r, 2)]
    # surjectivity: any Pi in D(S1) lifts to Omega = D(eta)Pi in D(S2); any rho in conv lifts
    Pi = rdist(S1r, 2)
    Om = Dmap(delta, Pi)
    ok_surj &= in_L3(Om, S2r) and mu(Om) == Pi
    # monotonicity: if Pi not in L2 then any Omega over it is not in L3
    bad = rdist(S1r + [P(F(1, 5), F(4, 5))], 3)
    if not in_L2(bad, S1r):
        for Omega in [delta(bad), Dmap(delta, bad)]:
            ok_mono &= not in_L3(Omega, S2r)
    # distance formula and affinity at level 2
    cands = S1r + [rdist(list(B)) for _ in range(2)]
    X, Y = rdist(cands, 3), rdist(cands, 3)
    ok_dist &= l1(X, project(X, S1r)) == 2 * mass_out(X, S1r)
    a = F(R.randint(1, 9), 10)
    Z = mix([(a, X), (1 - a, Y)])
    ok_aff &= mass_out(Z, S1r) == a * mass_out(X, S1r) + (1 - a) * mass_out(Y, S1r)
    # minimax at level 1 (floating point)
    x = rdist(list(B))
    ok_minimax &= abs(l1_distance(x, S1r, B) - dual_margin(x, S1r, B)) < 1e-7
    # committed regime: sigma(Sigma) components lie in L1, L2, L3
    w = rdist(list(range(len(S1r))), 2)
    Pi_s = Dist([(S1r[i], p) for i, p in w.items()])
    sig = (mu(Pi_s), Pi_s, Dmap(delta, Pi_s))
    ok_sigma &= in_L2(sig[1], S1r) and in_L3(sig[2], [delta(r) for r in S1r]) \
        and l1_distance(sig[0], S1r, B) < 1e-9
check("mu(L3) = L2 and lifts exist (explicit Omega = D(eta) Pi)", ok_surj)
check("failure level is monotone (L2 failure propagates to L3)", ok_mono)
check("level-2 distance = 2 * mass outside S1 (attained)", ok_dist)
check("level-2 obstruction is affine under pooling", ok_aff)
check("level-1 LP distance = dual certificate margin (minimax)", ok_minimax)
check("canonical contexts lie in L1 x L2 x L3", ok_sigma)

# level-1 obstructions can cancel under pooling
S1c = [rh]
c1 = l1_distance(P(1, 0), S1c, B); c2 = l1_distance(P(0, 1), S1c, B)
c12 = l1_distance(P(F(1, 2), F(1, 2)), S1c, B)
check("level-1 obstructions cancel: d(d0)=d(d1)=1, d(pooled)=0",
      abs(c1 - 1) < 1e-9 and abs(c2 - 1) < 1e-9 and abs(c12) < 1e-9)
X2 = Dist([(P(1, 0), F(1, 2)), (P(0, 1), F(1, 2))])
check("...but the same pooled claim held at level 2 stays obstructed (mass outside = 1)",
      mass_out(X2, S1c) == 1)

print(f"\n{sum(res)}/{len(res)} checks passed")
