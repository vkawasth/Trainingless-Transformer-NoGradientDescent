"""Computational companion to Section 10.4 (effective admissibility and secondary obstructions).

General admissibility data: finite S1 in D B, S2 in D(S1), with no saturation condition.
Local sets L1 = conv S1, L2 = D(S1), L3 = D(S2); effective sets E2 = conv S2, E1 = conv mu(S2).

Exact checks (rational arithmetic):
  * worked examples (a) gap, (b) choice dependence, (c) nondegenerate indeterminacy without lift:
    fibres, o3 along the fibre, indeterminacy intervals, certificates, explicit lifts
  * pooling can hide the secondary obstruction
  * mu_B is 1-Lipschitz in l1 (random)
  * explicit lifts for random x1 in E1
  * saturated data have E = L
Floating-point LPs with exact verification of the resulting certificates:
  * x1 in E1  <=>  o3^-(x1) = 0, and d(x1, E1) <= o3^-(x1)
  * for x1 not in E1, a rationalized dual certificate f separates x1 from mu(S2) exactly
"""
import contextlib, io, random
from fractions import Fraction as F
import numpy as np
from scipy.optimize import linprog
with contextlib.redirect_stdout(io.StringIO()):
    from check_tower import Dist, delta, Dmap, mu, mix

R = random.Random(23)
res = []
def check(n, c): res.append(bool(c)); print(("PASS " if c else "FAIL ") + n)

def vec(d, keys): return [d.m.get(k, F(0)) for k in keys]
def l1(X, Y): return sum(abs(X.m.get(k, 0) - Y.m.get(k, 0)) for k in set(X.m) | set(Y.m))
def P2(a, b): return Dist([(0, F(a)), (1, F(b))])

# =================== worked examples ===================
d0, d1, u = P2(1, 0), P2(0, 1), P2(F(1, 2), F(1, 2))
Pb = Dist([(d0, F(1, 2)), (d1, F(1, 2))])            # 1/2 d_{d0} + 1/2 d_{d1}
Pa = delta(u)                                         # d_u
B2 = (0, 1)

# (a) gap: S1 = {d0, d1}, S2 = {Pb};  L1 = D B,  E1 = {u}
x = P2(F(1, 4), F(3, 4))
fib = Dist([(d0, F(1, 4)), (d1, F(3, 4))])           # unique lift: d0, d1 affinely independent
check("(a) x1 = (1/4,3/4) is locally admissible and its level-two lift has barycenter x1", mu(fib) == x)
check("(a) secondary obstruction o3 = 1/2 on the (singleton) fibre", l1(fib, Pb) == F(1, 2))
check("(a) d(x1, E1) = 1/2 = o3^-  (Lipschitz bound attained)", l1(x, mu(Pb)) == F(1, 2))
f = (F(-1), F(1))
marg = sum(a * b for a, b in zip(f, vec(x, B2))) - sum(a * b for a, b in zip(f, vec(mu(Pb), B2)))
check("(a) certificate f = (-1, 1) separates x1 from mu(S2) with margin 1/2", marg == F(1, 2))

# (b) choice dependence: S1 = {d0, d1, u}, S2 = {Pb};  x1 = u
def Pt(t): return Dist([(d0, (1 - t) / 2), (d1, (1 - t) / 2), (u, t)])
ts = [F(k, 8) for k in range(9)]
check("(b) Pi_t lies in the fibre over u for t in [0,1]", all(mu(Pt(t)) == u for t in ts))
check("(b) o3(Pi_t) = 2t along the fibre", all(l1(Pt(t), Pb) == 2 * t for t in ts))
check("(b) endpoints: Pi_0 = Pb, Pi_1 = d_u", Pt(F(0)) == Pb and Pt(F(1)) == Pa)
Om = delta(Pb)
check("(b) Pb extends: Omega = d_Pb has mu(Omega) = Pb and mu(Pb) = u", mu(Om) == Pb and mu(Pb) == u)
check("(b) d_u does not extend (conv S2 = {Pb})", Pa != Pb)
g = {d0: F(0), d1: F(0), u: F(1)}
gx = sum(p * g[r] for r, p in Pa.items()); gS = sum(p * g[r] for r, p in Pb.items())
check("(b) second-order certificate g = 1_u: <g, d_u> = 1 > 0 = <g, Pb>", gx == 1 and gS == 0)

# (c) nondegenerate indeterminacy, no lift: same data, x1 = (3/4, 1/4)
x = P2(F(3, 4), F(1, 4))
def Pc(c): return Dist([(d0, F(3, 4) - c / 2), (d1, F(1, 4) - c / 2), (u, c)])
cs = [F(k, 16) for k in range(9)]                   # c in [0, 1/2]
check("(c) Pi_c lies in the fibre over (3/4,1/4) for c in [0,1/2]", all(mu(Pc(c)) == x for c in cs))
check("(c) o3(Pi_c) = 1/2 + c, so I(x1) = [1/2, 1]", all(l1(Pc(c), Pb) == F(1, 2) + c for c in cs))
check("(c) d(x1, E1) = 1/2 = o3^-", l1(x, u) == F(1, 2))
# fibre is exactly this segment: Pi = a d0 + b d1 + c u with mean x forces a = 3/4 - c/2, b = 1/4 - c/2
check("(c) fibre endpoints satisfy the constraints (b >= 0 iff c <= 1/2)",
      min(Pc(F(1, 2)).m.values()) >= 0 and Pc(F(1, 2)).m.get(d1, 0) == 0)

# pooling hides the secondary obstruction: S1 = {d0, d1}, S2 = {Pb}
o_d0 = l1(delta(d0), Pb); o_d1 = l1(delta(d1), Pb)
check("pooling: o3^-(d0) = o3^-(d1) = 1 but the pooled claim u lifts (o3^- = 0)",
      o_d0 == 1 and o_d1 == 1 and l1(Pb, Pb) == 0 and mu(Pb) == u)

# =================== random checks ===================
B = (0, 1, 2)
def rdist(vals, k):
    vs = R.sample(vals, k); ws = [F(R.randint(1, 6)) for _ in vs]; s = sum(ws)
    return Dist([(v, w / s) for v, w in zip(vs, ws)])
def rand_data(n1=4, n2=2, sat=False):
    S1 = []
    while len(S1) < n1:
        r = rdist(list(B), R.choice([1, 2, 3]))
        if r not in S1: S1.append(r)
    S2 = [delta(r) for r in S1] if sat else []
    while len(S2) < (n1 if sat else 0) + n2:
        P = rdist(S1, R.choice([2, 3]))
        if P not in S2: S2.append(P)
    return S1, S2

# Lipschitz
okL = True
for _ in range(40):
    S1, _ = rand_data()
    P, Q = rdist(S1, 3), rdist(S1, 2)
    okL &= l1(mu(P), mu(Q)) <= l1(P, Q)
check("mu_B is 1-Lipschitz for l1 (random, exact)", okL)

# explicit lifts for x1 in E1
okLift = True
for _ in range(40):
    S1, S2 = rand_data()
    w = rdist(list(range(len(S2))), len(S2))
    Om = Dist([(S2[j], p) for j, p in w.items()])        # Omega in D(S2) = L3
    Pi = mu(Om)
    okLift &= all(r in S1 for r in Pi.m) and mu(Pi) == mix([(p, mu(S2[j])) for j, p in w.items()])
check("x1 in E1 has the explicit coherent lift Omega = sum w_j d_{Pi_j} (exact)", okLift)

# saturated data: E2 = L2
okS = True
for _ in range(30):
    S1, S2 = rand_data(sat=True)
    Pi = rdist(S1, 3)                                     # arbitrary element of L2
    Om = Dmap(delta, Pi)                                  # supported in {d_rho} subset S2
    okS &= all(P in S2 for P in Om.m) and mu(Om) == Pi
check("saturated data: every Pi in L2 lies in E2 (explicit Omega = D(eta) Pi)", okS)

# LP: distance to conv of points, its dual certificate, and o3^-
def dist_conv(x, pts):
    n, m = len(pts), len(x)
    M = np.array(pts, dtype=float).T; xv = np.array(x, dtype=float)
    c = np.r_[np.zeros(n), np.ones(m)]
    A = np.r_[np.c_[-M, -np.eye(m)], np.c_[M, -np.eye(m)]]
    r = linprog(c, A_ub=A, b_ub=np.r_[-xv, xv], A_eq=np.r_[np.ones(n), np.zeros(m)][None],
                b_eq=[1], bounds=[(0, None)] * (n + m))
    return r.fun
def dual_cert(x, pts):
    m = len(x)
    c = np.r_[-np.array(x, dtype=float), 1.0]
    A = np.array([np.r_[np.array(p, dtype=float), -1.0] for p in pts])
    r = linprog(c, A_ub=A, b_ub=np.zeros(len(pts)), bounds=[(-1, 1)] * m + [(None, None)])
    return r.x[:m]
def o3_min(x1, S1, S2):
    n1, n2, m = len(S1), len(S2), len(B)
    # variables: Pi (n1), w (n2), t (n1)
    S2m = np.array([[float(P.m.get(r, 0)) for r in S1] for P in S2]).T      # n1 x n2
    rho = np.array([[float(v) for v in vec(r, B)] for r in S1]).T            # m x n1
    c = np.r_[np.zeros(n1 + n2), np.ones(n1)]
    I = np.eye(n1)
    A_ub = np.r_[np.c_[I, -S2m, -I], np.c_[-I, S2m, -I]]
    A_eq = np.r_[np.c_[rho, np.zeros((m, n2 + n1))],
                 np.r_[np.zeros(n1), np.ones(n2), np.zeros(n1)][None]]
    b_eq = np.r_[np.array([float(v) for v in vec(x1, B)]), 1.0]
    r = linprog(c, A_ub=A_ub, b_ub=np.zeros(2 * n1), A_eq=A_eq, b_eq=b_eq,
                bounds=[(0, None)] * (n1 + n2 + n1))
    return r.fun if r.status == 0 else None

okIff = okBound = okCert = True; n_out = n_in = 0
for _ in range(60):
    S1, S2 = rand_data()
    if R.random() < 0.4:   # inside E1
        w = rdist(list(range(len(S2))), len(S2))
        x1 = mix([(p, mu(S2[j])) for j, p in w.items()])
    else:                  # random point of L1
        w = rdist(list(range(len(S1))), R.choice([2, 3]))
        x1 = mix([(p, S1[j]) for j, p in w.items()])
    pts = [vec(mu(P), B) for P in S2]
    dE = dist_conv(vec(x1, B), pts)
    o3 = o3_min(x1, S1, S2)
    okIff &= (o3 is not None) and ((dE < 1e-8) == (o3 < 1e-8))
    okBound &= dE <= o3 + 1e-8
    if dE > 1e-6:
        n_out += 1
        fv = [F(v).limit_denominator(1000) for v in dual_cert(vec(x1, B), pts)]
        lhs = sum(a * b for a, b in zip(fv, vec(x1, B)))
        okCert &= all(lhs > sum(a * b for a, b in zip(fv, p)) for p in pts)
    else:
        n_in += 1
check(f"x1 in E1 <=> o3^-(x1) = 0  ({n_in} inside, {n_out} outside, LP)", okIff and n_in > 0 and n_out > 0)
check("d(x1, E1) <= o3^-(x1)  (LP)", okBound)
check("rationalized dual certificates separate x1 from mu(S2) exactly", okCert)

print(f"\n{sum(res)}/{len(res)} checks passed")
