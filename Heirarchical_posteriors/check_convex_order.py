"""Computational companion to Section 12.6 (convex-order Beck comparisons).

All certificates are exact (rational arithmetic).

  1. Unit:            lambda^sigma eta_C  <=_cx  C(eta)        (level one)
  2. Multiplication:  C(mu) lambda_D D(lambda)  <=_cx  lambda mu_C   (level one)
     Certificate for 1-2: an explicit martingale coupling (Strassen), checked for
     both marginals and for the mean condition atom by atom.
  3. Orientation:     the reverse inequalities fail; witness phi(v) = |v|^2 gives a strict
     inequality whenever the comparison is non-degenerate.  Relative to the shape of the
     Beck axioms, (1) puts the lambda-first side below and (2) puts it above.
  4. Comultiplication incomparability, for both the shift law and the averaging law:
     L !<=_cx R and R !<=_cx L, witnessed by two explicit convex functions
     phi_1 = (s - u(tau))^2 and phi_2 = (tau - t)^2 built from affine coordinates s, tau;
     the two sides have equal barycenters, so the failure is not a barycenter mismatch.
"""
import contextlib, io, random
from fractions import Fraction as F
with contextlib.redirect_stdout(io.StringIO()):
    import check_tower as T
Dist, delta, Dmap, mu, mix, Emix = T.Dist, T.delta, T.Dmap, T.mu, T.mix, T.Emix
C, E, eta, dlt = T.C, T.E, T.eta, T.dlt
lam_sig, lam_bar, ctx = T.lam_sig, T.lam_bar, T.ctx

R = random.Random(17)
res = []
def check(n, c): res.append(bool(c)); print(("PASS " if c else "FAIL ") + n)

BASE = (0, 1, 2)
def coords(rho): return tuple(rho.m.get(b, F(0)) for b in BASE)
def rdist(vals, k):
    vs = R.sample(vals, k); ws = [F(R.randint(1, 5)) for _ in vs]; s = sum(ws)
    return Dist([(v, w / s) for v, w in zip(vs, ws)])
def rpost(k=2): return rdist(list(BASE), k)
def sq(v): return sum(x * x for x in v)
def Eph(Pi, phi): return sum(p * phi(coords(r)) for r, p in Pi.items())

def verify_coupling(lower, upper, pi):
    """pi: dict (l,u) -> mass. Checks marginals and the martingale (mean) condition."""
    ml, mu_ = {}, {}
    for (l, u), w in pi.items():
        ml[l] = ml.get(l, 0) + w; mu_[u] = mu_.get(u, 0) + w
    if ml != lower.m or mu_ != upper.m: return False
    for l in ml:
        mean = [sum(w * coords(u)[i] for (l2, u), w in pi.items() if l2 == l) for i in range(len(BASE))]
        if tuple(mean) != tuple(ml[l] * c for c in coords(l)): return False
    return True

# ---------- 1 & 3: unit ----------
ok_u = ok_u_rev = ok_u_side = True
for _ in range(40):
    rho = rpost(R.choice([2, 3]))
    c = (rho,) + ctx(delta(0))[1:]
    p = (0, c)
    lam_side = lam_sig(delta(p))[1][0]          # lambda . eta_C, level one
    beck_side = C(eta)(p)[1][0]                 # C(eta), level one
    ok_u_side &= lam_side == delta(rho) and beck_side == Dmap(delta, rho)
    pi = {(rho, delta(x)): w for x, w in rho.items()}
    ok_u &= verify_coupling(lam_side, beck_side, pi)
    ok_u_rev &= Eph(beck_side, sq) > Eph(lam_side, sq)      # reverse fails strictly
check("unit sides are delta_rho (lambda side) and D(eta) rho (C side)", ok_u_side)
check("unit: lambda.eta_C <=_cx C(eta) (exact martingale coupling)", ok_u)
check("unit: reverse inequality fails (|v|^2 witness, rho non-Dirac)", ok_u_rev)

# ---------- 2 & 3: multiplication ----------
ok_m = ok_m_rev = True
for _ in range(40):
    groups = []
    for _ in range(R.choice([2, 3])):
        pts = [(0, (rpost(R.choice([1, 2])),) + ctx(delta(0))[1:]) for _ in range(3)]
        groups.append(rdist(pts, R.choice([2, 3])))
    Mp = rdist(groups, len(groups))
    lam_mu = lam_sig(mu(Mp))[1][0]                              # lambda . mu_C
    c_side = C(mu)(lam_sig(Dmap(lam_sig, Mp)))[1][0]            # C(mu) lambda_D D(lambda)
    pi = {}
    for Ml, nu in Mp.items():
        G = lam_sig(Ml)[1][0]
        lo = mu(G)
        for r, a in G.items():
            pi[(lo, r)] = pi.get((lo, r), 0) + nu * a
    ok_m &= verify_coupling(c_side, lam_mu, pi)
    nondeg = any(len(lam_sig(Ml)[1][0].m) > 1 for Ml in Mp.m)
    if nondeg:
        ok_m_rev &= Eph(lam_mu, sq) > Eph(c_side, sq)
check("multiplication: C(mu) lambda_D D(lambda) <=_cx lambda.mu_C (exact martingale coupling)", ok_m)
check("multiplication: reverse inequality fails (|v|^2 witness, non-degenerate groups)", ok_m_rev)
check("orientation: lambda-first side is BELOW in the unit comparison and ABOVE in the multiplication comparison",
      ok_u and ok_u_rev and ok_m and ok_m_rev)

# ---------- 4: comultiplication incomparability ----------
def comult_sides(law):
    c0, c1 = ctx(delta(0)), ctx(delta(1))
    M = Dist([((0, c0), F(1, 2)), ((0, c1), F(1, 2))])
    Lft = dlt(law(M))[1][0]                         # delta_D . lambda, level one
    Rgt = C(law)(law(Dmap(dlt, M)))[1][0]           # C(lambda) lambda_C D(delta), level one
    return Lft, Rgt
def s_coord(P): return P.m.get(1, F(0))             # affine on DX
def tau_coord(e): return e[0].m.get(delta(1), F(0))  # affine on E(DX): level-one mass at delta_1
def E_on(D_, phi): return sum(p * phi(P, e) for (P, e), p in D_.items())
def bary_pairs(D_):
    return (mix([(p, P) for (P, e), p in D_.items()]), Emix([(p, e) for (P, e), p in D_.items()]))

for name, law in (("shift law", lam_sig), ("averaging law", lam_bar)):
    Lft, Rgt = comult_sides(law)
    check(f"{name}: comultiplication sides differ", Lft != Rgt)
    check(f"{name}: comultiplication sides have equal barycenters", bary_pairs(Lft) == bary_pairs(Rgt))
    taus = sorted({tau_coord(e) for (P, e) in Rgt.m})
    t0, t1 = taus[0], taus[-1]
    t = tau_coord(next(iter(Lft.m))[1])
    phi1 = lambda P, e: (s_coord(P) - (tau_coord(e) - t0) / (t1 - t0)) ** 2
    phi2 = lambda P, e: (tau_coord(e) - t) ** 2
    check(f"{name}: L !<=_cx R  (phi_1: E_L = {E_on(Lft, phi1)} > E_R = {E_on(Rgt, phi1)})",
          E_on(Lft, phi1) > E_on(Rgt, phi1))
    check(f"{name}: R !<=_cx L  (phi_2: E_R = {E_on(Rgt, phi2)} > E_L = {E_on(Lft, phi2)})",
          E_on(Rgt, phi2) > E_on(Lft, phi2))

print(f"\n{sum(res)}/{len(res)} checks passed")
