"""Checks for Section 12 (coalgebras, the lifted monad and bialgebras of the base-relative law),
and for the convex-order comparison of the absolute shift law.

  * C^B-coalgebras are exactly maps g: Y -> A_B (counit forces the form, coassociativity is automatic)
  * cofree adjunction U -| F: coalgebra maps (Y,g) -> F X  <->  set maps Y -> X
  * the lifted monad: eta and mu are maps over A_B for D~(Y,g) = (DY, beta . D g)
  * lambda^B is recovered as the coalgebra map D~(F X) -> F(DX) over D(pr1)
  * bialgebra condition <=> g affine (sigma passes; a non-affine g fails)
  * shift law, level one: unit and multiplication defects are ordered in the convex order
    (checked on a family of convex test functions; the proof is Jensen's inequality)
"""
import contextlib, io, random, itertools
from fractions import Fraction as F
with contextlib.redirect_stdout(io.StringIO()):
    from check_tower import Dist, delta, Dmap, mu, mix, rdist, THETA
    from check_base_relative import bary, sigma, rshadow, coherent, lam, B

R = random.Random(3)
res = []
def check(n, c): res.append(bool(c)); print(("PASS " if c else "FAIL ") + n)

eps = lambda p: p[0]
dlt = lambda p: (p, p[1])

# ---------- coalgebras ----------
Y = ("y0", "y1", "y2")
ok_form = ok_coassoc = True
for _ in range(20):
    g = {y: sigma(rshadow(B)) for y in Y}
    gamma = lambda y: (y, g[y])
    ok_form &= all(eps(gamma(y)) == y for y in Y)
    # coassociativity: C(gamma) gamma = delta gamma
    ok_coassoc &= all(((gamma(y)[0], gamma(y)[1]) == gamma(y)) and
                      ((gamma(gamma(y)[0]), gamma(y)[1]) == (gamma(y), g[y])) and
                      dlt(gamma(y)) == (gamma(y), g[y]) for y in Y)
check("coalgebra structure is <id, g> and coassociativity holds automatically", ok_form and ok_coassoc)

# ---------- cofree adjunction ----------
X = ("a", "b")
ok_adj = True
for _ in range(20):
    g = {y: sigma(rshadow(B)) for y in Y}
    phi = {y: R.choice(X) for y in Y}
    h = {y: (phi[y], g[y]) for y in Y}            # (phi x id) . gamma
    # h is a coalgebra map (Y,g) -> F X = (X x A, pr2):  pr2 . h = g
    ok_adj &= all(h[y][1] == g[y] for y in Y)
    # round trip
    ok_adj &= all(h[y][0] == phi[y] for y in Y)
check("cofree adjunction: coalgebra maps (Y,g) -> F X correspond to set maps Y -> X", ok_adj)

# ---------- lifted monad ----------
def lift_g(g):  # structure of D~(Y, g)
    return lambda P: bary(Dmap(lambda y: g[y], P))
ok_eta = ok_mu = True
for _ in range(15):
    g = {y: sigma(rshadow(B)) for y in Y}
    gD = lift_g(g)
    ok_eta &= all(gD(delta(y)) == g[y] for y in Y)
    PP = rdist([rdist(list(Y), 2) for _ in range(3)], 2)
    gDD = lambda Q: bary(Dmap(gD, Q))
    ok_mu &= gD(mu(PP)) == gDD(PP)
check("eta is a map over A_B for the lifted monad", ok_eta)
check("mu is a map over A_B for the lifted monad", ok_mu)

# lambda recovered from the cofree coalgebra
ok_rec = True
for _ in range(15):
    M = rdist([(R.choice(X), sigma(rshadow(B))) for _ in range(4)], 3)
    rec = (Dmap(eps, M), bary(Dmap(lambda p: p[1], M)))   # <D pr1, beta . D pr2>
    ok_rec &= rec == lam(M)
check("lambda^B equals the induced coalgebra map D~(F X) -> F(DX)", ok_rec)

# ---------- bialgebras ----------
# Y = Sh(B) with mu as D-algebra; g = sigma is affine -> bialgebra
ok_bi = True
for _ in range(15):
    Ss = rdist([rshadow(B) for _ in range(3)], 3)          # element of D(Sh B)
    ok_bi &= sigma(mu(Ss)) == bary(Dmap(sigma, Ss))
check("(Sh B, mu, sigma) satisfies the bialgebra condition", ok_bi)
# a non-affine context map fails: g(S) = sigma(delta_{first hypothesis in support})
def g_bad(S):
    h = sorted(S.m.keys(), key=repr)[0]
    return sigma(delta(h))
Ss = Dist([(delta(("t1", delta(0))), F(1, 2)), (delta(("t2", delta(1))), F(1, 2))])
check("a non-affine context map violates the bialgebra condition",
      g_bad(mu(Ss)) != bary(Dmap(g_bad, Ss)))

# ---------- convex order, shift law level one ----------
# convex test functions on D{0,1} ~ [0,1] (prob of 1) and on D{0,1,2}
def phis(dim):
    fs = [lambda v: sum(x * x for x in v)]
    for _ in range(6):
        w = [F(R.randint(-5, 5)) for _ in range(dim)]; c = F(R.randint(-3, 3))
        w2 = [F(R.randint(-5, 5)) for _ in range(dim)]
        fs.append(lambda v, w=w, c=c, w2=w2: max(sum(a * b for a, b in zip(w, v)) + c,
                                                 sum(a * b for a, b in zip(w2, v))))
    return fs
def E(Pi, phi, base):
    return sum(p * phi([r.m.get(b, F(0)) for b in base]) for r, p in Pi.items())
base = (0, 1, 2)
ok_unit = ok_mult = ok_bary = True
fs = phis(3)
for _ in range(25):
    rho = rdist(list(base), 3)
    shift_unit = delta(rho)                                  # lambda^sigma(eta) level one
    beck_unit = Dmap(delta, rho)                             # C(eta) level one = D(eta) rho
    ok_unit &= all(E(shift_unit, f, base) <= E(beck_unit, f, base) for f in fs)
    # multiplication: groups l with members k
    groups = [rdist([rdist(list(base), 2) for _ in range(3)], 2) for _ in range(2)]
    nu = rdist([0, 1], 2)
    lhs = Dist([(r, nu.m.get(l, 0) * a) for l, G in enumerate(groups) for r, a in G.items()])
    rhs = Dist([(mu(G), nu.m.get(l, 0)) for l, G in enumerate(groups)])
    ok_mult &= all(E(rhs, f, base) <= E(lhs, f, base) for f in fs)
    ok_bary &= mu(lhs) == mu(rhs) and mu(shift_unit) == mu(beck_unit)
check("shift law, unit at level one: delta_rho <=_cx D(eta) rho", ok_unit)
check("shift law, multiplication at level one: pooled <=_cx unpooled", ok_mult)
check("both comparisons have equal barycenters", ok_bary)

print(f"\n{sum(res)}/{len(res)} checks passed")
