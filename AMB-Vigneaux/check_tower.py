"""Computational checks for the revised Three-Level Posterior Tower paper.

Exact rational arithmetic on small examples. Checks:
  * comonad laws and naturality for C(X) = X x E(X), E = D x D^2 x D^3 x Sh
  * the averaging law lambda-bar: naturality, unit, counit, multiplication (pass),
    comultiplication (fails, explicit counterexample)
  * the shift law lambda-sigma: counit (pass); unit, multiplication (fail)
  * K naturality (K/J coherence with D^3 j)
  * Prop 4.6 counterexample, Theorem 11.3 two-point argument
"""
from fractions import Fraction as F
import itertools, random

class Dist:
    __slots__ = ("m",)
    def __init__(self, items):
        m = {}
        for v, p in items:
            if p:
                m[v] = m.get(v, 0) + p
        self.m = m
    def __eq__(self, o): return isinstance(o, Dist) and self.m == o.m
    def __hash__(self): return hash(frozenset(self.m.items()))
    def __repr__(self): return "D{" + ", ".join(f"{p}:{v!r}" for v, p in self.m.items()) + "}"
    def items(self): return self.m.items()

def delta(v): return Dist([(v, F(1))])
def Dmap(f, d): return Dist([(f(v), p) for v, p in d.items()])
def mu(dd): return Dist([(v, p * q) for d, p in dd.items() for v, q in d.items()])
def mix(pairs):  # pairs of (weight, Dist)
    return Dist([(v, w * p) for w, d in pairs for v, p in d.items()])
def Dk(k, f):
    g = f
    for _ in range(k):
        g = (lambda h: (lambda d: Dmap(h, d)))(g)
    return g

THETA = ("t1", "t2")
# Hyp(X) = Theta x D(X); Hyp(f)(th, rho) = (th, D f rho)
def Hyp(f): return lambda h: (h[0], Dmap(f, h[1]))
def Sh(f): return lambda s: Dmap(Hyp(f), s)

# E(X) = (D X, D^2 X, D^3 X, Sh X)
def E(f):
    return lambda c: (Dk(1, f)(c[0]), Dk(2, f)(c[1]), Dk(3, f)(c[2]), Sh(f)(c[3]))
def Emix(pairs):  # convex combination in E(Y), componentwise
    return tuple(mix([(w, c[i]) for w, c in pairs]) for i in range(4))

def C(f): return lambda p: (f(p[0]), E(f)(p[1]))
def eps(p): return p[0]
def j(c): return lambda y: (y, c)
def dlt(p):
    x, c = p
    return (p, E(j(c))(c))
eta = delta

# ---------- random generators ----------
R = random.Random(7)
def rdist(vals, k=2):
    vs = R.sample(vals, min(k, len(vals)))
    ws = [F(R.randint(1, 4)) for _ in vs]; s = sum(ws)
    return Dist([(v, w / s) for v, w in zip(vs, ws)])
def rctx(X):
    D1 = [delta(x) for x in X] + [rdist(X) for _ in range(2)]
    D2 = [rdist(D1) for _ in range(3)]
    D3 = [rdist(D2) for _ in range(3)]
    rho = R.choice(D1); Pi = R.choice(D2); Om = R.choice(D3)
    hyps = [(R.choice(THETA), R.choice(D1)) for _ in range(3)]
    Sig = rdist(hyps)
    return (rho, Pi, Om, Sig)
def rpoint(X): return (R.choice(X), rctx(X))

X = (0, 1); Y = ("a", "b", "c")
f = {0: "a", 1: "a"}.get   # non-injective
g = {0: "a", 1: "b"}.get

results = []
def check(name, cond):
    results.append((name, cond)); print(("PASS " if cond else "FAIL ") + name)

# ---------- comonad ----------
ok = {k: True for k in ["counit1", "counit2", "coassoc", "nat_eps", "nat_delta"]}
for _ in range(40):
    p = rpoint(X)
    ok["counit1"] &= eps(dlt(p)) == p
    ok["counit2"] &= C(eps)(dlt(p)) == p
    ok["coassoc"] &= C(dlt)(dlt(p)) == dlt(dlt(p))
    for h in (f, g):
        ok["nat_eps"] &= eps(C(h)(p)) == h(eps(p))
        ok["nat_delta"] &= dlt(C(h)(p)) == C(C(h))(dlt(p))
for k, v in ok.items(): check("comonad " + k, v)

# ---------- K: Hyp(X) -> Hyp(D^3 X), (th,rho) -> (th, eta eta eta rho) ----------
def K(h): return (h[0], delta(delta(delta(h[1]))))
okK = True
for _ in range(30):
    c = rctx(X); h = (R.choice(THETA), rdist(list(X)))
    lhs = Hyp(Dk(3, j(c)))(K(h)); rhs = K(Hyp(j(c))(h))
    okK &= lhs == rhs
check("K/J coherence with D^3 j", okK)
okKn = all(Hyp(Dk(3, f))(K(h)) == K(Hyp(f)(h)) for h in [(t, rdist(list(X))) for t in THETA for _ in range(5)])
check("K natural", okKn)

# ---------- averaging law lambda-bar ----------
def lam_bar(M):
    P = Dmap(eps, M)
    e = Emix([(a, E(eta)(p[1])) for p, a in M.items()])
    return (P, e)
def rM(Xs, n=2): return rdist([rpoint(Xs) for _ in range(4)], n)

okn = oku = okc = okm = True
for _ in range(25):
    M = rM(X)
    for h in (f, g):
        okn &= C(Dk(1, h))(lam_bar(M)) == lam_bar(Dmap(C(h), M))
    okc &= eps(lam_bar(M)) == Dmap(eps, M)
    p = rpoint(X)
    oku &= lam_bar(delta(p)) == C(eta)(p)
    Mp = rdist([rM(X) for _ in range(3)], 2)
    lhs = lam_bar(mu(Mp))
    rhs = C(mu)(lam_bar(Dmap(lam_bar, Mp)))
    okm &= lhs == rhs
check("lambda-bar natural", okn)
check("lambda-bar unit", oku)
check("lambda-bar counit", okc)
check("lambda-bar multiplication", okm)

# comultiplication: explicit counterexample
def ctx(rho):
    return (rho, delta(delta(0)), delta(delta(delta(0))), delta(("t1", delta(0))))
c1, c2 = ctx(delta(0)), ctx(delta(1))
M = Dist([((0, c1), F(1, 2)), ((0, c2), F(1, 2))])
lhs = dlt(lam_bar(M))
rhs = C(lam_bar)(lam_bar(Dmap(dlt, M)))
check("lambda-bar comultiplication FAILS on counterexample", lhs != rhs)
check("  ...and first components agree", lhs[0] == rhs[0])
lv1, rv1 = lhs[1][0], rhs[1][0]
e = lam_bar(M)[1]; e1 = E(eta)(c1); e2 = E(eta)(c2)
check("  ...LHS level-1 = 1/2 (d0,e) + 1/2 (d1,e)",
      lv1 == Dist([((delta(0), e), F(1, 2)), ((delta(1), e), F(1, 2))]))
check("  ...RHS level-1 = 1/2 (d0,e1) + 1/2 (d1,e2)",
      rv1 == Dist([((delta(0), e1), F(1, 2)), ((delta(1), e2), F(1, 2))]))
okcm = True
for _ in range(15):
    Mr = rM(X)
    a = dlt(lam_bar(Mr)); b = C(lam_bar)(lam_bar(Dmap(dlt, Mr)))
    okcm &= C(eps)(a) == C(eps)(b) and eps(a) == eps(b)
check("lambda-bar comult sides agree after C(eps) and eps", okcm)

# ---------- shift law lambda-sigma ----------
def K1(h): return (h[0], delta(h[1]))
def lam_sig(M):
    P = Dmap(eps, M)
    L1 = Dist([(p[1][0], a) for p, a in M.items()])
    L2 = Dist([(p[1][1], a) for p, a in M.items()])
    L3 = Dist([(p[1][2], a) for p, a in M.items()])
    S = mix([(a, Dmap(K1, p[1][3])) for p, a in M.items()])
    return (P, (L1, L2, L3, S))
oksn = oksc = True
for _ in range(20):
    Mr = rM(X)
    for h in (f, g):
        oksn &= C(Dk(1, h))(lam_sig(Mr)) == lam_sig(Dmap(C(h), Mr))
    oksc &= eps(lam_sig(Mr)) == Dmap(eps, Mr)
check("lambda-sigma natural", oksn)
check("lambda-sigma counit", oksc)
u = 0.5
pt = (0, (Dist([(0, F(1, 2)), (1, F(1, 2))]),) + ctx(delta(0))[1:])
check("lambda-sigma unit FAILS (rho = 1/2 d0 + 1/2 d1)", lam_sig(delta(pt)) != C(eta)(pt))
check("  ...level-1 agree after mu",
      mu(lam_sig(delta(pt))[1][0]) == mu(C(eta)(pt)[1][0]))
cA, cB = ctx(delta(0)), ctx(delta(1))
Mp = delta(Dist([((0, cA), F(1, 2)), ((1, cB), F(1, 2))]))
L = lam_sig(mu(Mp)); Rr = C(mu)(lam_sig(Dmap(lam_sig, Mp)))
check("lambda-sigma multiplication FAILS", L != Rr)
check("  ...level-1 agree after mu", mu(L[1][0]) == mu(Rr[1][0]))

# ---------- algebra ----------
def normalize(d):
    z = sum(p for _, p in d.items()); return Dist([(v, p / z) for v, p in d.items()])
def cond(P, L): return normalize(Dist([((x, h), L[h] * p) for (x, h), p in P.items()]))
def marg(P): return Dmap(lambda xh: xh[0], P)
P1 = Dist([(("x1", "h1"), F(1, 2)), (("x2", "h2"), F(1, 2))])
P2 = Dist([(("x1", "h2"), F(1, 2)), (("x2", "h1"), F(1, 2))])
Lk = {"h1": F(2), "h2": F(1)}
check("Prop 4.6 counterexample: same marginal", marg(P1) == marg(P2))
check("Prop 4.6 counterexample: conditioned marginals differ",
      marg(cond(P1, Lk)) != marg(cond(P2, Lk)))
print(" cond marginals:", marg(cond(P1, Lk)), marg(cond(P2, Lk)))

print()
print(f"{sum(c for _, c in results)}/{len(results)} checks as expected")

# lambda-sigma comultiplication
bad = 0
for _ in range(15):
    Mr = rM(X)
    a = dlt(lam_sig(Mr)); b = C(lam_sig)(lam_sig(Dmap(dlt, Mr)))
    bad += (a != b)
print("lambda-sigma comultiplication failures:", bad, "/ 15")
a = dlt(lam_sig(M)); b = C(lam_sig)(lam_sig(Dmap(dlt, M)))
print("lambda-sigma comult on two-point M fails:", a != b, "| Dirac input holds:",
      all(dlt(lam_sig(delta(p))) == C(lam_sig)(lam_sig(Dmap(dlt, delta(p)))) for p in [rpoint(X) for _ in range(10)]))
