"""Checks for the base-relative three-level construction (option 2).

Fixed base B. Context algebra A_B = D B x D^2 B x D^3 B x Sh B, convex componentwise.
Comonad C^B(Y) = Y x A_B (reader comonad).  Law lambda(M) = (D pr1 M, bary(D pr2 M)).
Checks: comonad laws; naturality in Y; all four mixed-law axioms (U, Co, M, Cm);
coherent towers closed under barycenter; canonical section sigma: Sh B -> A_B is affine
and coherent; base change B -> B' is affine, preserves coherence, commutes with lambda.
"""
import random
from fractions import Fraction as F
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):   # reuse primitives; silence the other script's output
    from check_tower import (Dist, delta, Dmap, mu, mix, Dk, THETA, Hyp, Sh, rdist)

R = random.Random(11)
def A_map(f):  # base change on contexts
    return lambda a: (Dk(1, f)(a[0]), Dk(2, f)(a[1]), Dk(3, f)(a[2]), Sh(f)(a[3]))
def bary(dA):  # D(A) -> A
    return tuple(mix([(w, a[i]) for a, w in dA.items()]) for i in range(4))
q = lambda h: h[1]
def sigma(S):  # canonical hierarchy of a shadow
    Pi = Dmap(q, S)
    return (mu(Pi), Pi, Dmap(delta, Pi), S)   # (rho, Pi, Omega) = (mu Pi, Pi, D(eta) Pi)
def coherent(a):
    rho, Pi, Om, S = a
    return mu(Pi) == rho and mu(Om) == Pi and Dmap(q, S) == Pi

B = (0, 1); Bp = ("u", "v", "w")
fB = {0: "u", 1: "u"}.get; gB = {0: "u", 1: "v"}.get
def rshadow(base):
    D1 = [delta(x) for x in base] + [rdist(list(base)) for _ in range(2)]
    return rdist([(R.choice(THETA), R.choice(D1)) for _ in range(3)])
def rctx(base, coh=True):
    if coh:
        return sigma(rshadow(base))
    D1 = [delta(x) for x in base] + [rdist(list(base))]
    D2 = [rdist(D1) for _ in range(2)]
    return (R.choice(D1), R.choice(D2), rdist(D2), rshadow(base))

def C(f): return lambda p: (f(p[0]), p[1])
eps = lambda p: p[0]
dlt = lambda p: (p, p[1])
def lam(M): return (Dmap(eps, M), bary(Dmap(lambda p: p[1], M)))

Y = ("y0", "y1"); fY = {"y0": "z", "y1": "z"}.get
def rpt(coh=True): return (R.choice(Y), rctx(B, coh))
def rM(coh=True): return rdist([rpt(coh) for _ in range(4)], 2)

res = []
def check(n, c): res.append(c); print(("PASS " if c else "FAIL ") + n)

ok = dict(counit=True, coassoc=True, nat=True, U=True, Co=True, M=True, Cm=True)
for _ in range(30):
    for coh in (True, False):
        p = rpt(coh); M = rM(coh)
        ok["counit"] &= eps(dlt(p)) == p and C(eps)(dlt(p)) == p
        ok["coassoc"] &= C(dlt)(dlt(p)) == dlt(dlt(p))
        ok["nat"] &= C(Dk(1, fY))(lam(M)) == lam(Dmap(C(fY), M))
        ok["U"] &= lam(delta(p)) == C(delta)(p)
        ok["Co"] &= eps(lam(M)) == Dmap(eps, M)
        Mp = rdist([rM(coh) for _ in range(3)], 2)
        ok["M"] &= lam(mu(Mp)) == C(mu)(lam(Dmap(lam, Mp)))
        ok["Cm"] &= dlt(lam(M)) == C(lam)(lam(Dmap(dlt, M)))
for k, v in ok.items(): check("base-relative " + k, v)

okc = oks = okb = okbc = True
for _ in range(30):
    cs = [rctx(B) for _ in range(3)]
    w = rdist(list(range(3)), 3)
    okc &= coherent(bary(Dist([(cs[i], p) for i, p in w.items()])))
    Ss = [rshadow(B) for _ in range(3)]
    okc &= all(coherent(sigma(S)) for S in Ss)
    oks &= sigma(mix([(p, Ss[i]) for i, p in w.items()])) == bary(Dist([(sigma(Ss[i]), p) for i, p in w.items()]))
    for f in (fB, gB):
        a = rctx(B)
        okb &= coherent(A_map(f)(a)) and A_map(f)(sigma(a[3])) == sigma(Sh(f)(a[3]))
        M = rM()
        lhs = (lam(M)[0], A_map(f)(lam(M)[1]))
        rhs = lam(Dmap(lambda p: (p[0], A_map(f)(p[1])), M))
        okbc &= lhs == rhs
check("coherent towers closed under barycenter; sigma lands in coherent towers", okc)
check("sigma is affine (pooling commutes with hierarchy formation)", oks)
check("base change preserves coherence and commutes with sigma", okb)
check("base change commutes with lambda", okbc)

# canonical hierarchy keeps distinct posteriors at level 2
r0, r1 = delta(0), Dist([(0, F(1, 3)), (1, F(2, 3))])
S0, S1 = delta(("t1", r0)), delta(("t1", r1))
M = Dist([(("y0", sigma(S0)), F(1, 2)), (("y1", sigma(S1)), F(1, 2))])
ctx = lam(M)[1]
check("pooled level 2 = 1/2 d_r0 + 1/2 d_r1 (hierarchy kept)",
      ctx[1] == Dist([(r0, F(1, 2)), (r1, F(1, 2))]))
check("pooled level 1 = barycenter", ctx[0] == Dist([(0, F(2, 3)), (1, F(1, 3))]))

# conditioning commutes with pooling: lambda of the evidence-reweighted sample
def condS(S, L):
    w = Dist([(h, L[h[0]] * p) for h, p in S.items()]); z = sum(p for _, p in w.items())
    return Dist([(h, p / z) for h, p in w.items()]), z
okcond = True
for _ in range(30):
    L = {"t1": F(R.randint(1, 5)), "t2": F(R.randint(1, 5))}
    Ss = [rshadow(B) for _ in range(3)]; a = rdist(list(range(3)), 3)
    pooled, _ = condS(mix([(p, Ss[i]) for i, p in a.items()]), L)
    parts = [condS(S, L) for S in Ss]
    Z = sum(p * parts[i][1] for i, p in a.items())
    Mc = Dist([(("y0", sigma(parts[i][0])), p * parts[i][1] / Z) for i, p in a.items()])
    okcond &= lam(Mc)[1] == sigma(pooled)
check("conditioning commutes with pooling (evidence-reweighted samples)", okcond)
print(f"\n{sum(res)}/{len(res)} checks passed")
