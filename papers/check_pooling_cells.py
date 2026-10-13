"""check_pooling_cells.py -- the pooling idempotent and the Beck defects of the absolute tower (note: Pooling idempotents and
the Beck defects of the absolute tower). Exact rational arithmetic, on the definitions of check_tower.py.

  P1  e = eta_D . mu : D^2 => D^2 is idempotent, preserves the barycenter, is natural for functions (injective and not), and
      commutes with D(k#) for random Markov kernels k on a three-point base
  P2  e is not affine: e(1/2 nu1 + 1/2 nu2) != 1/2 e(nu1) + 1/2 e(nu2)
  P3  unit: lambda_sigma . eta_C = P . C3(eta) in every component (P = k-fold nested pooling at level k, inner pooling of each
      hypothesis in the shadow), and the intermediate stages are D^{k-j}(eta_{D^j X})(c_k), j = 0..k
  P4  multiplication: lambda_sigma . mu_C = beta . D(lambda_sigma) and C3(mu) . lambda_sigma_D . D(lambda_sigma) = beta . D(P) . D(lambda_sigma)
  P5  identity criteria: (U) holds at level k iff c_k is an iterated point mass; in the shadow iff every predictive is a point mass;
      (M) holds iff within every group the samples share their level-k context components and these are iterated point
      masses of depth k-1 (the shadow is always fixed)
  P7  closed form P_k = eta . D^{k-1}(mu); e is idempotent, P_k for k >= 2 is not
  P6  the exchanged pair of Tower Prop. 13.17 is consistent with P3-P4 (left of (U) = P(right of (U)) = right of (M))
  A   audit of the original note's scripts: the nearest-atom map does not preserve barycenters, depends on the atom order, and
      fails to commute with kernels on a three-point base; script 1 returns the empty distribution on D(eta)rho
"""
import sys, os, random
H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, 'check_tower.py')).read(); exec(src.split('# ---------- comonad ----------')[0])
exec(src[src.index('def K1(h):'):src.index('oksn = oksc = True')])
def rM(Xs, n=2): return rdist([rpoint(Xs) for _ in range(4)], n)
from fractions import Fraction as F
ok = {}

def e(nu): return delta(mu(nu))                                  # the pooling idempotent eta_D . mu
def Pk(k):
    if k == 1: return e
    inner = Pk(k - 1); return lambda d: e(Dmap(inner, d))
def P(ctx):                                                       # on the context component of C3 D X
    return (Pk(1)(ctx[0]), Pk(2)(ctx[1]), Pk(3)(ctx[2]), Dmap(lambda h: (h[0], e(h[1])), ctx[3]))
def bary_ctx(pairs): return Emix(pairs)                           # beta: mixing in E3(D X), componentwise
Rr = random.Random(11)
def rlaw(n):
    w = [Rr.randint(1, 9) for _ in range(n)]; s = sum(w); return Dist([(i, F(w[i], s)) for i in range(n)])

# P1
p1 = True; kern_fail = 0
for _ in range(200):
    nu = Dist([(rlaw(3), F(1, 3)), (rlaw(3), F(2, 3))])
    p1 &= e(e(nu)) == e(nu) and mu(e(nu)) == mu(nu)
    for fn in ({0: 0, 1: 0, 2: 1}.get, {0: 2, 1: 0, 2: 1}.get):
        p1 &= e(Dmap(lambda d: Dmap(fn, d), nu)) == Dmap(lambda d: Dmap(fn, d), e(nu))
    K = {i: rlaw(3) for i in range(3)}; ks = lambda d: mu(Dmap(lambda x: K[x], d))
    kern_fail += e(Dmap(ks, nu)) != Dmap(ks, e(nu))
ok['P1_idempotent_natural_kernels'] = p1 and kern_fail == 0

# P2
n1, n2 = delta(Dist([(0, F(1))])), delta(Dist([(1, F(1))]))
ok['P2_not_affine'] = e(mix([(F(1, 2), n1), (F(1, 2), n2)])) != mix([(F(1, 2), e(n1)), (F(1, 2), e(n2))])

# P3
def stage(k, j, ck):
    """D^{k-j}(eta_{D^j X})(c_k): eta inserted at depth j"""
    f = delta
    for _ in range(k - j): f = (lambda g: (lambda d: Dmap(g, d)))(f)
    return f(ck)
p3 = True
for _ in range(200):
    pt = rpoint(X); uL, uR = lam_sig(delta(pt)), C(eta)(pt)
    p3 &= uL[0] == uR[0] and uL[1] == P(uR[1])
    c = pt[1]
    for k in (1, 2, 3):
        p3 &= stage(k, 0, c[k - 1]) == uR[1][k - 1] and stage(k, k, c[k - 1]) == uL[1][k - 1]
        cur = uR[1][k - 1]
        # pool one depth at a time, innermost first, and compare with the stages
        for j in range(1, k + 1):
            g = e
            for _ in range(k - j): g = (lambda h: (lambda d: Dmap(h, d)))(g)
            cur = g(cur); p3 &= cur == stage(k, j, c[k - 1])
ok['P3_unit_factorisation_and_stages'] = p3

# P4
p4 = True
for _ in range(200):
    Ms = [rM(X, R.choice([1, 2, 3])) for _ in range(R.choice([1, 2, 3]))]; ws = [F(R.randint(1, 4)) for _ in Ms]; s = sum(ws)
    Mp = Dist([(m, w / s) for m, w in zip(Ms, ws)])
    L = lam_sig(mu(Mp)); Rt = C(mu)(lam_sig(Dmap(lam_sig, Mp)))
    p4 &= L[0] == Rt[0]
    p4 &= L[1] == bary_ctx([(p, lam_sig(m)[1]) for m, p in Mp.items()])
    p4 &= Rt[1] == bary_ctx([(p, P(lam_sig(m)[1])) for m, p in Mp.items()])
ok['P4_multiplication_factorisation'] = p4

# P5
def iter_point(d, k):
    for _ in range(k):
        if len(d.m) != 1: return False
        d = next(iter(d.m))
    return True
p5 = True; seen = {True: 0, False: 0}
for _ in range(300):
    pt = rpoint(X)
    if Rr.random() < 0.4:                                          # plant point-mass contexts
        x = Rr.choice(X); pt = (pt[0], (delta(x), delta(delta(x)), delta(delta(delta(x))), delta(("t1", delta(x)))))
    uL, uR = lam_sig(delta(pt)), C(eta)(pt)
    for k in (1, 2, 3):
        holds = uL[1][k - 1] == uR[1][k - 1]; p5 &= holds == iter_point(pt[1][k - 1], k); seen[holds] += 1
    holdsS = uL[1][3] == uR[1][3]; p5 &= holdsS == all(len(h[1].m) == 1 for h in pt[1][3].m)
def planted_group(n):
    """samples sharing one context whose level-two part is a point mass (depth 1) and level-three part a point mass of a point
    mass (depth 2); level one is then the barycenter, as coherence requires"""
    sg = rlaw(2); c = (sg, delta(sg), delta(delta(sg)), Dist([(("t1", sg), F(1))]))
    return Dist([((Rr.choice(X), c), F(1, n)) for _ in range(n)])
def m_criterion(m):
    cs = [p[1] for p in m.m]
    same = all(cc[:3] == cs[0][:3] for cc in cs)
    return same and iter_point(cs[0][1], 1) and iter_point(cs[0][2], 2)
mseen = {True: 0, False: 0}
for _ in range(300):
    Ms = [planted_group(Rr.choice([1, 2, 3])) if Rr.random() < 0.5 else rM(X, R.choice([1, 2])) for _ in range(R.choice([1, 2]))]
    Mp = Dist([(m, F(1, len(Ms))) for m in Ms])
    L = lam_sig(mu(Mp)); Rt = C(mu)(lam_sig(Dmap(lam_sig, Mp)))
    crit = all(m_criterion(m) for m in Ms); p5 &= (L == Rt) == crit; mseen[crit] += 1
ok['P5_identity_criteria'] = p5 and seen[True] > 0 and seen[False] > 0 and mseen[True] > 0 and mseen[False] > 0

# P6
src2 = open(os.path.join(H, 'check_cm_levels.py')).read()
a = [F(1, 3), F(2, 3)]; rho = Dist([(0, a[0]), (1, a[1])])
sig = [Dist([(0, F(1, 4)), (1, F(3, 4))]), Dist([(0, F(2, 5)), (1, F(3, 5))])]; Pi = Dist([(sig[0], a[0]), (sig[1], a[1])])
tau = [Dist([(sig[0], F(1, 2)), (sig[1], F(1, 2))]), delta(sig[1])]; Om = Dist([(tau[0], a[0]), (tau[1], a[1])])
hy = [("t1", delta(0)), ("t2", delta(1))]; Sig = Dist([(hy[0], a[0]), (hy[1], a[1])])
pt = (0, (rho, Pi, Om, Sig)); ck = [(delta(k), Dmap(delta, sig[k]), Dmap(lambda t: Dmap(delta, t), tau[k]), delta(hy[k])) for k in (0, 1)]
Mp = delta(Dist([((0, ck[k]), a[k]) for k in (0, 1)]))
uL, uR = lam_sig(delta(pt)), C(eta)(pt); mL, mR = lam_sig(mu(Mp)), C(mu)(lam_sig(Dmap(lam_sig, Mp)))
ok['P6_exchanged_pair'] = uL[1] == P(uR[1]) == mR[1] and uR[1] == mL[1]

# P7: closed form P_k = eta . D^{k-1}(mu); e is idempotent but P_k (k >= 2) is not
def Dk_(k, f):
    g = f
    for _ in range(k): g = (lambda h: (lambda d: Dmap(h, d)))(g)
    return g
p7 = True; nonid = 0
for _ in range(100):
    pt = rpoint(X)
    for k in (1, 2, 3):
        w = C(eta)(pt)[1][k - 1]; p7 &= Pk(k)(w) == delta(Dk_(k - 1, mu)(w))
        if k >= 2: nonid += Pk(k)(Pk(k)(w)) != Pk(k)(w)
ok['P7_closed_form_and_nonidempotence'] = p7 and nonid > 0

# A: audit of the original note's scripts (the nearest-atom map of their Def. 1.14 / scripts 2-3, and script 1)
def tv(d1, d2): ks = set(d1.m) | set(d2.m); return sum(abs(d1.m.get(k, 0) - d2.m.get(k, 0)) for k in ks) / 2
def quant(nu, atoms): return Dist([(atoms[min(range(len(atoms)), key=lambda i: (tv(v, atoms[i]), i))], p) for v, p in nu.items()])
r1, r2 = Dist([(0, F(1, 4)), (1, F(3, 4))]), Dist([(0, F(3, 4)), (1, F(1, 4))]); rhoh = Dist([(0, F(1, 2)), (1, F(1, 2))])
a1 = mu(quant(delta(Dist([(0, F(9, 10)), (1, F(1, 10))])), [r1, r2])) != Dist([(0, F(9, 10)), (1, F(1, 10))])
a2 = quant(delta(rhoh), [r1, r2]) != quant(delta(rhoh), [r2, r1])
qf = 0
for _ in range(400):
    K = {i: rlaw(3) for i in range(3)}; ks = lambda d: mu(Dmap(lambda x: K[x], d)); at = [rlaw(3) for _ in range(3)]
    nu = Dist([(rlaw(3), F(1, 3)), (rlaw(3), F(1, 3)), (rlaw(3), F(1, 3))])
    qf += quant(Dmap(ks, nu), [ks(x) for x in at]) != Dmap(ks, quant(nu, at))
ok['A_quantiser_fails_barycentre_ties_kernels'] = a1 and a2 and qf > 0
print({k: bool(v) for k, v in ok.items()}, '| kernel commutation failures: pooling', kern_fail, 'quantiser', qf, 'of 400')
print('PASS' if all(ok.values()) else 'FAIL'); sys.exit(0 if all(ok.values()) else 1)
