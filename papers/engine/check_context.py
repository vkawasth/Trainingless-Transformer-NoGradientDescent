"""Checks for Step 1 of "Probability, Context and Differentiation: The Layers" (exact rational arithmetic unless stated).
  DL1  distributive law, unit axiom:      lam . W eta = eta W                       (W X -> D W X)
  DL2  distributive law, multiplication:  lam . W mu = mu W . D lam . lam D         (W D D X -> D W X)
  DL3  distributive law, counit:          D eps . lam = eps D                       (W D X -> D X)
  DL4  distributive law, comultiplication: D delta . lam = lam W . W lam . delta D  (W D X -> D W W X)
  DL5  the two-sided Kleisli category: composite = sum_y f(l, x)(y) g(l, y) (one context throughout), associative, identity eta . eps
  DL6  marginalisation: D of projections, so iterated = direct; affine in the law (exact for mixtures); commutes with forgetting context
  DL7  conditioning is partial (undefined exactly where p(A) = 0); total probability p = sum_z p(Y = z) C_z p; iterated C_B C_A = C_{A n B}
  DL8  conditioning and marginalisation: the square commutes when A is a cylinder over the kept coordinates; a counterexample otherwise
  DL9  context: conditioning a point-context law keeps its label; contextual Bayes decomposition C_A(sum pi_l p_l) = sum pi_l^A C_A(p_l),
       and pooling-then-conditioning differs from conditioning-within-sources-then-pooling-with-the-old-weights
  DL10 forgetting context is marginalisation over L: it commutes with every window marginalisation, but it is not injective: two labelled
       families with the same pooled law and the same pooled window tables differ in every labelled window (a source contrast)
  DL11 derivatives (floating point): marginalisation has D^2 = 0; DC_A and D^2C_A match the closed forms; |DC_A| grows like 1/p(A)
  DL12 no continuous conditional across the wall: along rays into a law with p(A) = 0, C_A is constant and takes every value of the
       simplex on A, so no extension of C_A to the wall is continuous (exact); a FinStoch conditional exists there, with arbitrary values
  DL13 the worked examples of the paper, exactly: source weights 1/4, 3/4 become 8/23, 15/23; P(H=1 | R=1) is 9/23, not 63/160; the
       conditioning-marginalisation square with the non-cylinder event H = 1 gives 9/19 against 23/40
"""
import itertools, os, sys, json, random
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import context as CT
random.seed(20261031); results = []; OUT = {}
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
def rdist(support):
    w = [Fr(random.randint(1, 9)) for _ in support]; s = sum(w); return CT.clean({x: v / s for x, v in zip(support, w)})
X = ["a", "b", "c"]; L = ["l1", "l2"]

# DL1-DL4
ok = [True] * 4
for _ in range(50):
    l = random.choice(L); x = random.choice(X)
    ok[0] &= CT.lam(l, CT.eta(x)) == CT.eta((l, x))
    ps = [rdist(X) for _ in range(3)]
    wts = rdist([0, 1, 2]); Phi = {}
    for i, w in wts.items(): k = CT.freeze(ps[i]); Phi[k] = Phi.get(k, Fr(0)) + w
    lhs = CT.lam(l, CT.mu(Phi))
    lamD = CT.lam(l, Phi)                                                  # D(W D X): distribution over (l, frozen p)
    Dlam = {}
    for (ll, k), w in lamD.items():
        kk = CT.freeze(CT.lam(ll, CT.thaw(k))); Dlam[kk] = Dlam.get(kk, Fr(0)) + w
    ok[1] &= lhs == CT.mu(Dlam)
    p = rdist(X)
    ok[2] &= CT.dmap(CT.eps, CT.lam(l, p)) == p
    lhs4 = CT.dmap(CT.delta, CT.lam(l, p))
    l2, (l3, pp) = (l, (l, p)); rhs4 = CT.lam(l2, CT.lam(l3, pp))            # delta D, then W lam, then lam W
    ok[3] &= lhs4 == rhs4
names = ["DL1 unit axiom: lam . W eta = eta W", "DL2 multiplication axiom: lam . W mu = mu W . D lam . lam D",
         "DL3 counit axiom: D eps . lam = eps D", "DL4 comultiplication axiom: D delta . lam = lam W . W lam . delta D"]
for n, o in zip(names, ok): check(n, o, "exact, 50 random elements")

# DL5
def rker(Y):
    tab = {(l, x): rdist(Y) for l in L for x in X}; return lambda lx: tab[lx]
Y = ["u", "v"]; Z = ["s", "t", "w"]; okc = True
for _ in range(20):
    f = rker(Y); g = (lambda tab: (lambda ly: tab[ly]))({(l, y): rdist(Z) for l in L for y in Y}); h = (lambda tab: (lambda lz: tab[lz]))({(l, z): rdist(["p", "q"]) for l in L for z in Z})
    gf = CT.bk_compose(f, g)
    for l in L:
        for x in X:
            expl = {}
            for y, v in f((l, x)).items():
                for z, w in g((l, y)).items(): expl[z] = expl.get(z, Fr(0)) + v * w
            okc &= gf((l, x)) == CT.clean(expl)
            okc &= CT.bk_compose(CT.bk_compose(f, g), h)((l, x)) == CT.bk_compose(f, CT.bk_compose(g, h))((l, x))
            ident = lambda lx: CT.eta(CT.eps(lx)); okc &= CT.bk_compose(ident, f)((l, x)) == f((l, x)) == CT.bk_compose(f, ident)((l, x))
check("DL5 two-sided Kleisli category: composite uses one context throughout; associative; identity eta . eps", okc)

# DL6
X3 = list(itertools.product((0, 1), repeat=3)); ok6 = True
for _ in range(40):
    p = rdist(X3); q = rdist(X3); t = Fr(random.randint(1, 9), 10)
    ok6 &= CT.marg(CT.marg(p, [0, 2]), [0]) == CT.marg(p, [0])
    ok6 &= CT.marg(CT.pool([t, 1 - t], [p, q]), [1, 2]) == CT.pool([t, 1 - t], [CT.marg(p, [1, 2]), CT.marg(q, [1, 2])])
    lab = CT.pool([t, 1 - t], [CT.lam("l1", p), CT.lam("l2", q)])                  # a law on L x X^3
    ok6 &= CT.marg(CT.dmap(CT.eps, lab), [0, 1]) == CT.dmap(CT.eps, CT.dmap(lambda lx: (lx[0], (lx[1][0], lx[1][1])), lab))
check("DL6 marginalisation: iterated = direct; affine; commutes with forgetting context", ok6, "exact")

# DL7
ok7 = True
for _ in range(40):
    p = rdist(X3); A = lambda x: x[0] == 1; B = lambda x: x[1] == 0
    zero = CT.clean({x: v for x, v in p.items() if x[2] == 0}); s = sum(zero.values()); zero = {x: v / s for x, v in zero.items()}
    ok7 &= CT.cond(zero, lambda x: x[2] == 1) is None and CT.cond(zero, lambda x: x[2] == 0) is not None
    tot = CT.pool([CT.prob(p, lambda x, z=z: x[2] == z) for z in (0, 1)], [CT.cond(p, lambda x, z=z: x[2] == z) for z in (0, 1)])
    ok7 &= tot == p
    ok7 &= CT.cond(CT.cond(p, A), B) == CT.cond(p, lambda x: A(x) and B(x))
check("DL7 conditioning: partial (undefined exactly at p(A) = 0); total probability; iterated conditioning", ok7, "exact")

# DL8
ok8 = True; counter = None
for _ in range(40):
    p = rdist(X3); Acyl = lambda x: x[0] == 1; Anon = lambda x: x[2] == 1
    ok8 &= CT.marg(CT.cond(p, Acyl), [0, 1]) == CT.cond(CT.marg(p, [0, 1]), lambda y: y[0] == 1)
    lhs = CT.marg(CT.cond(p, Anon), [0, 1]); rhs = CT.marg(p, [0, 1])
    if lhs != rhs and counter is None: counter = (lhs, rhs)
check("DL8 conditioning and marginalisation commute when A is a cylinder over the kept coordinates; a non-cylinder event breaks it", ok8 and counter is not None)

# DL9
ok9 = True; differs = 0
for _ in range(40):
    p = rdist(X3); ok9 &= CT.cond(CT.lam("l1", p), lambda lx: lx[1][0] == 1) == CT.lam("l1", CT.cond(p, lambda x: x[0] == 1))
    ps = [rdist(X3) for _ in range(3)]; pis = list(rdist([0, 1, 2]).values()); A = lambda x: x[0] == 1 and x[1] == 0
    if len(pis) < 3 or any(CT.prob(q, A) == 0 for q in ps): continue
    lhs = CT.cond(CT.pool(pis, ps), A); piA = CT.bayes_weights(pis, ps, A)
    ok9 &= lhs == CT.pool(piA, [CT.cond(q, A) for q in ps]); differs += lhs != CT.pool(pis, [CT.cond(q, A) for q in ps])
check("DL9 conditioning keeps a point context; contextual Bayes decomposition (exact); the old weights are wrong in general", ok9 and differs > 0, f"old weights wrong in {differs} cases")

# DL10
ok10 = True; contrast = 0
P = rdist(X3); Q = {x: Fr(random.choice([-1, 1]), 40) for x in X3}; Q = {x: v - sum(Q.values()) / 8 for x, v in Q.items()}
p1 = CT.clean({x: P[x] + Q[x] for x in X3}); p2 = CT.clean({x: P[x] - Q[x] for x in X3})
famA = CT.pool([Fr(1, 2), Fr(1, 2)], [CT.lam("l1", P), CT.lam("l2", P)]); famB = CT.pool([Fr(1, 2), Fr(1, 2)], [CT.lam("l1", p1), CT.lam("l2", p2)])
ok10 &= CT.dmap(CT.eps, famA) == CT.dmap(CT.eps, famB) == P
for W in ([0, 1], [0, 2], [1, 2]):
    ok10 &= CT.marg(CT.dmap(CT.eps, famA), W) == CT.marg(CT.dmap(CT.eps, famB), W)
    labA = CT.dmap(lambda lx: (lx[0], tuple(lx[1][i] for i in W)), famA); labB = CT.dmap(lambda lx: (lx[0], tuple(lx[1][i] for i in W)), famB)
    contrast += labA != labB
check("DL10 forgetting context = marginalising over L: commutes with window marginals; not injective (same pooled law and windows, different labelled windows)",
      ok10 and contrast == 3, f"labelled windows differ on {contrast}/3")

# DL11
rng = np.random.default_rng(1); okd = True; ratios = []
for _ in range(30):
    p = rng.dirichlet(np.ones(8)); h = rng.normal(size=8); h -= h.mean(); k = rng.normal(size=8); k -= k.mean(); N = np.array([1, 1, 0, 1, 0, 0, 1, 0.0])
    a = lambda q: (N * q).sum(); C = lambda q: N * q / a(q); e = 1e-5
    D1 = N * h / a(p) - N * p * a(h) / a(p) ** 2
    D2 = -(N * h * a(k) + N * k * a(h)) / a(p) ** 2 + 2 * N * p * a(h) * a(k) / a(p) ** 3
    num1 = (C(p + e * h) - C(p - e * h)) / (2 * e)
    num2 = (C(p + e * h + e * k) - C(p + e * h - e * k) - C(p - e * h + e * k) + C(p - e * h - e * k)) / (4 * e * e)
    okd &= np.abs(num1 - D1).max() < 1e-6 and np.abs(num2 - D2).max() < 1e-3
    M = lambda q: np.array([q[:4].sum(), q[4:].sum()]); okd &= np.abs(M(p + e * h + e * k) - M(p + e * h - e * k) - M(p - e * h + e * k) + M(p - e * h - e * k)).max() < 1e-12
p = rng.dirichlet(np.ones(8)); h = rng.normal(size=8); h -= h.mean()
for s in (1e-1, 1e-2, 1e-3, 1e-4):
    q = p.copy(); q[N == 1] *= s / q[N == 1].sum(); q[N == 0] *= (1 - s) / q[N == 0].sum()
    ratios.append(np.abs(N * h / s - N * q * (N * h).sum() / s ** 2).max() * s)
check("DL11 marginalisation has D^2 = 0; DC_A and D^2C_A match the closed forms; |DC_A| p(A) bounded as p(A) -> 0", okd and max(ratios) / min(ratios) < 50,
      f"|DC_A| p(A) over four decades: {np.round(ratios, 3).tolist()}")
# DL12
lims = set()
for al in (Fr(1, 5), Fr(1, 2), Fr(4, 5)):
    vals = set()
    for t in (Fr(1, 10), Fr(1, 1000), Fr(1, 10 ** 6)):
        p = CT.clean({"a": t * al, "b": t * (1 - al), "c": 1 - t}); vals.add(CT.freeze(CT.cond(p, lambda x: x in ("a", "b"))))
    lims.add(tuple(vals))
check("DL12 no continuous conditional across the wall: C_A is constant on each ray into the wall, and different rays give different limits", all(len(v) == 1 for v in lims) and len(lims) == 3)

# DL13
p1 = {(1, 1): Fr(3, 10), (1, 0): Fr(1, 2), (0, 1): Fr(1, 10), (0, 0): Fr(1, 10)}; p2 = {(1, 1): Fr(1, 5), (1, 0): Fr(3, 10), (0, 1): Fr(3, 10), (0, 0): Fr(1, 5)}
pis = [Fr(1, 4), Fr(3, 4)]; A = lambda x: x[0] == 1; pooled = CT.pool(pis, [p1, p2])
ok13 = CT.bayes_weights(pis, [p1, p2], A) == [Fr(8, 23), Fr(15, 23)]
ok13 &= CT.prob(CT.cond(pooled, A), lambda x: x[1] == 1) == Fr(9, 23) and CT.prob(CT.pool(pis, [CT.cond(p1, A), CT.cond(p2, A)]), lambda x: x[1] == 1) == Fr(63, 160)
ok13 &= CT.marg(CT.cond(pooled, lambda x: x[1] == 1), [0])[(1,)] == Fr(9, 19) and CT.marg(pooled, [0])[(1,)] == Fr(23, 40)
check("DL13 the paper's worked examples, exactly (8/23, 15/23; 9/23 against 63/160; 9/19 against 23/40)", ok13)
json.dump(OUT, open(os.path.join(HERE, "..", "lab", "context.json"), "w"))
print(f"{sum(results)}/{len(results)} passed")
