"""check_glue.py -- computational checks for Section 7 ("The glue") of
"Precise Attribution of Model Interventions", draft 4.

Groups
  K  kernel 1-cells (Kleisli extension): affine, functorial, compatible with flattening,
     non-expansive, attribution can only lose precision along a 1-cell.
  C  coupling 2-cells: gluing is associative and unital (exact), cost is subadditive,
     minimal cost = TV, and the two whiskerings: which laws hold and which fail (counterexamples).
  S  metric shadow: the hom-pseudometric d(f,g) = sup_x TV(f(x), g(x)) and its composition laws.
  A  assembly: two-context gluing at levels 1 and 2, contextual fraction = mu_min,
     approximate gluing eps* and its bounds.
  G  geometry: Fisher = 4 x round metric under sqrt, Fisher-Rao distance, Lagrangian torus fibres
     of the moment map of CP^2.

Exact checks use fractions.Fraction; LPs use floats with exactly verified certificates where stated.
"""
import itertools, math, random
from fractions import Fraction as F
import numpy as np
from scipy.optimize import linprog
from provenance import tv, mu_min_sources, attribution_report, coarsen

random.seed(7)
RESULTS = []
def check(name, ok, info=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"   {info}" if info else ""))

# ------------------------------------------------------------------ helpers
def rdist(n, zeros=False):
    v = [F(random.randint(0 if zeros else 1, 6)) for _ in range(n)]
    if sum(v) == 0: v[0] = F(1)
    s = sum(v); return [a / s for a in v]
def rkernel(nx, ny): return [rdist(ny) for _ in range(nx)]
def kext(k, p):                       # Kleisli extension k^# : D(X) -> D(Y)
    return [sum(p[x] * k[x][y] for x in range(len(p))) for y in range(len(k[0]))]
def kcomp(k, h): return [kext(h, row) for row in k]          # h o k
def eta(n): return [[F(int(i == j)) for j in range(n)] for i in range(n)]
def flatten(P): return [sum(w * p[b] for w, p in P) for b in range(len(P[0][1]))]
def Dmap(fn, P): return [(w, fn(p)) for w, p in P]
def rows(a): return [sum(r) for r in a]
def cols(a): return [sum(a[i][j] for i in range(len(a))) for j in range(len(a[0]))]
def matmix(ws, ms):
    return [[sum(w * m[i][j] for w, m in zip(ws, ms)) for j in range(len(ms[0][0]))] for i in range(len(ms[0]))]

# couplings: a coupling of f in D(Y) and g in D(Y') is a matrix a[y][y'] with rows f, cols g
def glue(a, b):
    """vertical composition: glue a (rows f, cols g) and b (rows g, cols h) along g, forget g"""
    g = cols(a); assert g == rows(b)
    return [[sum(a[i][m] * b[m][j] / g[m] for m in range(len(g)) if g[m] > 0)
             for j in range(len(b[0]))] for i in range(len(a))]
def diag(f): return [[f[i] if i == j else F(0) for j in range(len(f))] for i in range(len(f))]
def cost(a): return 1 - sum(a[i][i] for i in range(len(a)))
def maximal(f, g):
    m = [min(x, y) for x, y in zip(f, g)]; t = 1 - sum(m)
    return [[(m[i] if i == j else F(0)) + ((f[i] - m[i]) * (g[j] - m[j]) / t if t > 0 else F(0))
             for j in range(len(g))] for i in range(len(f))]
def rcoupling_from(f_or_g, n, side="rows"):
    """random coupling with a given row marginal: rows g, cols drawn from a random kernel"""
    K = rkernel(len(f_or_g), n)
    return [[f_or_g[i] * K[i][j] for j in range(n)] for i in range(len(f_or_g))]

n = 4
# ================================================================== K: kernel 1-cells
print("=" * 72); print("K  kernel 1-cells"); print("=" * 72)
ok = True
for _ in range(50):
    k = rkernel(n, 5); p, q = rdist(n), rdist(n); t = F(random.randint(0, 9), 9)
    ok &= kext(k, [t * a + (1 - t) * b for a, b in zip(p, q)]) == [t * a + (1 - t) * b for a, b in zip(kext(k, p), kext(k, q))]
check("K1 k^# is affine (exact)", ok)
ok = True
for _ in range(50):
    k, h, p = rkernel(n, 5), rkernel(5, 3), rdist(n)
    ok &= kext(kcomp(k, h), p) == kext(h, kext(k, p)) and kext(eta(n), p) == p
check("K2 functorial: (h o k)^# = h^# k^#, eta^# = id (exact)", ok)
ok = True
for _ in range(30):
    k = rkernel(n, 5); P = [(w, rdist(n)) for w in rdist(3)]
    ok &= kext(k, flatten(P)) == flatten(Dmap(lambda p: kext(k, p), P))
check("K3 compatible with flattening: k^# mu = mu D(k^#) on level-2 states (exact)", ok)
ok = True
for _ in range(200):
    k = rkernel(n, 5); p, q = rdist(n, True), rdist(n, True)
    ok &= tv(kext(k, p), kext(k, q)) <= tv(p, q)
check("K4 non-expansive: TV(k^# p, k^# q) <= TV(p, q) (exact, 200 cases)", ok)

# K5 attribution along a 1-cell, on the demo's stage-3 state at x1
def d10(*v): return [F(a, 10) for a in v]
A1, A2, NN = d10(4, 3, 2, 1, 0, 0), d10(0, 1, 2, 3, 4, 0), d10(0, 0, 0, 2, 3, 5)
w2 = (F(9, 20), F(1, 4), F(3, 10))
p2 = [sum(w * s[b] for w, s in zip(w2, (A1, A2, NN))) for b in range(6)]
p3 = [F(4, 5) * a + (F(1, 5) if b == 2 else 0) for b, a in enumerate(p2)]    # stage-3 edit at x1
groups = [[0, 1], [2, 3], [4, 5]]
coarse = [[F(int(b in g)) for g in groups] for b in range(6)]                 # deterministic kernel
noisy = [[F(9, 10) * F(int(b in g)) + F(1, 30) for g in groups] for b in range(6)]   # stochastic kernel
ok, info = True, []
for name, k in (("coarsening", coarse), ("noisy coarsening", noisy)):
    m0, lb0, _ = mu_min_sources(p3, [A1, A2, NN])
    m1, lb1, _ = mu_min_sources(kext(k, p3), [kext(k, s) for s in (A1, A2, NN)])
    r0 = attribution_report(p3, [("exact", s) for s in (A1, A2, NN)], 0.25)
    r1 = attribution_report(kext(k, p3), [("exact", kext(k, s)) for s in (A1, A2, NN)], 0.25)
    contain = all(a1[0] <= a0[0] + 1e-9 and a1[1] >= a0[1] - 1e-9 for a0, a1 in zip(r0["lambda"], r1["lambda"]))
    ok &= m1 <= m0 + 1e-9 and contain
    info.append(f"{name}: mu_min {m0:.3f} -> {m1:.3f}, widths {[round(w, 3) for w in r0['widths']]} -> {[round(w, 3) for w in r1['widths']]}")
check("K5 along a 1-cell mu_min can only fall and every weight interval can only widen", ok)
for s in info: print("     " + s)

# ================================================================== C: coupling 2-cells
print("=" * 72); print("C  coupling 2-cells (witnesses)"); print("=" * 72)
ok = True
for _ in range(40):
    f = rdist(4); a = rcoupling_from(f, 4); g = cols(a)
    b = rcoupling_from(g, 4); h = cols(b); c = rcoupling_from(h, 4)
    ba = glue(a, b)
    ok &= rows(ba) == f and cols(ba) == h
check("C1 glued coupling has the outer marginals (exact)", ok)
ok = True
for _ in range(40):
    f = rdist(4, True); a = rcoupling_from(f, 4); g = cols(a)
    b = rcoupling_from(g, 4); h = cols(b); c = rcoupling_from(h, 4)
    ok &= glue(glue(a, b), c) == glue(a, glue(b, c))
check("C2 vertical composition (gluing) is associative (exact, zeros allowed)", ok)
ok = True
for _ in range(40):
    f = rdist(4, True); a = rcoupling_from(f, 4); g = cols(a)
    ok &= glue(diag(f), a) == a and glue(a, diag(g)) == a
check("C3 diagonal couplings are identities (exact)", ok)
ok = True
for _ in range(200):
    f = rdist(4, True); a = rcoupling_from(f, 4); b = rcoupling_from(cols(a), 4)
    ok &= cost(glue(a, b)) <= cost(a) + cost(b)
check("C4 cost P(y != y') is subadditive under gluing (exact, 200 cases)", ok)
ok = True; worst = 0
for _ in range(30):
    f, g = rdist(4, True), rdist(4, True)
    mc = maximal(f, g)
    ok &= rows(mc) == f and cols(mc) == g and cost(mc) == tv(f, g)
    A_eq = []; b_eq = []
    for i in range(4):
        r = np.zeros(16); r[4 * i:4 * i + 4] = 1; A_eq.append(r); b_eq.append(float(f[i]))
        r = np.zeros(16); r[i::4] = 1; A_eq.append(r); b_eq.append(float(g[i]))
    cvec = np.array([0.0 if i == j else 1.0 for i in range(4) for j in range(4)])
    lp = linprog(cvec, A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=[(0, None)] * 16)
    worst = max(worst, abs(lp.fun - float(tv(f, g))))
check("C5 min cost over couplings = TV, attained by the maximal coupling", ok and worst < 1e-9, f"max LP gap {worst:.1e}")

# whiskering by precomposition with a 1-cell u: W -> D(X); 2-cells are kernels X -> D(Y x Y)
def gluek(A, B): return [glue(a, b) for a, b in zip(A, B)]
def pre(A, u): return [matmix(urow, A) for urow in u]
ok = True
for _ in range(30):
    X = 3; fK = rkernel(X, 3)
    A = [rcoupling_from(fK[x], 3) for x in range(X)]; B = [rcoupling_from(cols(A[x]), 3) for x in range(X)]
    u = [[F(int(j == random.randrange(X))) for j in range(X)] for _ in range(4)]
    u = [[F(int(j == r)) for j in range(X)] for r in [random.randrange(X) for _ in range(4)]]
    ok &= pre(gluek(A, B), u) == gluek(pre(A, u), pre(B, u))
check("C6 whiskering by a deterministic precomposition is strictly functorial (exact)", ok)
uni = [F(1, 2), F(1, 2)]
I2, SW = diag(uni), [[F(0), F(1, 2)], [F(1, 2), F(0)]]
A, B = [I2, SW], [I2, SW]                    # alpha(x0)=id, alpha(x1)=swap; same for beta
u = [[F(1, 2), F(1, 2)]]                    # stochastic 1-cell: a fair coin over the two probes
lhs, rhs = pre(gluek(A, B), u)[0], gluek(pre(A, u), pre(B, u))[0]
check("C7 whiskering by a stochastic precomposition is NOT functorial (counterexample)",
      lhs != rhs, f"(beta.alpha)u has cost {cost(lhs)}, (beta u).(alpha u) has cost {cost(rhs)}")
# whiskering by postcomposition with h: Y -> D(Z): same-input pairs copied, different-input pairs independent
def post(h, a):
    nz = len(h[0])
    out = [[F(0)] * nz for _ in range(nz)]
    for y in range(len(a)):
        for y2 in range(len(a)):
            if a[y][y2] == 0: continue
            for z in range(nz):
                if y == y2: out[z][z] += a[y][y] * h[y][z]
                else:
                    for z2 in range(nz): out[z][z2] += a[y][y2] * h[y][z] * h[y2][z2]
    return out
ok1 = ok2 = ok3 = True
for _ in range(60):
    f = rdist(4, True); a = rcoupling_from(f, 4); h = rkernel(4, 3)
    ok1 &= post(h, diag(f)) == diag(kext(h, f))
    ha = post(h, a)
    ok2 &= rows(ha) == kext(h, f) and cols(ha) == kext(h, cols(a))
    ok3 &= cost(ha) <= cost(a)
check("C8 postcomposition whiskering: preserves identities, has the right marginals, cost non-expansive",
      ok1 and ok2 and ok3)
nu = [[F(1, 2), F(1, 2)], [F(1, 2), F(1, 2)]]  # h forgets its input
lhs, rhs = post(nu, glue(SW, SW)), glue(post(nu, SW), post(nu, SW))
check("C9 postcomposition whiskering is NOT functorial (swap.swap = id; noise forgets it)",
      lhs != rhs, f"h(beta.alpha) cost {cost(lhs)}, (h beta).(h alpha) cost {cost(rhs)}")
found = None
for _ in range(400):
    f = rdist(3, True); a = rcoupling_from(f, 3); b = rcoupling_from(cols(a), 3)
    hdet = [[F(1), F(0)], [F(1), F(0)], [F(0), F(1)]]
    if post(hdet, glue(a, b)) != glue(post(hdet, a), post(hdet, b)): found = (a, b); break
check("C10 even a deterministic postcomposition (merging outcomes) is not functorial", found is not None,
      "random search found a counterexample" if found else "")

# ================================================================== S: metric shadow
print("=" * 72); print("S  metric shadow d(f,g) = sup_x TV(f(x), g(x))"); print("=" * 72)
def dK(f, g): return max(tv(a, b) for a, b in zip(f, g))
okT = okL = okR = okH = True
for _ in range(100):
    f, g, k = rkernel(3, 4), rkernel(3, 4), rkernel(3, 4)
    h, h2, u = rkernel(4, 3), rkernel(4, 3), rkernel(2, 3)
    okT &= dK(f, k) <= dK(f, g) + dK(g, k)
    okL &= dK(kcomp(f, h), kcomp(g, h)) <= dK(f, g)
    okR &= dK(kcomp(u, f), kcomp(u, g)) <= dK(f, g)
    okH &= dK(kcomp(f, h), kcomp(g, h2)) <= dK(f, g) + dK(h, h2)
check("S1 triangle inequality (vertical composition of the shadow)", okT)
check("S2 both whiskerings are non-expansive for kernel 1-cells", okL and okR)
check("S3 horizontal composition: d(h f, h' g) <= d(f, g) + d(h, h')  (Lawvere-enriched)", okH)

# ================================================================== A: assembly
print("=" * 72); print("A  assembly of contexts"); print("=" * 72)
# A1 two contexts {a,b}, {b,c} sharing b: gluing = the coupling composition before forgetting b
e1 = [[F(1, 5), F(1, 10)], [F(1, 10), F(3, 5)]]            # e1[a][b]
eb = cols(e1)
e2 = rcoupling_from(eb, 3)                                   # e2[b][c], rows = eb
G3 = [[[e1[a][b] * e2[b][c] / eb[b] for c in range(3)] for b in range(2)] for a in range(2)]
r1 = [[sum(G3[a][b]) for b in range(2)] for a in range(2)]
r2 = [[sum(G3[a][b][c] for a in range(2)) for c in range(3)] for b in range(2)]
forget = [[sum(G3[a][b][c] for b in range(2)) for c in range(3)] for a in range(2)]
check("A1 two compatible contexts always glue; forgetting the overlap gives vertical composition",
      r1 == e1 and r2 == e2 and forget == glue(e1, e2))

# CHSH scenario: observables a0,a1,b0,b1 in {0,1}; contexts (ai, bj); global assignments s in {0,1}^4
CTX = [(i, j) for i in range(2) for j in range(2)]
ASG = list(itertools.product(range(2), repeat=4))           # (a0, a1, b0, b1)
def model(v):
    """v * PR box + (1 - v) * uniform noise, as one vector over (context, o1, o2), total mass 1"""
    out = []
    for i, j in CTX:
        for o1, o2 in itertools.product(range(2), repeat=2):
            pr = F(1, 2) if (o1 ^ o2) == (i & j) else F(0)
            out.append((v * pr + (1 - v) * F(1, 4)) / 4)
    return out
def det(s):
    return [F(int((s[i], s[2 + j]) == (o1, o2)), 4) for i, j in CTX for o1, o2 in itertools.product(range(2), repeat=2)]
DET = [det(s) for s in ASG]
cf_ok, rows_cf = True, []
for v in (F(1), F(3, 4), F(1, 2), F(1, 4)):
    m, lb, y = mu_min_sources(model(v), DET)
    target = max(2 * float(v) - 1, 0)
    cf_ok &= abs(m - target) < 1e-9 and abs(float(lb) - target) < 1e-6
    rows_cf.append(f"v={v}: CF = mu_min = {m:.4f} (certified >= {float(lb):.4f}; expected {target})")
check("A2 contextual fraction = mu_min with global assignments as sources; PR box has CF = 1", cf_ok)
for s in rows_cf: print("     " + s)
# no-signalling: the PR box is compatible on every overlap, yet has no global section
def marg_overlap(vec, ctx, side):
    k = CTX.index(ctx); blk = vec[4 * k:4 * k + 4]
    return [sum(blk[2 * o1 + o2] for o1 in range(2) for o2 in range(2) if (o1 if side == 0 else o2) == o) * 4 for o in range(2)]
pr = model(F(1))
compat = all(marg_overlap(pr, (i, 0), 0) == marg_overlap(pr, (i, 1), 0) for i in range(2)) and \
         all(marg_overlap(pr, (0, j), 1) == marg_overlap(pr, (1, j), 1) for j in range(2))
check("A3 PR box: compatible on all overlaps, but no global section (cycle of 4 contexts)", compat)

# A4 level-2 two-context gluing
def restrict_ab(G):  # G[a][b][c] -> [a][b]
    return [[sum(G[a][b]) for b in range(len(G[0]))] for a in range(len(G))]
def restrict_bc(G):
    return [[sum(G[a][b][c] for a in range(len(G))) for c in range(len(G[0][0]))] for b in range(len(G[0]))]
def glue_state(x, y):
    xb = cols(x)
    return [[[x[a][b] * y[b][c] / xb[b] if xb[b] > 0 else F(0) for c in range(len(y[0]))]
             for b in range(len(xb))] for a in range(len(x))]
def key(m): return str(m)
def as_dist(P):
    d = {}
    for w, s in P: d[key(s)] = d.get(key(s), 0) + w
    return d
qs = [[F(1, 2), F(1, 2)], [F(1, 5), F(4, 5)]]                 # the two possible overlap marginals
wq = [F(1, 3), F(2, 3)]
def with_cols(q, nrows):        # random level-1 state on {a,b} with b-marginal q
    K = rkernel(2, nrows); return [[K[b][a] * q[b] for b in range(2)] for a in range(nrows)]
P1 = []; P2 = []
for q, w in zip(qs, wq):
    for s in rdist(2): P1.append((w * s, with_cols(q, 2)))
    for t in rdist(2): P2.append((w * t, [[r[c] for c in range(3)] for r in rcoupling_from(q, 3)]))
# fibrewise independent gluing of the two level-2 states along their common overlap image, then level-1 gluing
Gl = []
for q in qs:
    f1 = [(w, s) for w, s in P1 if cols(s) == q]; f2 = [(w, s) for w, s in P2 if rows(s) == q]
    m = sum(w for w, _ in f1)
    for w1, s1 in f1:
        for w2_, s2 in f2: Gl.append((w1 * w2_ / m, glue_state(s1, s2)))
ok_restr = as_dist([(w, restrict_ab(s)) for w, s in Gl]) == as_dist(P1) and \
           as_dist([(w, restrict_bc(s)) for w, s in Gl]) == as_dist(P2)
flatG = [[[sum(w * s[a][b][c] for w, s in Gl) for c in range(3)] for b in range(2)] for a in range(2)]
mP1 = [[sum(w * s[a][b] for w, s in P1) for b in range(2)] for a in range(2)]
mP2 = [[sum(w * s[b][c] for w, s in P2) for c in range(3)] for b in range(2)]
ok_flat = restrict_ab(flatG) == mP1 and restrict_bc(flatG) == mP2
ci = glue_state(mP1, mP2)
gap = sum(abs(flatG[a][b][c] - ci[a][b][c]) for a in range(2) for b in range(2) for c in range(3)) / 2
check("A4 two compatible level-2 states glue; restrictions recover both; flattening is a level-1 gluing",
      ok_restr and ok_flat)
check("A5 level-2 gluing carries correlation that gluing the flattened states loses", gap > 0,
      f"TV(flatten(level-2 glue), CI glue of flattenings) = {float(gap):.4f}")

# A6 approximate gluing eps*: min eps s.t. some global g has TV(g|C, e_C) <= eps for every context
def eps_star(vec):
    nA, nC = len(ASG), len(CTX)
    nt = nC * 4; nv = nA + nt + 1                # g (16), t per (context, outcome) (16), eps
    A_ub, b_ub = [], []
    for k, (i, j) in enumerate(CTX):
        for o, (o1, o2) in enumerate(itertools.product(range(2), repeat=2)):
            row = np.zeros(nv)
            for si, s in enumerate(ASG):
                if (s[i], s[2 + j]) == (o1, o2): row[si] = 1
            e = float(vec[4 * k + o] * 4)
            r1 = row.copy(); r1[nA + 4 * k + o] = -1; A_ub.append(r1); b_ub.append(e)
            r2 = -row; r2[nA + 4 * k + o] = -1; A_ub.append(r2); b_ub.append(-e)
        row = np.zeros(nv); row[nA + 4 * k:nA + 4 * k + 4] = 0.5; row[-1] = -1; A_ub.append(row); b_ub.append(0)
    A_eq = [np.r_[np.ones(nA), np.zeros(nt + 1)]]
    c = np.zeros(nv); c[-1] = 1
    r = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=np.array(A_eq), b_eq=[1], bounds=[(0, None)] * nv)
    return r.fun, r.x[:nA]
e_pr, gpr = eps_star(model(F(1)))
# exact verification of the optimum 1/4: a feasible global section achieving 1/4, and a lower bound:
# for every deterministic s, the number of contexts where s violates the PR parity is odd (>= 1), so
# sum_C P_g(violate C) >= 1 and max_C TV >= max_C P_g(violate C) >= 1/4.
viol = [sum(int((s[i] ^ s[2 + j]) != (i & j)) for i, j in CTX) for s in ASG]
gstar = [F(1, 8) if v_ == 1 else F(0) for v_ in viol]           # uniform on the 8 one-violation assignments
def tv_ctx(g, vec):
    worst = F(0)
    for k, (i, j) in enumerate(CTX):
        loc = [sum(g[si] for si, s in enumerate(ASG) if (s[i], s[2 + j]) == oo) for oo in itertools.product(range(2), repeat=2)]
        worst = max(worst, tv(loc, [vec[4 * k + o] * 4 for o in range(4)]))
    return worst
check("A6 PR box: eps* = 1/4 (LP, with an exact feasible section and an exact parity lower bound)",
      abs(e_pr - 0.25) < 1e-9 and sum(gstar) == 1 and tv_ctx(gstar, model(F(1))) == F(1, 4) and all(v_ % 2 == 1 for v_ in viol),
      f"LP eps* = {e_pr:.6f}")
ok, info = True, []
for v in (F(3, 4), F(1, 2), F(9, 10)):
    es, _ = eps_star(model(v)); cfv = max(2 * float(v) - 1, 0)
    ok &= es <= cfv + 1e-9; info.append(f"v={v}: eps* = {es:.4f} <= CF = {cfv:.4f}")
check("A7 eps* <= contextual fraction", ok)
for s in info: print("     " + s)
# A8 lower bound from overlap disagreement, on a signalling model
sig = model(F(1, 2))
k = CTX.index((0, 1)); sig = sig[:]
sig[4 * k:4 * k + 4] = [F(1, 4) * 4 / 4 * x for x in (F(7, 10), F(1, 10), F(1, 10), F(1, 10))]   # a0 biased to 0 in context (0,1)
over = tv(marg_overlap(sig, (0, 0), 0), marg_overlap(sig, (0, 1), 0))
es, _ = eps_star(sig)
check("A8 eps* >= (max overlap TV)/2 on a signalling model", es >= float(over) / 2 - 1e-9,
      f"overlap TV {float(over):.3f}, eps* = {es:.4f}")

# A9 cross-context attribution on the demo: one weight vector for all probes vs per-probe weights
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import provenance_demo as DEMO
def catp(d, P): return [v / len(P) for x in P for v in d[x]]
info, ok = [], True
for P in (["x1", "x2"], ["x1", "x2", "x3"]):
    mg, lbg, yg = mu_min_sources(catp(DEMO.M2, P), [catp(S, P) for S in (DEMO.A1, DEMO.A2, DEMO.N)])
    ml = max(mu_min_sources(DEMO.M2[x], [S[x] for S in (DEMO.A1, DEMO.A2, DEMO.N)])[0] for x in P)
    info.append(f"probes {P}: per-probe mu_min {ml:.4f}; one weight vector for all: mu_min {mg:.4f} (certified >= {lbg})")
    ok &= ml < 1e-9 and lbg > 0
check("A9 demo stage 2: per-probe attributions are exact, yet no single prompt-independent mixture explains "
      "the probes jointly (certified)", ok)
for s in info: print("     " + s)

# ================================================================== G: geometry
print("=" * 72); print("G  geometry of the simplex"); print("=" * 72)
rng = np.random.default_rng(3); okg = True
for _ in range(50):
    p = rng.dirichlet(np.ones(5)); v = rng.normal(size=5); v -= v.mean()
    fisher = np.sum(v * v / p); round_ = np.sum((v / (2 * np.sqrt(p))) ** 2)
    okg &= abs(fisher - 4 * round_) < 1e-9 * fisher
check("G1 Fisher metric = 4 x round metric pulled back by p -> sqrt(p)", okg)
okd = True
for _ in range(10):
    p, q = rng.dirichlet(np.ones(4)), rng.dirichlet(np.ones(4))
    u0, u1 = np.sqrt(p), np.sqrt(q); th = math.acos(min(1, float(u0 @ u1)))
    w = (u1 - math.cos(th) * u0) / math.sin(th)
    ts = np.linspace(0, th, 20001); length = 0.0
    prev = None
    for t in ts:
        pt = (math.cos(t) * u0 + math.sin(t) * w) ** 2
        if prev is not None:
            dp = pt - prev; mid = (pt + prev) / 2; length += math.sqrt(np.sum(dp * dp / mid))
        prev = pt
    okd &= abs(length - 2 * th) < 1e-6
check("G2 Fisher-Rao distance = 2 arccos(sum sqrt(p q)) (length of the image of a great circle)", okd)
okl = oknd = True
for _ in range(50):
    z = rng.normal(size=2) + 1j * rng.normal(size=2); s = 1 + np.vdot(z, z).real
    H = (s * np.eye(2) - np.outer(z.conj(), z)) / s ** 2       # H[j,k] = ((1+|z|^2) delta - conj(z_j) z_k)/(1+|z|^2)^2
    omega = lambda v, w: np.imag(v @ H @ w.conj())
    tang = [1j * z[0] * np.array([1, 0]), 1j * z[1] * np.array([0, 1])]   # phase rotations: tangent to the fibre
    okl &= all(abs(omega(a, b)) < 1e-12 for a in tang for b in tang)
    oknd &= abs(omega(np.array([1, 0]), np.array([1j, 0]))) > 1e-6
    mu = np.r_[1, abs(z[0]) ** 2, abs(z[1]) ** 2] / s
    okl &= abs(mu.sum() - 1) < 1e-12
check("G3 CP^2: moment map lands in the simplex; its torus fibres (phases) are Lagrangian", okl and oknd)

print("=" * 72)
print(f"{sum(o for _, o in RESULTS)}/{len(RESULTS)} checks pass")
