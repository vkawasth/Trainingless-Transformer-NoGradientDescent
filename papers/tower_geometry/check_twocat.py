"""check_twocat.py -- checks for the sharpened open problem on a 2-category for the tower (Section 6.6) of
"Shadows, Fibres and Secants of the Posterior Tower". Uses the tower code (check_tower.py: C_3, delta, eta, mu, lambda-bar) unchanged.

  TC1  the (Cm) cell of lambda-bar, level one: the left side pairs every label with the pooled context e = sum_k a_k E3(eta)(c_k), the
      right side pairs it with the item's own context E3(eta)(c_k), and the label marginals of the two sides agree (200 random samples)
  TC2  the geometric step of the collapse: for random points u, v in the relative interior of a random polytope S there are a in (0,1)
      and w in S with v = a u + (1-a) w (linear programming, 300 cases)
  TC3  convex order is not a Barr lifting: delta_r <=cx delta_r' only for r = r', yet <=cx is not equality
  TC4  coupling liftings: Lemma 6.11(d) holds for Strassen's martingale lifting -- mixtures of martingale couplings are martingale
      couplings, so mu is monotone, and composing martingale couplings gives a martingale coupling (200 random instances each)
  TC5  martingale orientation of the (Cm) cell: it fails on the paper's counterexample, holds whenever every label has the same
      conditional mean context (a decorrelated example), and holds in only some random samples; the context marginals are always
      ordered, delta_e <=cx sum_k a_k delta_{E3(eta)(c_k)}
  TC6 the level-one output of delta(x, c) has context marginal equal to the point mass at c (200 random points), so under any order
      that compares level-one laws by the convex order of their context marginals, delta(x, c0) <= delta(x, c1) forces c0 = c1
  TC7 the context-marginal order breaks monotonicity of D(epsilon): two level-one laws with the same context marginal (hence
      equivalent) have label marginals delta_{delta_0} and delta_{delta_1}, which are incomparable
  TC8 the first step of delta-monotonicity for label-preserving liftings: if j(y) R j'(y) for every y, then D(j)(p) and D(j')(p) are
      related in the Barr lifting of R, for every p (linear programming, 300 random relations and maps)
  TC9  the coupling 2-category: vertical composition of couplings by conditional product is associative and unital, with the
       diagonal coupling as identity (300 random cases)
  TC10 whiskering is not functorial: pushing a glued coupling forward along a non-injective map differs from gluing the pushed-forward
       couplings in most random cases; every difference comes with a Beck-Chevalley defect c > 0 for the windows sigma(hX,hY),
       sigma(hY,hZ) of the pushed law, and when that law is built conditionally independent given hY the two agree (500 cases each)
  TC11 counit coherence of lambda-bar's comultiplication 2-cell: pushing the forced coupling (label, own context) -> (label, pooled
       context) forward along the label projection gives the diagonal coupling, i.e. the identity 2-cell (200 random samples)
Run: python check_twocat.py   (from the folder that contains check_tower.py)
"""
import io, os, contextlib, itertools
from fractions import Fraction as F
import numpy as np
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(HERE, "check_tower.py")
if not os.path.exists(src): src = os.path.join(HERE, "..", "check_tower.py")
ns = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(src).read(), ns)
lam_bar, dlt, C, Dmap, E, eta, Emix, rM, X, ctx, Dist, delta = (ns[k] for k in
    ["lam_bar", "dlt", "C", "Dmap", "E", "eta", "Emix", "rM", "X", "ctx", "Dist", "delta"])

rng = np.random.default_rng(20261013)
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""))

def sides(M):
    lhs = dlt(lam_bar(M)); rhs = C(lam_bar)(lam_bar(Dmap(dlt, M)))
    return lhs[1][0], rhs[1][0], lam_bar(M)[1]

# TC1
ok = True; n = 0
for _ in range(200):
    M = rM(X); L1, R1, e = sides(M)
    imgs = {E(eta)(p[1]) for p in M.m}
    labs = lambda D: Dmap(lambda t: t[0], D)
    ok &= all(c == e for _, c in L1.m) and all(c in imgs for _, c in R1.m)
    ok &= e == Emix([(a, E(eta)(p[1])) for p, a in M.items()]) and labs(L1) == labs(R1); n += 1
check("TC1 (Cm) cell: pooled context e on the left, own contexts E3(eta)(c_k) on the right, e their mixture, equal label marginals", ok,
      f"{n} random samples")

# TC2
ok = True; smallest = 1.0
for _ in range(300):
    d, k = rng.integers(2, 5), rng.integers(4, 8)
    V = rng.normal(size=(k, d))                                     # S = conv(V)
    u = rng.dirichlet(np.ones(k)) @ V; v = rng.dirichlet(np.ones(k)) @ V   # interior points (all weights positive)
    found = False
    for a in [2.0 ** -m for m in range(1, 41)]:                     # w = (v - a u)/(1 - a) must lie in S
        w = (v - a * u) / (1 - a)
        r = linprog(np.zeros(k), A_eq=np.vstack([V.T, np.ones(k)]), b_eq=np.append(w, 1), bounds=(0, None), method="highs")
        if r.status == 0: found = True; smallest = min(smallest, a); break
    ok &= found
check("TC2 any two relative-interior points u, v of a polytope S satisfy v = a u + (1-a) w with a in (0,1), w in S", ok,
      f"300 random cases; smallest a needed {smallest:.1e}")

# TC3
def mart(P, Q, pts_P, pts_Q):
    """exists a coupling of P (atoms pts_P) and Q (atoms pts_Q) with E[target | source] = source?"""
    n, m = len(pts_P), len(pts_Q); A, b = [], []
    for i in range(n):
        row = np.zeros(n * m); row[i * m:(i + 1) * m] = 1; A.append(row); b.append(P[i])
        for coord in range(pts_P.shape[1]):
            row = np.zeros(n * m); row[i * m:(i + 1) * m] = pts_Q[:, coord] - pts_P[i, coord]; A.append(row); b.append(0)
    for j in range(m):
        row = np.zeros(n * m); row[j::m] = 1; A.append(row); b.append(Q[j])
    return linprog(np.zeros(n * m), A_eq=np.array(A), b_eq=np.array(b), bounds=(0, None), method="highs").status == 0
pts = np.array([[0.2], [0.5], [0.8]])
point_masses = all(mart([1], [1], pts[[i]], pts[[j]]) == (i == j) for i in range(3) for j in range(3))
nontrivial = mart([1], [0.5, 0.5], np.array([[0.5]]), np.array([[0.2], [0.8]]))
check("TC3 convex order relates point masses only when equal, yet is not equality, so it is no Barr lifting", point_masses and nontrivial)

# TC4
def rand_mart(src_pts):
    """a random martingale kernel from each source point: spread it into two points with the same mean."""
    out = []
    for x in src_pts:
        dvec = rng.normal(size=x.shape) * 0.1; t = rng.uniform(0.2, 0.8)
        out.append([(x + dvec * (1 - t) / t, t), (x - dvec, 1 - t)])     # mean t(x + d(1-t)/t) + (1-t)(x - d) = x
    return out
ok_mix = ok_comp = True
for _ in range(200):
    k = rng.integers(2, 4); comps = []
    for _ in range(k):                                             # pairs p_i <=cx p'_i
        src_pts = rng.dirichlet(np.ones(3), size=2); w = rng.dirichlet(np.ones(2))
        ker = rand_mart(src_pts)
        tgt = [(y, wi * t) for wi, row in zip(w, ker) for y, t in row]
        comps.append((src_pts, w, np.array([y for y, _ in tgt]), np.array([t for _, t in tgt])))
    lam = rng.dirichlet(np.ones(k))
    P = np.concatenate([l * c[1] for l, c in zip(lam, comps)]); Pp = np.concatenate([c[0] for c in comps])
    Q = np.concatenate([l * c[3] for l, c in zip(lam, comps)]); Qp = np.concatenate([c[2] for c in comps])
    ok_mix &= mart(P, Q, Pp, Qp)                                   # mu(mixture) <=cx mu(mixture')
    src_pts = rng.dirichlet(np.ones(3), size=2); w = rng.dirichlet(np.ones(2))
    k1 = rand_mart(src_pts); mid = np.array([y for row in k1 for y, _ in row]); midw = np.array([wi * t for wi, row in zip(w, k1) for _, t in row])
    k2 = rand_mart(mid); end = np.array([y for row in k2 for y, _ in row]); endw = np.array([wi * t for wi, row in zip(midw, k2) for _, t in row])
    ok_comp &= mart(w, endw, src_pts, end)
check("TC4 martingale lifting: mixtures of martingale couplings are martingale couplings (mu monotone); compositions too (preorder)",
      ok_mix and ok_comp, "200 instances each")

# TC5
def mart_oriented(M):
    L1, R1, e = sides(M); ok = True
    for r in {lab for lab, _ in R1.m}:
        pairs = [(p, c) for (lab, c), p in R1.items() if lab == r]; tot = sum(p for p, _ in pairs)
        ok &= Emix([(p / tot, c) for p, c in pairs]) == e
    marg_ok = Emix([(p, c) for (lab, c), p in R1.items()]) == e      # context marginal of the right side has mean e
    return ok, marg_ok
c1, c2 = ctx(delta(0)), ctx(delta(1))
ce, _ = mart_oriented(Dist([((0, c1), F(1, 2)), ((0, c2), F(1, 2))]))
c3 = ctx(Dist([(0, F(1, 2)), (1, F(1, 2))])); c4 = (c3[0], c3[1], c3[2], delta(("t2", delta(0))))
dec, _ = mart_oriented(Dist([((0, c3), F(1, 2)), ((0, c4), F(1, 2))]))
rand = [mart_oriented(rM(X)) for _ in range(200)]
held = sum(r[0] for r in rand); marg = all(r[1] for r in rand)
check("TC5 martingale orientation fails on the counterexample, holds when labels and contexts are decorrelated, and only sometimes "
      "otherwise; the context marginals are always ordered", (not ce) and dec and 0 < held < 200 and marg,
      f"holds in {held} of 200 random samples")

# TC6
rpoint = ns["rpoint"]
ok = True
for _ in range(200):
    pt = rpoint(X); x, c = pt
    ok &= all(cc == c for (_, cc) in dlt(pt)[1][0].m)
check("TC6 level-one output of delta(x,c) has context marginal delta_c, so the context-marginal order forces c0 = c1", ok,
      "200 random points")

# TC7
c = ctx(delta(0))
L = Dist([((delta(0), c), F(1))]); Lp = Dist([((delta(1), c), F(1))])        # same context marginal, different labels
same_ctx = Dmap(lambda t: t[1], L) == Dmap(lambda t: t[1], Lp)
lab, labp = Dmap(lambda t: t[0], L), Dmap(lambda t: t[0], Lp)               # D(epsilon) images: point masses at delta_0, delta_1
pts = np.array([[1.0, 0.0], [0.0, 1.0]])                                    # delta_0, delta_1 as points of the simplex
incomparable = not mart([1], [1], pts[[0]], pts[[1]]) and not mart([1], [1], pts[[1]], pts[[0]])
check("TC7 the context-marginal order identifies laws whose D(epsilon) images are incomparable, so D(epsilon) is not monotone",
      same_ctx and lab != labp and incomparable)

# TC8
def barr(P, Q, R):
    pairs = [(i, j) for i in range(len(P)) for j in range(len(Q)) if R[i, j]]
    if not pairs: return False
    A = np.zeros((len(P) + len(Q), len(pairs)))
    for k, (i, j) in enumerate(pairs): A[i, k] = 1; A[len(P) + j, k] = 1
    return linprog(np.zeros(len(pairs)), A_eq=A, b_eq=np.concatenate([P, Q]), bounds=(0, None), method="highs").status == 0
ok = True
for _ in range(300):
    ny, nz = rng.integers(2, 6), rng.integers(2, 7)
    R = rng.random((nz, nz)) < 0.3; np.fill_diagonal(R, True)
    jmap = rng.integers(0, nz, ny); jp = np.array([rng.choice(np.flatnonzero(R[jmap[y]])) for y in range(ny)])  # j(y) R j'(y)
    p = rng.dirichlet(np.ones(ny))
    P = np.bincount(jmap, weights=p, minlength=nz); Q = np.bincount(jp, weights=p, minlength=nz)
    ok &= barr(P, Q, R)
check("TC8 pointwise-related maps give related pushforwards in the Barr lifting (first step of delta-monotonicity)", ok,
      "300 random relations and maps")

# TC9-TC11: the coupling 2-category
def glue(w1, w2):
    py = w1.sum(0); return np.einsum("xy,yz->xz", w1 / np.where(py > 0, py, 1)[None, :], w2)
ok = True
for _ in range(300):
    n = rng.integers(2, 5); g = [rng.dirichlet(np.ones(n)) for _ in range(4)]
    def coup(a, b):                                                # a random coupling of g[a] and g[b] (rows a, columns b)
        K = rng.dirichlet(np.ones(n), n); w = g[a][:, None] * K   # w has row marginal g[a]; fix the columns by IPF
        for _ in range(500):
            w *= (g[b] / w.sum(0))[None, :]; w *= (g[a] / w.sum(1))[:, None]
        return w
    w1, w2, w3 = coup(0, 1), coup(1, 2), coup(2, 3)
    ok &= np.abs(glue(glue(w1, w2), w3) - glue(w1, glue(w2, w3))).max() < 1e-9
    ident = np.diag(g[1]); ok &= np.abs(glue(w1, ident) - w1).max() < 1e-9 and np.abs(glue(ident, w2) - w2).max() < 1e-9
check("TC9 vertical composition by conditional product is associative and unital (diagonal coupling)", ok, "300 random cases")

def cdefect(Q):
    """Q[hx, hy, hz]: max over hy of the largest non-trivial singular value of the normalised (hx, hz) table given hy."""
    c = 0.0
    for b in range(Q.shape[1]):
        T = Q[:, b, :]; m = T.sum()
        if m <= 0: continue
        T = T / m; r, k = T.sum(1), T.sum(0); keep_r, keep_k = r > 0, k > 0
        K = T[np.ix_(keep_r, keep_k)] / np.sqrt(np.outer(r[keep_r], k[keep_k]))
        sv = np.linalg.svd(K, compute_uv=False)
        if len(sv) > 1: c = max(c, sv[1])
    return c
differ = 0; ok = True; ok_ci = True
for _ in range(500):
    n = 4; gy = rng.dirichlet(np.ones(n)); K1 = rng.dirichlet(np.ones(n), n); K2 = rng.dirichlet(np.ones(n), n)
    P = np.einsum("y,yx,yz->xyz", gy, K1, K2)                     # glued law: X _||_ Z | Y
    h = rng.integers(0, 2, n); h[0], h[1] = 0, 1                   # non-injective, onto {0, 1}
    H = np.eye(2)[h]
    Q = np.einsum("xyz,xa,yb,zc->abc", P, H, H, H)                 # law of (hX, hY, hZ)
    lhs = Q.sum(1)                                                 # h_*(w1 ; w2), the composite coupling of the endpoints
    rhs = glue(Q.sum(2), Q.sum(0))                                 # (h_* w1) ; (h_* w2), glued along the coarse middle
    d = np.abs(lhs - rhs).max()
    if d > 1e-9:
        differ += 1; ok &= cdefect(Q) > 1e-9
    gy2 = rng.dirichlet(np.ones(2)); Q2 = np.einsum("b,ba,bc->abc", gy2, rng.dirichlet(np.ones(2), 2), rng.dirichlet(np.ones(2), 2))
    ok_ci &= np.abs(Q2.sum(1) - glue(Q2.sum(2), Q2.sum(0))).max() < 1e-12 and cdefect(Q2) < 1e-9
check("TC10 whiskering is not functorial on vertical composites; every failure has c > 0, and c = 0 laws agree", ok and ok_ci and differ > 300,
      f"differ in {differ} of 500 random cases")

ok = True
for _ in range(200):
    M = rM(X); L1, R1, e = sides(M)
    forced = Dist([(((lab, c), (lab, e)), p) for (lab, c), p in R1.items()])          # (label, own context) -> (label, pooled)
    proj = Dmap(lambda pr: (pr[0][0], pr[1][0]), forced)                               # push both ends along the label projection
    ok &= all(a == b for a, b in proj.m) and Dmap(lambda pr: pr[1], forced) == L1 and Dmap(lambda pr: pr[0], forced) == R1
check("TC11 the forced comultiplication 2-cell of lambda-bar is a coupling of the two sides, and its label projection is the identity",
      ok, "200 random samples")

print(f"\n{sum(results)}/{len(results)} checks passed")
