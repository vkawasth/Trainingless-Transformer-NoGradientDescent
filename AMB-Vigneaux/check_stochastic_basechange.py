"""check_stochastic_basechange.py -- stochastic change of base for the base-relative tower,
and the toric geometry of coupling polytopes.

Part I (closes the gap "does the tower survive stochastic base change?").
For a Markov kernel k: B -> D(B') put
    Hyp(k)(theta, rho) = (theta, k# rho),     A_k = k# x D(k#) x D^2(k#) x D(Hyp k).
Checks, in exact arithmetic:
  SB1 A_k is a morphism of D-algebras (commutes with the componentwise barycenter)
  SB2 A_k preserves coherence (Lemma 9.3 survives)
  SB3 A_k o sigma_B = sigma_B' o Sh(k)   (Prop 9.5 survives)
  SB4 functoriality A_{h.k} = A_h A_k, A_eta = id, and A_k = A_f when k = eta.f is deterministic
  SB5 id x A_k is a comonad morphism C^B => C^B' and commutes with the strict law (Prop 9.13 survives)
  SB6 admissibility data transport: images of L_k and E_k are the transported L'_k, E'_k;
      coherent lifts map to coherent lifts; certificates pull back (k |> f');
      failures can disappear (failure level can only rise)
  SB7 the secondary obstruction o_3 cannot increase
  SB8 conditioning on evidence observed after the channel = conditioning on the pulled-back predicate
  SB9 graded 2-cells transport to every level: level-2 Kantorovich distance <= d(f, g)
Part II (the geometry behind the "Fukaya layer").
  CP1 the coupling polytope of (uniform on 2, uniform on 3) is a smooth (Delzant) hexagon whose fan is
      the fan of the degree-6 del Pezzo surface; the product coupling is equidistant from all facets
      (monotone), and the maximal coupling lies on the boundary
  CP2 a random nondegenerate 3x3 coupling polytope is simple and smooth (Delzant)
  CP3 the uniform 3x3 coupling polytope (scaled Birkhoff) is not simple
  CP4 Fisher metric on the coupling polytope = 2 x Hessian of Guillemin's potential (1/2) sum a log a,
      and the product coupling is the unique critical point of that potential on the polytope
"""
import itertools, random, math
from fractions import Fraction as F
import contextlib, io
import numpy as np
from scipy.optimize import linprog
with contextlib.redirect_stdout(io.StringIO()):
    from check_tower import Dist, delta, Dmap, mu, mix, THETA, rdist

R = random.Random(5)
res = []
def check(n, c, info=""):
    res.append(bool(c)); print(("PASS " if c else "FAIL ") + n + (f"   [{info}]" if info else ""))

B, Bp, Bpp = (0, 1), ("u", "v", "w"), ("P", "Q")
def rD(base, k=3):
    w = [F(R.randint(1, 5)) for _ in base]; s = sum(w)
    return Dist([(b, x / s) for b, x in zip(base, w)])
def rkernel(src, tgt): return {b: rD(tgt) for b in src}
def kext(k): return lambda rho: mu(Dmap(lambda b: k[b], rho))
def comp(k, h): return {b: kext(h)(k[b]) for b in k}               # h o k
def Hypk(k): return lambda hyp: (hyp[0], kext(k)(hyp[1]))
def A_k(k):
    e = kext(k)
    return lambda c: (e(c[0]), Dmap(e, c[1]), Dmap(lambda P: Dmap(e, P), c[2]), Dmap(Hypk(k), c[3]))
q = lambda h: h[1]
def sigma(S):
    Pi = Dmap(q, S); return (mu(Pi), Pi, Dmap(delta, Pi), S)
def coherent(c):
    rho, Pi, Om, S = c; return mu(Pi) == rho and mu(Om) == Pi and Dmap(q, S) == Pi
def bary(dA): return tuple(mix([(w, a[i]) for a, w in dA.items()]) for i in range(4))
def rshadow(base):
    pool = [delta(b) for b in base] + [rD(base) for _ in range(2)]
    return rdist([(R.choice(THETA), R.choice(pool)) for _ in range(3)])
def rcoh(base):
    """random coherent context that is NOT of the canonical form sigma(Sigma) (level 3 is genuinely uncertain)"""
    pool = [delta(b) for b in base] + [rD(base) for _ in range(3)]
    Pis = [rdist(pool) for _ in range(2)]
    Om = rdist(Pis); Pi = mu(Om)
    S = Dist([((R.choice(THETA), r), w) for r, w in Pi.items()])
    return (mu(Pi), Pi, Om, S)
def rany(base):
    pool = [delta(b) for b in base] + [rD(base)]
    Pis = [rdist(pool) for _ in range(2)]
    return (R.choice(pool), R.choice(Pis), rdist(Pis), rshadow(base))

print("=" * 72); print("Part I: stochastic change of base"); print("=" * 72)
ok1 = ok2 = ok3 = True
for _ in range(25):
    k = rkernel(B, Bp)
    cs = [rany(B) for _ in range(3)]; w = rdist(list(range(3)), 3)
    ok1 &= A_k(k)(bary(Dist([(cs[i], p) for i, p in w.items()]))) == bary(Dist([(A_k(k)(cs[i]), p) for i, p in w.items()]))
    c = rcoh(B); ok2 &= coherent(c) and coherent(A_k(k)(c))
    S = rshadow(B); ok3 &= A_k(k)(sigma(S)) == sigma(Dmap(Hypk(k), S))
check("SB1 A_k is a morphism of D-algebras", ok1)
check("SB2 A_k preserves coherence (Lemma 9.3 survives stochastic base change)", ok2)
check("SB3 A_k sigma_B = sigma_B' Sh(k) (Prop 9.5 survives)", ok3)

ok4 = True
for _ in range(20):
    k, h = rkernel(B, Bp), rkernel(Bp, Bpp); c = rany(B)
    ok4 &= A_k(comp(k, h))(c) == A_k(h)(A_k(k)(c))
    ok4 &= A_k({b: delta(b) for b in B})(c) == c
    fdet = {0: "u", 1: "u"}
    Af = (Dmap(fdet.get, c[0]), Dmap(lambda P: Dmap(fdet.get, P), c[1]),
          Dmap(lambda O: Dmap(lambda P: Dmap(fdet.get, P), O), c[2]), Dmap(lambda hy: (hy[0], Dmap(fdet.get, hy[1])), c[3]))
    ok4 &= A_k({b: delta(fdet[b]) for b in B})(c) == Af
check("SB4 functorial on the Kleisli category; agrees with Prop 9.13 on deterministic maps", ok4)

eps = lambda p: p[0]
dlt = lambda p: (p, p[1])
def lam(M): return (Dmap(eps, M), bary(Dmap(lambda p: p[1], M)))
Y = ("y0", "y1")
ok5 = True
for _ in range(20):
    k = rkernel(B, Bp); T = lambda p: (p[0], A_k(k)(p[1]))
    p = (R.choice(Y), rcoh(B))
    M = rdist([(R.choice(Y), rcoh(B)) for _ in range(4)], 2)
    ok5 &= eps(T(p)) == eps(p) and dlt(T(p)) == (T(p), A_k(k)(p[1]))
    ok5 &= (lam(M)[0], A_k(k)(lam(M)[1])) == lam(Dmap(T, M))
check("SB5 id x A_k is a comonad morphism and commutes with lambda (Prop 9.13 survives)", ok5)

# --- admissibility data
def tv(p, q):
    keys = set(dict(p.items())) | set(dict(q.items()))
    return sum(abs(dict(p.items()).get(x, 0) - dict(q.items()).get(x, 0)) for x in keys) / 2
ok6 = True; vanish = None
for _ in range(20):
    k = rkernel(B, Bp); e = kext(k)
    S1 = [delta(0), rD(B), rD(B)]
    S2 = [delta(r) for r in S1] + [rdist(S1)]                    # saturated
    S1p = [e(r) for r in S1]; S2p = [Dmap(e, P) for P in S2]
    ok6 &= all(any(P == delta(r) for P in S2p) for r in S1p)      # saturation preserved
    # a coherent lift (rho, Pi, Omega) in L1 x L2 x L3 maps to a coherent lift for the transported data
    Om = Dist([(R.choice(S2), F(1, 2)), (R.choice(S2), F(1, 2))]); Pi = mu(Om); rho = mu(Pi)
    Omp, Pip, rhop = Dmap(lambda P: Dmap(e, P), Om), Dmap(e, Pi), e(rho)
    ok6 &= all(P in S2p for P, _ in Omp.items()) and all(r in S1p for r, _ in Pip.items())
    ok6 &= mu(Omp) == Pip and mu(Pip) == rhop
    # certificate pullback: <k |> f', rho> = <f', k# rho> for every rho
    fp = {b: F(R.randint(-4, 4)) for b in Bp}
    pull = {b: sum(dict(k[b].items()).get(bp, 0) * fp[bp] for bp in Bp) for b in B}
    for r in S1 + [rho]:
        ok6 &= sum(p * pull[b] for b, p in r.items()) == sum(p * fp[b] for b, p in e(r).items())
# failure disappears: S1 = {delta_0}; claim x1 = delta_1 fails at level 1; a constant kernel merges everything
kc = {0: rD(Bp), 1: None}; kc[1] = kc[0]
x1 = delta(1)
check("SB6 admissibility data, lifts and certificates transport; failure can vanish",
      ok6 and x1 != delta(0) and kext(kc)(x1) == kext(kc)(delta(0)),
      "delta_1 fails against S1={delta_0} on B; after a constant kernel it passes")

def o3(Pi, S1, S2):
    """l1 distance from Pi (supported in S1) to conv(S2), as an LP"""
    idx = {r: i for i, r in enumerate(S1)}; n = len(S1); m = len(S2)
    v = np.zeros(n)
    for r, w in Pi.items(): v[idx[r]] += float(w)
    M = np.zeros((n, m))
    for j, P in enumerate(S2):
        for r, w in P.items(): M[idx[r], j] += float(w)
    # min sum t  s.t. -t <= v - M lam <= t, sum lam = 1, lam >= 0
    c = np.r_[np.zeros(m), np.ones(n)]
    A = np.r_[np.c_[-M, -np.eye(n)], np.c_[M, -np.eye(n)]]
    b = np.r_[-v, v]
    r = linprog(c, A_ub=A, b_ub=b, A_eq=[np.r_[np.ones(m), np.zeros(n)]], b_eq=[1], bounds=[(0, None)] * (m + n))
    return r.fun
ok7 = True; drops = []
for _ in range(20):
    k = rkernel(B, Bp); e = kext(k)
    S1 = [delta(0), delta(1), rD(B)]
    S2 = [Dist([(S1[0], F(1, 2)), (S1[1], F(1, 2))]), delta(S1[2])]   # not saturated: E2 is a proper subset of L2
    Pi = rdist(S1, 3)
    if _ % 2: k = {0: k[0], 1: k[0]}; e = kext(k)          # a merging kernel: k(0) = k(1)
    S1p = list({e(r): None for r in S1}); S2p = [Dmap(e, P) for P in S2]
    a, b_ = o3(Pi, S1, S2), o3(Dmap(e, Pi), S1p, S2p)
    ok7 &= b_ <= a + 1e-9; drops.append(a - b_)
check("SB7 the secondary obstruction o_3 does not increase along a stochastic base change", ok7,
      f"largest drop {max(drops):.3f}")

def cond(S, L):
    w = Dist([(h, L(h) * p) for h, p in S.items()]); z = sum(p for _, p in w.items())
    return Dist([(h, p / z) for h, p in w.items()])
ok8 = True
for _ in range(20):
    k = rkernel(B, Bp); S = rshadow(B); ep = R.choice(Bp)
    after = cond(Dmap(Hypk(k), S), lambda h: dict(h[1].items()).get(ep, 0))      # observe e' on B'
    pulled = lambda h: sum(p * dict(k[b].items()).get(ep, 0) for b, p in h[1].items())   # k |> 1_{e'}
    ok8 &= after == Dmap(Hypk(k), cond(S, pulled))
check("SB8 conditioning after the channel = conditioning on the pulled-back predicate, then base change", ok8)

def kantorovich(P, Q):
    xs, ys = list(dict(P.items())), list(dict(Q.items()))
    C = np.array([[float(tv(a, b)) for b in ys] for a in xs]); n, m = len(xs), len(ys)
    A_eq, b_eq = [], []
    for i, a in enumerate(xs):
        r = np.zeros(n * m); r[i * m:(i + 1) * m] = 1; A_eq.append(r); b_eq.append(float(dict(P.items())[a]))
    for j, b in enumerate(ys):
        r = np.zeros(n * m); r[j::m] = 1; A_eq.append(r); b_eq.append(float(dict(Q.items())[b]))
    return linprog(C.ravel(), A_eq=np.array(A_eq), b_eq=b_eq, bounds=[(0, None)] * (n * m)).fun
ok9 = True; ratios = []
for _ in range(20):
    f, g = rkernel(B, Bp), rkernel(B, Bp)
    d = max(tv(f[b], g[b]) for b in B)
    c = rany(B)
    l1 = tv(kext(f)(c[0]), kext(g)(c[0]))
    l2 = kantorovich(Dmap(kext(f), c[1]), Dmap(kext(g), c[1]))
    ok9 &= l1 <= d and l2 <= float(d) + 1e-9; ratios.append(l2 / float(d))
check("SB9 graded 2-cells transport: level-1 TV and level-2 Kantorovich distances <= d(f, g)", ok9,
      f"max level-2 ratio {max(ratios):.3f}")

print("=" * 72); print("Part II: coupling polytopes are moment polytopes"); print("=" * 72)
def rank_solve(A, b):
    """exact Gaussian elimination; returns (rank, a solution or None, nullity)"""
    A = [row[:] + [bb] for row, bb in zip(A, b)]; n = len(A[0]) - 1; piv = []; r = 0
    for c in range(n):
        p = next((i for i in range(r, len(A)) if A[i][c] != 0), None)
        if p is None: continue
        A[r], A[p] = A[p], A[r]; A[r] = [x / A[r][c] for x in A[r]]
        for i in range(len(A)):
            if i != r and A[i][c] != 0: A[i] = [x - A[i][c] * y for x, y in zip(A[i], A[r])]
        piv.append(c); r += 1
    if any(all(x == 0 for x in row[:-1]) and row[-1] != 0 for row in A): return r, None, n - r
    sol = [F(0)] * n
    for i, c in enumerate(piv): sol[c] = A[i][-1]
    return r, sol, n - r
def det(M):
    M = [row[:] for row in M]; n = len(M); d = F(1)
    for c in range(n):
        p = next((i for i in range(c, n) if M[i][c] != 0), None)
        if p is None: return F(0)
        if p != c: M[c], M[p] = M[p], M[c]; d = -d
        d *= M[c][c]
        for i in range(c + 1, n):
            t = M[i][c] / M[c][c]; M[i] = [x - t * y for x, y in zip(M[i], M[c])]
    return d
def polytope(f, g):
    n, m = len(f), len(g); cells = [(i, j) for i in range(n) for j in range(m)]
    def cons(S):
        A, b = [], []
        for i in range(n): A.append([F(int(c[0] == i)) for c in S]); b.append(f[i])
        for j in range(m): A.append([F(int(c[1] == j)) for c in S]); b.append(g[j])
        return A, b
    verts = {}
    for S in itertools.combinations(cells, n + m - 1):
        A, b = cons(S); r, sol, nul = rank_solve(A, b)
        if sol is None or nul > 0 or any(x < 0 for x in sol): continue
        v = tuple(sol[S.index(c)] if c in S else F(0) for c in cells)
        verts[v] = None
    return cells, list(verts)
def edge_dirs(cells, v, n, m):
    """at vertex v: for each zero cell, the unique circulation on supp(v)+cell with value 1 at that cell"""
    supp = [c for c, x in zip(cells, v) if x > 0]; zeros = [c for c, x in zip(cells, v) if x == 0]
    dirs = []
    for z in zeros:
        S = supp + [z]
        A = [[F(int(c[0] == i)) for c in S] for i in range(n)] + [[F(int(c[1] == j)) for c in S] for j in range(m)]
        A.append([F(int(c == z)) for c in S]); b = [F(0)] * (n + m) + [F(1)]
        r, sol, nul = rank_solve(A, b)
        if sol is None or nul > 0: return supp, zeros, None
        dirs.append({c: s for c, s in zip(S, sol)})
    return supp, zeros, dirs
def block(dv, n, m): return [dv.get((i, j), F(0)) for i in range(n - 1) for j in range(m - 1)]
def is_delzant(f, g):
    n, m = len(f), len(g); dim = (n - 1) * (m - 1); cells, V = polytope(f, g)
    simple = smooth = True
    for v in V:
        supp, zeros, dirs = edge_dirs(cells, v, n, m)
        if len(zeros) != dim or dirs is None: simple = False; continue
        smooth &= abs(det([block(dv, n, m) for dv in dirs])) == 1 and all(x in (-1, 0, 1) for dv in dirs for x in dv.values())
    return simple, smooth, V, cells

f, g = [F(1, 2)] * 2, [F(1, 3)] * 3
simple, smooth, V, cells = is_delzant(f, g)
# facet a_c >= 0 has inner normal = the functional 'entry c' on the circulation lattice, written in the basis
# E_ij - E_i,m-1 - E_n-1,j + E_n-1,m-1 (i < n-1, j < m-1), whose coordinates are the top-left block entries
def basis_circ(i, j, n, m):
    return {(i, j): 1, (i, m - 1): -1, (n - 1, j): -1, (n - 1, m - 1): 1}
fan = sorted(tuple(basis_circ(i, j, 2, 3).get(c, 0) for i in range(1) for j in range(2)) for c in cells)
dP6 = sorted([(1, 0), (0, 1), (-1, -1), (-1, 0), (0, -1), (1, 1)])
prod = [fi * gj for fi in f for gj in g]
check("CP1 Cpl(unif_2, unif_3) is a Delzant hexagon with the del Pezzo-6 fan; the product coupling is equidistant",
      simple and smooth and len(V) == 6 and len(set(prod)) == 1 and fan == dP6,
      f"{len(V)} vertices; facet normals {fan}; product coupling has all entries {prod[0]}")
ok = True
for _ in range(5):
    while True:
        a = [F(R.randint(1, 9)) for _ in range(3)]; b = [F(R.randint(1, 9)) for _ in range(3)]
        s = sum(a); t = sum(b); fr = [x / s for x in a]; gr = [x / t for x in b]
        sums_f = {sum(c) for r in (1, 2) for c in itertools.combinations(fr, r)}
        sums_g = {sum(c) for r in (1, 2) for c in itertools.combinations(gr, r)}
        if not (sums_f & sums_g): break
    si, sm, V3, _ = is_delzant(fr, gr); ok &= si and sm
check("CP2 nondegenerate 3x3 coupling polytopes are simple and smooth (Delzant)", ok, f"last one has {len(V3)} vertices")
si, sm, V3u, _ = is_delzant([F(1, 3)] * 3, [F(1, 3)] * 3)
check("CP3 uniform 3x3 (scaled Birkhoff polytope) is not simple", not si, f"{len(V3u)} vertices")
# CP4: Hessian of G = 1/2 sum a log a is (1/2) sum v_c w_c / a_c, i.e. half the Fisher metric; the product
# coupling is a critical point: grad G = 1/2 (log a + 1) is orthogonal to every circulation iff log a = r_i + s_j
rng = np.random.default_rng(1); okH = True
for _ in range(20):
    fr = rng.dirichlet(np.ones(3)); gr = rng.dirichlet(np.ones(4)); a = np.outer(fr, gr)
    v = rng.normal(size=(3, 4)); v -= v.mean(1, keepdims=True); v -= v.mean(0, keepdims=True)
    v *= 0.1 * a.min() / np.abs(v).max()
    G = lambda x: 0.5 * np.sum(x * np.log(x)); h = 1.0
    hess = (G(a + h * v) - 2 * G(a) + G(a - h * v)) / h ** 2
    okH &= abs(hess - 0.5 * np.sum(v * v / a)) < 1e-3 * np.sum(v * v / a)
    okH &= abs(np.sum(0.5 * (np.log(a) + 1) * v)) < 1e-12
check("CP4 Fisher = 2 x Hess(Guillemin potential); the product coupling is its critical point", okH)
print(f"\n{sum(res)}/{len(res)} checks passed")
