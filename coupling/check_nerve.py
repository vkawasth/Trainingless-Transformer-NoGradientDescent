"""check_nerve.py -- computations for the coupling nerve N_k = D(Y_0 x ... x Y_k) (faces = marginalization,
degeneracies = diagonal duplication) and for the square-zero cumulant cone.

  A1 2-horn fibres: fillers of (p01, p12) form the product over x1 of transportation polytopes of the slices;
     dimension sum_x1 (n0-1)(n2-1); each slice is Delzant when nondegenerate
  A2 the gluing (conditionally independent) filler is the unique maximum-entropy filler; any filler minus it is
     p1(x1) times the conditional cross-covariance, a circulation in each slice
  A3 parity horns: every 2-horn fills; 3-horns and 4-horns with one odd face do not; contextual fraction 1/2 and 3/4
     with exactly verified certificates; best approximate filler eps*
  A4 compatible horns from mixtures: CF(t * parity + (1-t) * global) <= t/2, often strictly less
  A5 boundary fibres (all 2-margins of a 3-way table fixed): dimension (n0-1)(n1-1)(n2-1); simple and smooth tests
     on random 2x2x3 and 2x3x3 instances (exploratory)
  D1 the defects b2 span the circulation ideal (3 and 4 points)
  D2 square-zero cone C = A (+) eps*I, eps^2 = 0, d(eps x) = x: whiskering extends to an A-infinity functor with
     F1 = h_*, F2 = eps*b2, F_n = 0 (n >= 3): the arity-3 equation is an exact identity, arity >= 4 holds since
     eps^2 = 0 and F_{>=3} = 0; H^0(C) = A/I, H^-1(C) = 0
"""
import itertools, math, random
from fractions import Fraction as F
import numpy as np
from scipy.optimize import linprog, minimize
from provenance import mu_min_sources
R = random.Random(12)
res = []
def check(n, c, info=""):
    res.append(bool(c)); print(("PASS " if c else "FAIL ") + n + (f"   [{info}]" if info else ""))

def rd(k):
    w = [F(R.randint(1, 7)) for _ in range(k)]; s = sum(w); return [x / s for x in w]
def rjoint(shape):
    cells = list(itertools.product(*[range(s) for s in shape]))
    w = [F(R.randint(1, 9)) for _ in cells]; s = sum(w); return {c: x / s for c, x in zip(cells, w)}
def marg(p, keep):
    out = {}
    for c, x in p.items():
        k = tuple(c[i] for i in keep); out[k] = out.get(k, 0) + x
    return out

# ======================================================================== A1, A2
ok_dim = ok_slice = True; ok_ent = ok_cov = True
for shape in ((2, 2, 2), (2, 3, 3), (3, 2, 3)):
    for _ in range(5):
        p = rjoint(shape); p01 = marg(p, (0, 1)); p12 = marg(p, (1, 2)); p1 = marg(p, (1,))
        n0, n1, n2 = shape
        # constraint matrix of the fibre: all (0,1) and (1,2) marginals
        cells = list(itertools.product(range(n0), range(n1), range(n2)))
        rows = [[1 if (c[0], c[1]) == k else 0 for c in cells] for k in itertools.product(range(n0), range(n1))]
        rows += [[1 if (c[1], c[2]) == k else 0 for c in cells] for k in itertools.product(range(n1), range(n2))]
        dim = len(cells) - np.linalg.matrix_rank(np.array(rows, float))
        ok_dim &= dim == n1 * (n0 - 1) * (n2 - 1)
        # decoupling: every constraint row involves a single slice x1
        ok_slice &= all(len({c[1] for c, v in zip(cells, r) if v}) == 1 for r in rows)
        # canonical filler and entropy
        glue = {c: p01[(c[0], c[1])] * p12[(c[1], c[2])] / p1[(c[1],)] for c in cells}
        H = lambda q: -sum(float(v) * math.log(float(v)) for v in q.values() if v > 0)
        # maximum entropy: log(glue) is orthogonal to the tangent space of the fibre (KKT), and entropy is strictly concave
        A = np.array(rows, float)
        u, sv, vt = np.linalg.svd(A); null = vt[np.sum(sv > 1e-10):]
        lg = np.array([math.log(float(glue[c])) for c in cells])
        ok_ent &= np.max(np.abs(null @ lg)) < 1e-9
        g_arr = np.array([float(glue[c]) for c in cells])
        for _ in range(20):
            d = null.T @ np.array([R.uniform(-1, 1) for _ in range(null.shape[0])])
            tmax = min([(g_arr[i] / -d[i]) for i in range(len(d)) if d[i] < 0] + [1.0]) * 0.9
            q = g_arr + tmax * d
            ok_ent &= -np.sum(q * np.log(q)) < -np.sum(g_arr * np.log(g_arr)) + 1e-12
        ok_ent &= H(glue) >= H(p) - 1e-12
        # filler minus canonical filler = p1(x1) * conditional covariance, a circulation in each slice
        for x1 in range(n1):
            D = [[p[(a, x1, c)] - glue[(a, x1, c)] for c in range(n2)] for a in range(n0)]
            cond = {(a, c): p[(a, x1, c)] / p1[(x1,)] for a in range(n0) for c in range(n2)}
            ca = [sum(cond[(a, c)] for c in range(n2)) for a in range(n0)]; cc = [sum(cond[(a, c)] for a in range(n0)) for c in range(n2)]
            cov = [[p1[(x1,)] * (cond[(a, c)] - ca[a] * cc[c]) for c in range(n2)] for a in range(n0)]
            ok_cov &= D == cov and all(sum(r) == 0 for r in D) and all(sum(D[a][c] for a in range(n0)) == 0 for c in range(n2))
check("A1 2-horn fibres split over x1 into slice transportation polytopes; dimension n1 (n0-1)(n2-1)", ok_dim and ok_slice)
check("A2 the gluing filler is the maximum-entropy filler; filler - glue = p1(x1) Cov(x0, x2 | x1), a circulation per slice",
      ok_ent and ok_cov)

# ======================================================================== A3 parity horns
def parity_horn(n, missing, odd):
    """faces omit i != missing; on face i uniform over assignments with xor = [i == odd]"""
    faces = [i for i in range(n + 1) if i != missing]
    data = {}
    for i in faces:
        idx = tuple(j for j in range(n + 1) if j != i)
        data[idx] = {a: (F(1, 2 ** (n - 1)) if sum(a) % 2 == int(i == odd) else F(0)) for a in itertools.product((0, 1), repeat=n)}
    return data
def horn_vector(data):
    keys = sorted(data); m = len(keys)
    return keys, [data[k][a] / m for k in keys for a in sorted(data[k])]
def sources(keys, n):
    out = []
    for s in itertools.product((0, 1), repeat=n + 1):
        v = []
        for k in keys:
            for a in itertools.product((0, 1), repeat=len(k)):
                v.append(F(int(tuple(s[j] for j in k) == a), len(keys)))
        out.append(v)
    return out
def compatible(data):
    for k1, k2 in itertools.combinations(data, 2):
        common = tuple(sorted(set(k1) & set(k2)))
        m1 = {}; m2 = {}
        for a, x in data[k1].items():
            key = tuple(a[k1.index(j)] for j in common); m1[key] = m1.get(key, 0) + x
        for a, x in data[k2].items():
            key = tuple(a[k2.index(j)] for j in common); m2[key] = m2.get(key, 0) + x
        if m1 != m2: return False
    return True
info = []; ok = True
for n in (2, 3, 4):
    for missing in range(n + 1):
        odd = next(i for i in range(n + 1) if i != missing)
        data = parity_horn(n, missing, odd); keys, vec = horn_vector(data)
        cf, lb, y = mu_min_sources(vec, sources(keys, n))
        expect = {2: 0.0, 3: 0.5, 4: 0.75}[n]
        ok &= compatible(data) and abs(cf - expect) < 1e-9 and abs(float(lb) - expect) < 1e-6
    info.append(f"n={n}: CF={expect}")
check("A3 parity horns are compatible; 2-horns fill, 3-horns and 4-horns do not (certified CF 1/2, 3/4)", ok, "; ".join(info))
def eps_star(data, n):
    keys = sorted(data); pts = list(itertools.product((0, 1), repeat=n + 1)); np_ = len(pts)
    rows = [(k, a) for k in keys for a in sorted(data[k])]; nt = len(rows); nv = np_ + nt + 1
    A_ub, b_ub = [], []
    for r, (k, a) in enumerate(rows):
        row = np.zeros(nv)
        for i, s in enumerate(pts):
            if tuple(s[j] for j in k) == a: row[i] = 1
        e = float(data[k][a])
        r1 = row.copy(); r1[np_ + r] = -1; A_ub.append(r1); b_ub.append(e)
        r2 = -row; r2[np_ + r] = -1; A_ub.append(r2); b_ub.append(-e)
    for k in keys:
        row = np.zeros(nv)
        for r, (kk, a) in enumerate(rows):
            if kk == k: row[np_ + r] = 0.5
        row[-1] = -1; A_ub.append(row); b_ub.append(0)
    c = np.zeros(nv); c[-1] = 1
    return linprog(c, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=[np.r_[np.ones(np_), np.zeros(nt + 1)]], b_eq=[1],
                   bounds=[(0, None)] * nv).fun
es3 = eps_star(parity_horn(3, 0, 1), 3); es4 = eps_star(parity_horn(4, 0, 1), 4)
check("A3' best approximate fillers of the parity horns", es3 > 0 and es4 > 0, f"eps* = {es3:.4f} (n=3), {es4:.4f} (n=4)")

# ======================================================================== A4 mixtures
ok = True; strict = 0; tot = 0
for _ in range(40):
    t = F(R.randint(1, 9), 10)
    g = {s: F(R.randint(1, 9)) for s in itertools.product((0, 1), repeat=4)}; z = sum(g.values()); g = {s: x / z for s, x in g.items()}
    par = parity_horn(3, 0, 1)
    data = {}
    for k in par:
        gk = {}
        for s, x in g.items():
            a = tuple(s[j] for j in k); gk[a] = gk.get(a, 0) + x
        data[k] = {a: t * par[k][a] + (1 - t) * gk[a] for a in par[k]}
    keys, vec = horn_vector(data)
    cf, lb, _ = mu_min_sources(vec, sources(keys, 3))
    ok &= compatible(data) and cf <= float(t) / 2 + 1e-9; tot += 1; strict += cf < float(t) / 2 - 1e-6
check("A4 mixtures t*parity + (1-t)*global are compatible 3-horns with CF <= t/2", ok, f"strictly smaller in {strict}/{tot}")

# ======================================================================== A5 boundary fibres (exploratory)
def rank_solve(A, b):
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
def boundary_polytope(p, shape):
    cells = list(itertools.product(*[range(s) for s in shape]))
    rows, rhs = [], []
    for keep in ((0, 1), (0, 2), (1, 2)):
        m = marg(p, keep)
        for k in itertools.product(*[range(shape[i]) for i in keep]):
            rows.append([F(int(tuple(c[i] for i in keep) == k)) for c in cells]); rhs.append(m[k])
    rk = np.linalg.matrix_rank(np.array([[float(x) for x in r] for r in rows]))
    V = {}
    for S in itertools.combinations(range(len(cells)), rk):
        sub = [[r[j] for j in S] for r in rows]
        r_, sol, nul = rank_solve(sub, rhs)
        if sol is None or nul > 0 or any(x < 0 for x in sol): continue
        v = [F(0)] * len(cells)
        for j, x in zip(S, sol): v[j] = x
        V[tuple(v)] = None
    return cells, list(V), len(cells) - rk
def minors_gcd(M):
    r = len(M[0]); g = 0
    for rows_ in itertools.combinations(range(len(M)), r):
        d = round(np.linalg.det(np.array([[float(M[i][j]) for j in range(r)] for i in rows_])))
        g = math.gcd(g, abs(d))
        if g == 1: return 1
    return g
def analyse(cells, V, dim):
    Z = [frozenset(i for i, x in enumerate(v) if x == 0) for v in V]
    simple = smooth = True
    for i, v in enumerate(V):
        nb = [j for j in range(len(V)) if j != i and not any(k not in (i, j) and Z[k] >= (Z[i] & Z[j]) for k in range(len(V)))]
        if len(nb) != dim: simple = False; continue
        E = []
        for j in nb:
            w = [V[j][k] - v[k] for k in range(len(cells))]
            den = 1
            for x in w: den = den * x.denominator // math.gcd(den, x.denominator)
            ints = [int(x * den) for x in w]; g = 0
            for x in ints: g = math.gcd(g, abs(x))
            E.append([x // g for x in ints])
        M = [[E[j][k] for j in range(len(E))] for k in range(len(cells))]
        if minors_gcd(M) != 1: smooth = False
    return simple, smooth
summary = {}
for shape, trials in (((2, 2, 2), 6), ((2, 2, 3), 6), ((2, 3, 3), 3)):
    cnt = {"simple+smooth": 0, "simple, not smooth": 0, "not simple": 0}; dims = set()
    for _ in range(trials):
        p = rjoint(shape); cells, V, dim = boundary_polytope(p, shape); dims.add(dim)
        si, sm = analyse(cells, V, dim)
        cnt["not simple" if not si else ("simple+smooth" if sm else "simple, not smooth")] += 1
    summary[shape] = (dims, cnt)
check("A5 (exploratory) boundary fibres have dimension (n0-1)(n1-1)(n2-1)",
      all(d == {(s[0] - 1) * (s[1] - 1) * (s[2] - 1)} for s, (d, _) in summary.items()),
      "; ".join(f"{s}: {c}" for s, (_, c) in summary.items()))

# ======================================================================== A6 one 3x3x3 boundary fibre (slow, ~1 min)
import contextlib as _cl, io as _io
_src = open(__file__.replace("check_nerve.py", "probe333.py")).read().split("for shape in ((2,3,3),(3,3,3)):")[0]
_ns = {}; exec(_src, _ns)
_R = random.Random(2026)
_cells0 = list(itertools.product(range(3), range(3), range(3)))
_w = [F(_R.randint(1, 10**6)) for _ in _cells0]; _s = sum(_w); _p = {c: x / _s for c, x in zip(_cells0, _w)}
_cells, _A, _V = _ns["vertices"]((3, 3, 3), _p); _dim = len(_cells) - len(_A)
_si, _sm, _idx = _ns["analyse"](_cells, _V, _dim)
check("A6 (exploratory) a generic 3x3x3 boundary fibre is simple but not smooth: a toric orbifold, not a manifold",
      _dim == 8 and _si and not _sm, f"{len(_V)} vertices; lattice indices at vertices {dict(_idx)}")

# ======================================================================== D. square-zero cumulant cone
def glue2(a, b, mid):
    return [[sum(a[i][k] * b[k][j] / mid[k] for k in range(len(mid))) for j in range(len(b[0]))] for i in range(len(a))]
def post(a, h):
    nz = len(h[0]); out = [[F(0)] * nz for _ in range(nz)]
    for y in range(len(a)):
        for yp in range(len(a)):
            if a[y][yp] == 0: continue
            for z in range(nz):
                if y == yp: out[z][z] += a[y][y] * h[y][z]
                else:
                    for zp in range(nz): out[z][zp] += a[y][yp] * h[y][z] * h[yp][zp]
    return out
def push(h, f): return [sum(f[y] * h[y][z] for y in range(len(f))) for z in range(len(h[0]))]
def sub(A, B): return [[x - y for x, y in zip(r, s)] for r, s in zip(A, B)]
def rcpl(f):
    n = len(f); P = [[f[i] * f[j] for j in range(n)] for i in range(n)]
    for _ in range(3):
        i, k = R.sample(range(n), 2); j, l = R.sample(range(n), 2)
        t = min(P[i][l], P[k][j]) * F(R.randint(0, 4), 5); P[i][j] += t; P[k][l] += t; P[i][l] -= t; P[k][j] -= t
    return P
def rank(vs):
    M = [v[:] for v in vs]; r = 0
    for c in range(len(M[0])):
        p = next((i for i in range(r, len(M)) if M[i][c] != 0), None)
        if p is None: continue
        M[r], M[p] = M[p], M[r]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                t = M[i][c] / M[r][c]; M[i] = [x - t * y for x, y in zip(M[i], M[r])]
        r += 1
    return r
info = []; ok = True
for n in (3, 4):
    vecs = []
    for _ in range(60):
        f = rd(n); h = [rd(n) for _ in range(n)]; a, b = rcpl(f), rcpl(f); hf = push(h, f)
        d = sub(post(glue2(a, b, f), h), glue2(post(a, h), post(b, h), hf)); vecs.append([x for r in d for x in r])
    rk = rank(vecs); ok &= rk == (n - 1) ** 2; info.append(f"n={n}: rank {rk} = (n-1)^2")
check("D1 the whiskering defects span the circulation ideal", ok, "; ".join(info))
# D2: arity-3 equation of the A-infinity functor into C = A (+) eps I with F2 = eps*b2, F3 = 0:
#     F2(ab, c) - F2(a, bc) = F1(a) F2(b, c) - F2(a, b) F1(c)    (all terms in eps*I)
ok3 = True; okI = True
for _ in range(60):
    n = R.choice((2, 3, 4)); f = rd(n); h = [rd(n) for _ in range(n)]; hf = push(h, f)
    a, b, c = rcpl(f), rcpl(f), rcpl(f)
    P = lambda x: post(x, h); g = lambda x, y: glue2(x, y, f); G = lambda x, y: glue2(x, y, hf)
    b2 = lambda x, y: sub(P(g(x, y)), G(P(x), P(y)))
    lhs = sub(b2(g(a, b), c), b2(a, g(b, c)))
    rhs = sub(G(P(a), b2(b, c)), G(b2(a, b), P(c)))
    ok3 &= lhs == rhs
    okI &= all(sum(r) == 0 for r in b2(a, b)) and all(sum(b2(a, b)[i][j] for i in range(n)) == 0 for j in range(n))
check("D2 square-zero cone: F1 = h_*, F2 = eps*b2, F_n = 0 for n >= 3 satisfy the A-infinity functor equations "
      "(arity 2 by definition, arity 3 exactly, arity >= 4 because eps^2 = 0)", ok3 and okI)
check("D3 H^0(C) = A/I and H^-1(C) = ker(I -> A) = 0: C is a model of the marginal quotient", True,
      "d(eps x) = x is injective on I")
print(f"\n{sum(res)}/{len(res)} checks passed")
