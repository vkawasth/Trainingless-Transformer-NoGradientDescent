"""check_polytope_companion.py -- verification script for
"Coupling Polytopes and Toric Geometry: the 2x3 Family, the Square Case, Higher Levels and Stability".

Exact rational arithmetic unless marked (LP) or (numerical).
  F  2x3 family: chambers, types, Delzant, surfaces, symmetry orbits, del Pezzo fan, wall-crossing
  K  the Kaehler class is the product coupling: edge lengths, area, base-point independence
  S  square 2x2 case: costs, whiskerings, defects
  L  level two: Delzant, flattening, transport along kernels, level three
  W  disc potentials: del Pezzo critical points, the 2x2 segment, the 3x3 example, Fano test
  T  optimal transport on a coupling polytope is attained at a vertex (spanning tree)
  C  cycle counts of K_{n,n}
"""
import itertools, math, random, sys, contextlib, io
from fractions import Fraction as F
from collections import Counter
import numpy as np
from scipy.optimize import linprog
R = random.Random(2)
res = []
def check(n, c, info=""):
    res.append(bool(c)); print(("PASS " if c else "FAIL ") + n + (f"   [{info}]" if info else ""))

# ----------------------------------------------------------------- general transportation polytopes
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
def tree_solution(S, f, g):
    A = [[F(int(c[0] == i)) for c in S] for i in range(len(f))] + [[F(int(c[1] == j)) for c in S] for j in range(len(g))]
    return rank_solve(A, list(f) + list(g))
def polytope(f, g):
    n, m = len(f), len(g); cells = [(i, j) for i in range(n) for j in range(m)]; V = {}
    for S in itertools.combinations(cells, n + m - 1):
        r, sol, nul = tree_solution(S, f, g)
        if sol is None or nul > 0 or any(x < 0 for x in sol): continue
        V[tuple(sol[S.index(c)] if c in S else F(0) for c in cells)] = None
    return cells, list(V)
def zeros(v, cells): return frozenset(c for c, x in zip(cells, v) if x == 0)
def neighbours(V, cells):
    Z = [zeros(v, cells) for v in V]; nb = {i: [] for i in range(len(V))}
    for i, j in itertools.combinations(range(len(V)), 2):
        common = Z[i] & Z[j]
        if not any(k not in (i, j) and Z[k] >= common for k in range(len(V))): nb[i].append(j); nb[j].append(i)
    return nb
def primitive(w):
    den = 1
    for x in w.values(): den = den * x.denominator // math.gcd(den, x.denominator)
    g = 0
    for x in w.values(): g = math.gcd(g, abs(int(x * den)))
    return {c: x * den / g for c, x in w.items()}
def delzant(f, g):
    n, m = len(f), len(g); dim = (n - 1) * (m - 1); cells, V = polytope(f, g)
    if len(V) == 1: return True, True, V, cells
    nb = neighbours(V, cells); simple = smooth = True
    for i, v in enumerate(V):
        if len(nb[i]) != dim: simple = False; continue
        dirs = [primitive({c: V[j][k] - v[k] for k, c in enumerate(cells)}) for j in nb[i]]
        smooth &= abs(det([[d[(a, b)] for a in range(n - 1) for b in range(m - 1)] for d in dirs])) == 1
    return simple, simple and smooth, V, cells
def facet_cells(V, cells):
    Zs = {c: frozenset(i for i, v in enumerate(V) if v[cells.index(c)] == 0) for c in cells}
    return [c for c in cells if Zs[c] and not any(d != c and Zs[d] > Zs[c] for d in cells)]
def nondegenerate(f, g):
    sf = {sum(c) for r in range(1, len(f)) for c in itertools.combinations(f, r)}
    sg = {sum(c) for r in range(1, len(g)) for c in itertools.combinations(g, r)}
    return not (sf & sg)

# ----------------------------------------------------------------- F. the 2x3 family (planar coordinates)
NORMAL = {(0, 0): (1, 0), (0, 1): (0, 1), (0, 2): (-1, -1), (1, 0): (-1, 0), (1, 1): (0, -1), (1, 2): (1, 1)}
CELLS = list(NORMAL)
def alpha(x, y, f1, g1, g2):
    return {(0, 0): x, (0, 1): y, (0, 2): f1 - x - y, (1, 0): g1 - x, (1, 1): g2 - y, (1, 2): x + y - (f1 + g1 + g2 - 1)}
def pverts(f1, g1, g2):
    V = set(); k0 = alpha(F(0), F(0), f1, g1, g2)
    for c1, c2 in itertools.combinations(CELLS, 2):
        n1, n2 = NORMAL[c1], NORMAL[c2]; d = n1[0] * n2[1] - n1[1] * n2[0]
        if d == 0: continue
        x = (-k0[c1] * n2[1] + k0[c2] * n1[1]) / d; y = (-n1[0] * k0[c2] + n2[0] * k0[c1]) / d
        if all(v >= 0 for v in alpha(x, y, f1, g1, g2).values()): V.add((x, y))
    return V
def pfacets(f1, g1, g2, V): return [c for c in CELLS if sum(1 for (x, y) in V if alpha(x, y, f1, g1, g2)[c] == 0) >= 2]
def order(fs): return sorted(fs, key=lambda c: math.atan2(NORMAL[c][1], NORMAL[c][0]))
def selfint(fs):
    o = order(fs); k = len(o); out = []
    for i, c in enumerate(o):
        p, q = NORMAL[o[i - 1]], NORMAL[o[(i + 1) % k]]; n = NORMAL[c]
        s = (p[0] + q[0], p[1] + q[1]); t = F(s[0], n[0]) if n[0] else F(s[1], n[1]); out.append(-int(t))
    return tuple(out)
def smooth2(fs):
    o = order(fs); k = len(o)
    return all(abs(NORMAL[o[i]][0] * NORMAL[o[(i + 1) % k]][1] - NORMAL[o[i]][1] * NORMAL[o[(i + 1) % k]][0]) == 1 for i in range(k))
def signs(f1, g1, g2):
    f = (f1, 1 - f1); g = (g1, g2, 1 - g1 - g2)
    return tuple((f[i] > g[j]) - (f[i] < g[j]) for i in range(2) for j in range(3))
def surface(si):
    k = len(si)
    return {3: "CP2", 5: "Bl2CP2", 6: "dP6"}.get(k) or ("CP1xCP1" if sorted(si) == [0, 0, 0, 0] else "F1")
chambers, reps = {}, {}
for _ in range(40000):
    p = tuple(F(R.randint(1, 9999), 10000) for _ in range(3))
    if p[1] + p[2] >= 1: continue
    s = signs(*p)
    if 0 in s: continue
    V = pverts(*p); fs = pfacets(*p, V)
    reps.setdefault(s, p); chambers.setdefault(s, set()).add((len(V), selfint(order(fs)), smooth2(fs)))
grid = set()
N = 36
for a in range(1, N):
    for b in range(1, N):
        for c in range(1, N - b):
            s = signs(F(a, N) + F(1, 7 * N), F(b, N) + F(1, 11 * N), F(c, N) + F(1, 13 * N))
            if 0 not in s: grid.add(s)
check("F1 the six walls f_i = g_j cut the marginal prism into 18 chambers", len(chambers) == 18 and grid == set(chambers))
ok = all(len(t) == 1 for t in chambers.values())
summ = Counter(surface(next(iter(t))[1]) for t in chambers.values())
check("F2 type constant on chambers; all Delzant; surfaces CP2 x2, CP1xCP1 x3, F1 x6, Bl2CP2 x6, dP6 x1",
      ok and all(next(iter(t))[2] for t in chambers.values()) and summ == Counter({"CP2": 2, "CP1xCP1": 3, "F1": 6, "Bl2CP2": 6, "dP6": 1}),
      str(dict(summ)))
check("F3 self-intersections sum to 12 - 3k in every chamber",
      all(sum(next(iter(t))[1]) == 12 - 3 * len(next(iter(t))[1]) for t in chambers.values()))
# symmetry: swap rows (f1 -> 1 - f1) and permute columns of g
def act(s, rowswap, perm):
    M = [[s[3 * i + j] for j in range(3)] for i in range(2)]
    if rowswap: M = [M[1], M[0]]
    M = [[row[perm[j]] for j in range(3)] for row in M]
    return tuple(M[i][j] for i in range(2) for j in range(3))
orbits = set()
for s in chambers:
    orbits.add(min(act(s, r, p) for r in (0, 1) for p in itertools.permutations(range(3))))
osz = Counter()
for o in orbits:
    members = {act(o, r, p) for r in (0, 1) for p in itertools.permutations(range(3))} & set(chambers)
    osz[(surface(next(iter(chambers[o]))[1]), len(members))] += 1
check("F4 the symmetry group S2 x S3 permutes the 18 chambers in 5 orbits, one per surface", len(orbits) == 5, str(dict(osz)))
dP6 = order(CELLS)
check("F5 every facet normal is a ray of the del Pezzo-6 fan; each ray is the sum of its two neighbours",
      all((NORMAL[dP6[i - 1]][0] + NORMAL[dP6[(i + 1) % 6]][0], NORMAL[dP6[i - 1]][1] + NORMAL[dP6[(i + 1) % 6]][1]) == NORMAL[dP6[i]] for i in range(6)))
trans = Counter()
for s, t in itertools.combinations(reps, 2):
    if sum(a != b for a, b in zip(s, t)) == 1:
        trans[abs(next(iter(chambers[s]))[0] - next(iter(chambers[t]))[0])] += 1
check("F6 crossing any single wall changes the number of vertices by exactly one", set(trans) == {1}, f"{sum(trans.values())} crossings")
ex = (F(1, 10), F(1, 2), F(1, 5))
check("F7 the point (0.1, 0.5, 0.2) gives a triangle (CP2)", len(pverts(*ex)) == 3)

# ----------------------------------------------------------------- K. Kaehler class = product coupling
def edge_length(c, p, V):
    pts = sorted(v for v in V if alpha(v[0], v[1], *p)[c] == 0); (x0, y0), (x1, y1) = pts[0], pts[-1]
    n = NORMAL[c]; a, b = -n[1], n[0]
    return abs((x1 - x0) / a) if a else abs((y1 - y0) / b)
def area(V):
    cx = sum(v[0] for v in V) / len(V); cy = sum(v[1] for v in V) / len(V)
    P = sorted(V, key=lambda v: math.atan2(float(v[1] - cy), float(v[0] - cx)))
    return abs(sum(P[i][0] * P[(i + 1) % len(P)][1] - P[(i + 1) % len(P)][0] * P[i][1] for i in range(len(P)))) / 2
def imatrix(fs):
    o = order(fs); k = len(o); si = selfint(fs); M = {}
    for i, c in enumerate(o):
        for j, d in enumerate(o):
            M[(c, d)] = F(si[i]) if i == j else (F(1) if j in ((i - 1) % k, (i + 1) % k) else F(0))
    return M
ok1 = ok2 = ok3 = True
for s, p in reps.items():
    V = pverts(*p); fs = pfacets(*p, V); M = imatrix(fs)
    f = (p[0], 1 - p[0]); g = (p[1], p[2], 1 - p[1] - p[2])
    lam = {c: f[c[0]] * g[c[1]] for c in fs}
    omD = {c: sum(lam[d] * M[(d, c)] for d in fs) for c in fs}
    ok1 &= all(omD[c] == edge_length(c, p, V) > 0 for c in fs)
    ok2 &= sum(lam[c] * omD[c] for c in fs) / 2 == area(V)
    bx = sum(v[0] for v in V) / len(V); by = sum(v[1] for v in V) / len(V)
    lam2 = alpha(bx, by, *p)
    ok3 &= all(sum(lam2[d] * M[(d, c)] for d in fs) == omD[c] for c in fs)
check("K1 [omega] = sum (f x g)_c D_c has omega.D_c = edge length > 0 (ample) in all 18 chambers", ok1)
check("K2 omega^2 / 2 = lattice area of the coupling polygon", ok2)
check("K3 the class does not depend on the base point", ok3)

# ----------------------------------------------------------------- S. square 2x2
def k2(a, b): return [[1 - a, a], [1 - b, b]]
DET = {"e0": k2(F(0), F(0)), "e1": k2(F(1), F(1)), "sigma": k2(F(0), F(1)), "tau": k2(F(1), F(0))}
def det_cost(f, g): return max(int(f[x] != g[x]) for x in range(2))
check("S1 costs between deterministic 1-cells: 0 on the diagonal and 1 for every other pair",
      all(det_cost(DET[a], DET[b]) == (0 if a == b else 1) for a in DET for b in DET))
def glue(A, B):
    g = [A[0][j] + A[1][j] for j in range(2)]
    return [[sum(A[i][m] * B[m][j] / g[m] for m in range(2) if g[m]) for j in range(2)] for i in range(2)]
def post(M, h):
    out = [[F(0)] * 2 for _ in range(2)]
    for y in range(2):
        for yp in range(2):
            if M[y][yp] == 0: continue
            for z in range(2):
                if y == yp: out[z][z] += M[y][y] * h[y][z]
                else:
                    for zp in range(2): out[z][zp] += M[y][yp] * h[y][z] * h[yp][zp]
    return out
half = F(1, 2); D = [[half, 0], [0, half]]; SW = [[0, half], [half, 0]]
h = k2(F(9, 10), F(1, 10)); nu = k2(half, half)
def defect(A, B, h):
    L = glue(post(A, h), post(B, h)); Rr = post(glue(A, B), h); return [[L[i][j] - Rr[i][j] for j in range(2)] for i in range(2)]
check("S2 unital whiskering: h.Delta = Delta, and alpha = Delta gives zero defect", post(D, h) == D and defect(D, SW, h) == [[0, 0], [0, 0]])
d1, d2 = defect(SW, SW, h), defect(SW, SW, nu)
check("S3 alpha = beta = swap: defects 369/2500 (-1,1;1,-1) for h = k(9/10,1/10) and 1/4 (-1,1;1,-1) for constant h",
      d1 == [[-F(369, 2500), F(369, 2500)], [F(369, 2500), -F(369, 2500)]] and d2 == [[-F(1, 4), F(1, 4)], [F(1, 4), -F(1, 4)]])
ok = True
for _ in range(200):
    a = F(R.randint(0, 10), 10); b = F(R.randint(0, 10), 10); hh = k2(a, b)
    p = F(R.randint(1, 9), 10)
    A = [[F(R.randint(0, 5)) for _ in range(2)] for _ in range(2)]
    s = sum(map(sum, A)) or F(1); A = [[x / s for x in r] for r in A]
    gm = [A[0][j] + A[1][j] for j in range(2)]
    if 0 in gm: continue
    K = [[F(R.randint(0, 5)) for _ in range(2)] for _ in range(2)]
    K = [[x / (sum(r) or 1) if sum(r) else F(int(j == 0)) for j, x in enumerate(r)] for r in K]
    B = [[gm[i] * K[i][j] for j in range(2)] for i in range(2)]
    d = defect(A, B, hh)
    ok &= all(sum(r) == 0 for r in d) and all(d[0][j] + d[1][j] == 0 for j in range(2))
check("S4 every defect has zero row and column sums (a tangent vector of the coupling polytope)", ok)

# ----------------------------------------------------------------- L. level two and three
P = [F(40, 100), F(35, 100), F(25, 100)]; Q = [F(30, 100), F(43, 100), F(27, 100)]
si, de, V3, cells3 = delzant(P, Q); fac3 = facet_cells(V3, cells3)
check("L1 the 3x3 level-two example is nondegenerate and Delzant, with 18 vertices and 9 facets (one per cell)",
      nondegenerate(P, Q) and de and len(V3) == 18 and len(fac3) == 9)
def flat(Pi, Y):
    nb = len(Y[0])
    return [[sum(Pi[a][b] * Y[a][i] * Y[b][j] for a in range(len(Y)) for b in range(len(Y))) for j in range(nb)] for i in range(nb)]
pm = [(F(1), F(0)), (F(0), F(1))]
seg = [[[(1 - t) / 2, t / 2], [t / 2, (1 - t) / 2]] for t in (F(0), F(1, 3), F(1))]
check("L2 for Y = point masses the flattening is the identity", all(flat(S, pm) == S for S in seg))
rho = [(half, half)]
check("L3 flattening is not surjective: Y = {uniform}, image = {product}, diagonal coupling missed",
      flat([[F(1)]], rho) == [[F(1, 4)] * 2] * 2)
ok = True
for _ in range(50):
    r1 = (F(R.randint(1, 9), 10),); r1 = (r1[0], 1 - r1[0]); r2 = (F(R.randint(1, 9), 10),); r2 = (r2[0], 1 - r2[0])
    if r1 == r2: continue
    A0 = flat([[F(1, 2), 0], [0, F(1, 2)]], [r1, r2]); A1 = flat([[0, F(1, 2)], [F(1, 2), 0]], [r1, r2])
    ok &= A0 != A1
check("L4 n = 2 with distinct states: the flattening is injective on the segment", ok)
# non-injective: n = 3 states on |B| = 2; dimension 4 > 1
Y3 = [(F(1), F(0)), (F(1, 2), F(1, 2)), (F(0), F(1))]
U = [F(1, 3)] * 3
# two couplings of (U, U) with the same flattening: difference in the kernel
M = []
for i in range(2):
    for j in range(2):
        M.append([F(Y3[a][i] * Y3[b][j]) for a in range(3) for b in range(3)])
for a in range(3): M.append([F(int(x // 3 == a)) for x in range(9)])
for b in range(3): M.append([F(int(x % 3 == b)) for x in range(9)])
r, _, nul = rank_solve(M, [F(0)] * len(M))
check("L5 n = 3 states on two outcomes: the flattening has positive-dimensional fibres", nul > 0, f"fibre dimension {nul}")
# transport along a kernel commutes with flattening
ok = True
for _ in range(30):
    kk = [[F(R.randint(1, 5)) for _ in range(3)] for _ in range(2)]; kk = [[x / sum(r) for x in r] for r in kk]
    Ys = [tuple(F(R.randint(0, 4)) for _ in range(2)) for _ in range(2)]
    Ys = [tuple(x / (sum(y) or 1) for x in y) if sum(y) else (F(1), F(0)) for y in Ys]
    Pi = [[F(R.randint(1, 5)) for _ in range(2)] for _ in range(2)]; s = sum(map(sum, Pi)); Pi = [[x / s for x in r] for r in Pi]
    kext = lambda y: tuple(sum(y[b] * kk[b][c] for b in range(2)) for c in range(3))
    lhs = flat(Pi, [kext(y) for y in Ys])
    fl = flat(Pi, Ys)
    rhs = [[sum(fl[b][bp] * kk[b][c] * kk[bp][cp] for b in range(2) for bp in range(2)) for cp in range(3)] for c in range(3)]
    ok &= lhs == rhs
check("L6 pushing a level-two coupling along a kernel k commutes with flattening: mu*(k# x k#)_* = (k x k)# mu*", ok)
R3 = [F(2, 5), F(7, 20), F(1, 4)]; S3 = [F(3, 10), F(43, 100), F(27, 100)]
check("L7 level three uses the same theorem: a nondegenerate pair of third-order states gives a Delzant polytope",
      nondegenerate(R3, S3) and delzant(R3, S3)[1])

# ----------------------------------------------------------------- W. disc potentials
import sympy as sp
x, y = sp.symbols("x y")
Wd = x + y + 1 / x + 1 / y + x * y + 1 / (x * y)
eqs = [sp.numer(sp.together(x * sp.diff(Wd, x))), sp.numer(sp.together(y * sp.diff(Wd, y)))]
sols = [s for s in sp.solve(eqs, [x, y], dict=True) if s[x] != 0 and s[y] != 0]
vals = sorted(sp.nsimplify(sp.simplify(Wd.subs(s))) for s in sols)
unit = all(abs(abs(complex(s[x])) - 1) < 1e-12 and abs(abs(complex(s[y])) - 1) < 1e-12 for s in sols)
check("W1 the del Pezzo-6 potential has 6 critical points, all with |x| = |y| = 1 (over the product coupling), critical values 6, -2, -2, -2, -3, -3",
      len(sols) == 6 and unit and vals == [-3, -3, -2, -2, -2, 6])
pp = qq = F(1, 3); lo, hi = max(F(0), pp + qq - 1), min(pp, qq)
check("W2 2x2 segment: the special fibre is the midpoint, which differs from the product coupling in general",
      (lo + hi) / 2 == F(1, 6) and pp * qq == F(1, 9))
# Fano test for square 3x3: P_{-K} = {all facet functions >= -1} is the transportation polytope with margins (3,3,3), i.e. 3 x Birkhoff
s33, _, _, _ = delzant([F(3)] * 3, [F(3)] * 3)
nef = True
for v in V3:
    T = [c for c, xx in zip(cells3, v) if xx > 0]
    rr, sol, nul = tree_solution(T, [F(3)] * 3, [F(3)] * 3); nef &= all(xx >= 0 for xx in sol)
check("W3 nondegenerate 3x3: P_{-K} = 3 x Birkhoff is not simple, so X is not Fano; in the example -K is nef (weak Fano)",
      not s33 and nef and len(fac3) == 9)

# ----------------------------------------------------------------- T. optimal transport at a vertex (LP)
ok = True
for _ in range(20):
    while True:
        f = [F(R.randint(1, 9)) for _ in range(3)]; g = [F(R.randint(1, 9)) for _ in range(3)]
        f = [a / sum(f) for a in f]; g = [a / sum(g) for a in g]
        if nondegenerate(f, g): break
    cells, V = polytope(f, g)
    cost = {c: F(R.randint(0, 20)) for c in cells}
    best = min(sum(cost[c] * v[k] for k, c in enumerate(cells)) for v in V)
    A_eq = [[float(c[0] == i) for c in cells] for i in range(3)] + [[float(c[1] == j) for c in cells] for j in range(3)]
    lp = linprog([float(cost[c]) for c in cells], A_eq=A_eq, b_eq=[float(a) for a in f + g], bounds=[(0, None)] * 9)
    ok &= abs(lp.fun - float(best)) < 1e-9
check("T1 (LP) the optimal-transport value equals the minimum over vertices (spanning-tree couplings)", ok)

# ----------------------------------------------------------------- C. cycle counts
def count_cycles(n):
    out = set()
    for kk in range(2, n + 1):
        for rows in itertools.combinations(range(n), kk):
            for cols in itertools.combinations(range(n), kk):
                for rp in itertools.permutations(rows[1:]):
                    ro = (rows[0],) + rp
                    for perm in itertools.permutations(cols):
                        e = frozenset([(ro[i], perm[i]) for i in range(kk)] + [(ro[(i + 1) % kk], perm[i]) for i in range(kk)])
                        if len(e) == 2 * kk: out.add(e)
    return Counter(len(e) // 2 for e in out)
ok = all(dict(count_cycles(n)) == {kk: math.comb(n, kk) ** 2 * math.factorial(kk) * math.factorial(kk - 1) // 2 for kk in range(2, n + 1)} for n in (3, 4))
check("C1 the number of 2k-cycles of K_{n,n} is C(n,k)^2 k!(k-1)!/2 (n = 3, 4 by enumeration)", ok, str(dict(count_cycles(4))))
print(f"\n{sum(res)}/{len(res)} checks passed")
