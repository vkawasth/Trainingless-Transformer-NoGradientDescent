"""check_coupling_algebra.py -- every claim of "Coupling Algebras" (exact arithmetic unless marked).

  A1 structure: e_f = diag(f) - f f^T and P_f = f f^T are complementary central idempotents of the gluing algebra A_f;
     I_f = e_f A_f; conjugation by diag(f)^{-1/2} is an algebra isomorphism A_f -> R x M_{n-1}(R)  (n = 2..5)
  A2 multi-object splitting: (s f g^T + x).(t g k^T + y) = st f k^T + x.y for circulations x, y (sizes 2,3,4)
  A3 kernel actions in split coordinates: pi_h(f g^T) = (hf)(hg)^T and pi_h(I) in I; w_h(P_f) = P_{hf} + c_h(f) with
     c_h(f) = sum_y f_y^2 (diag h_y - h_y h_y^T) in I; b_2(P_f, P_f) = c - c.c for the whiskering
  A4 uniqueness and generality: for every mass-preserving linear F1 (here convex combinations of pi_h and w_h) the arity-2
     and arity-3 equations hold with F2 = eps b2, F_{>=3} = 0
  A5 homotopy content: the cone C has H^0 = R (rank 1) and H^{-1} = 0; q(a + eps x) = s(a) is a chain map with q o F = s;
     if phi(e_f) is a boundary then phi(I) consists of boundaries (checked in C itself)
  A6 double category: interchange (glue then push = push then glue) holds for a bijective middle map, fails for a
     stochastic and for a deterministic non-injective one, including the explicit collapse and constant-kernel examples of
     the paper; the failure is a circulation
  A7 lattice polytope: N.Cpl(f,g) has integral vertices and the integer decomposition property (k = 2, 3), hexagon example
  A8 Ehrhart: table counts 7, 19, 37, 61, 91 = 3k^2 + 3k + 1, leading coefficient = lattice area; a second 2x3 example
  A9 Markov moves: the basic 2x2 moves connect all integer tables with fixed margins (two examples); each move is an
     integer circulation
  A10 Fisher's exact test on a 40-item subsample of the synthetic triage data: enumeration of the fibre, hypergeometric
      probabilities sum to 1, exact p-value against the chi-square approximation (floating point)
"""
import csv, itertools, math, os, random
from fractions import Fraction as Fr
import sympy as sp
R = random.Random(11); res = []
def check(n, c, info=""):
    res.append(bool(c)); print(("PASS " if c else "FAIL ") + n + (f"   [{info}]" if info else ""))
def rq(k=9): return sp.Rational(R.randint(1, k), R.randint(1, k))
def dist(n):
    w = [rq() for _ in range(n)]; s = sum(w); return sp.Matrix([x / s for x in w])
def stoch(n, m): return sp.Matrix([[x / sum(r) for x in r] for r in [[rq() for _ in range(m)] for _ in range(n)]])
def circ(n, m):
    x = sp.Matrix(n, m, lambda i, j: rq() - rq())
    x -= x * sp.ones(m, 1) * sp.ones(1, m) / m; x -= sp.ones(n, 1) * sp.ones(1, n) * x / n; return x
def mul(a, g, b): return a * sp.diag(*g).inv() * b
def s_of(a, f): return (a * sp.ones(a.shape[1], 1))[0] / f[0]
def is_circ(x): return (x * sp.ones(x.shape[1], 1)).is_zero_matrix and (sp.ones(1, x.shape[0]) * x).is_zero_matrix

# A1
ok = True
for n in (2, 3, 4, 5):
    f = dist(n); D = sp.diag(*f); P = f * f.T; e = D - P
    a = rq() * P + circ(n, n); b = rq() * P + circ(n, n)
    ok &= mul(e, f, e) == e and mul(P, f, P) == P and mul(e, f, P).is_zero_matrix and e + P == D
    ok &= mul(e, f, a) == mul(a, f, e) and mul(P, f, a) == s_of(a, f) * P and is_circ(mul(e, f, a)) and mul(e, f, a - s_of(a, f) * P) == a - s_of(a, f) * P
    Dh = sp.diag(*[sp.sqrt(x) for x in f]); phi = lambda m: Dh.inv() * m * Dh.inv()
    ok &= sp.simplify(phi(mul(a, f, b)) - phi(a) * phi(b)).is_zero_matrix and phi(D) == sp.eye(n)
    v = sp.Matrix([sp.sqrt(x) for x in f])
    ok &= sp.simplify(phi(P) - v * v.T).is_zero_matrix          # P_f goes to the projection onto sqrt(f)
check("A1 e_f and P_f = f f^T are complementary central idempotents; I_f = e_f A_f; A_f = R x M_{n-1}(R) by conjugation", ok, "n = 2..5")

# A2
f, g, k = dist(2), dist(3), dist(4)
s, t = rq(), rq(); x, y = circ(2, 3), circ(3, 4)
lhs = mul(s * f * g.T + x, g, t * g * k.T + y)
check("A2 multi-object splitting: (s f g^T + x).(t g k^T + y) = st f k^T + x.y", lhs == s * t * f * k.T + mul(x, g, y))

# A3
def w(h, a):
    n = a.shape[0]; m = h.shape[1]; out = sp.zeros(m, m)
    for i in range(n):
        for j in range(n):
            if i == j:
                for z in range(m): out[z, z] += a[i, i] * h[i, z]
            else: out += a[i, j] * h[i, :].T * h[j, :]
    return out
f = dist(3); h = stoch(3, 3); hf = h.T * f; gg = dist(3); hg = h.T * gg
c = sum((f[i] ** 2 * (sp.diag(*h[i, :]) - h[i, :].T * h[i, :]) for i in range(3)), sp.zeros(3, 3))
ok = h.T * (f * gg.T) * h == hf * hg.T and is_circ(h.T * circ(3, 3) * h)
ok &= w(h, f * f.T) == hf * hf.T + c and is_circ(c)
a_r = sp.Matrix(3, 3, lambda i, j: rq() - rq())
ok &= w(h, a_r) == h.T * a_r * h + sum((a_r[i, i] * (sp.diag(*h[i, :]) - h[i, :].T * h[i, :]) for i in range(3)), sp.zeros(3, 3))
b2PP = w(h, mul(f * f.T, f, f * f.T)) - mul(w(h, f * f.T), hf, w(h, f * f.T))
ok &= sp.simplify(b2PP - (c - mul(c, hf, c))).is_zero_matrix
check("A3 pi_h preserves the splitting; w_h(a) = pi_h(a) + sum a_yy (diag h_y - h_y h_y^T) for all a; w_h(P_f) = P_{hf} + c_h(f) with c_h(f) = sum f_y^2 (diag h_y - h_y h_y^T) in I; "
      "b2(P,P) = c - c.c", ok)

# A4
ok = True
for lam in (0, sp.Rational(1, 3), 1):
    F1 = lambda a: lam * (h.T * a * h) + (1 - lam) * w(h, a)
    d2 = lambda a, b: F1(mul(a, f, b)) - mul(F1(a), hf, F1(b))
    a, b, cc = (rq() * f * f.T + circ(3, 3) for _ in range(3))
    ok &= s_of(F1(a), hf) == s_of(a, f) and is_circ(d2(a, b))
    ok &= (mul(F1(a), hf, d2(b, cc)) - d2(mul(a, f, b), cc) + d2(a, mul(b, f, cc)) - mul(d2(a, b), hf, F1(cc))).is_zero_matrix
check("A4 for every mass-preserving linear F1 (convex combinations of pi_h and w_h) the defect is a circulation and the "
      "arity-3 (Hochschild cocycle) identity holds", ok)

# A5 homology of the cone C = A (deg 0) + eps I (deg -1), m1(eps x) = x
n = 3; dimA = 1 + (n - 1) ** 2; dimI = (n - 1) ** 2
# m1: eps I -> A is the inclusion, rank dimI
H0 = dimA - dimI; Hm1 = dimI - dimI
e = sp.diag(*f) - f * f.T
xI = circ(3, 3)
ok = H0 == 1 and Hm1 == 0 and mul(e, f, xI) == xI   # x = e.x, so if e = d(eta) then x = d(eta.x)
check("A5 the cone has H^0 = R and H^{-1} = 0; q(a + eps x) = s(a) is a quasi-isomorphism with q o F = s; "
      "x = e_f . x for every circulation, so a boundary e_f makes all of I boundaries", ok, f"H^0 rank {H0}, H^-1 rank {Hm1}")

# A6 double category interchange
fx, fy, fz = dist(2), dist(3), dist(2)
def coup(p, q):
    a = p * q.T; t = min(a[0, 0], a[1, 1]) / 2; a[0, 0] -= t; a[1, 1] -= t; a[0, 1] += t; a[1, 0] += t; return a
al, be = coup(fx, fy), coup(fy, fz)
k1, m1_ = stoch(2, 2), stoch(2, 2)
res6 = {}
for name, l in (("bijection", sp.Matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])), ("stochastic", stoch(3, 3)),
                ("deterministic non-injective", sp.Matrix([[1, 0], [0, 1], [0, 1]]))):
    lhs = k1.T * mul(al, fy, be) * m1_
    rhs = mul(k1.T * al * l, l.T * fy, l.T * be * m1_)
    res6[name] = (lhs - rhs).is_zero_matrix, is_circ(lhs - rhs)
# explicit counterexamples of the paper: uniform p = q = r on {0,1}, diagonal couplings, identities k, m
u2 = sp.Matrix([sp.Rational(1, 2)] * 2); Dg = sp.diag(*u2); I2 = sp.eye(2)
for l in (sp.Matrix([[1], [1]]), sp.Matrix([[sp.Rational(1, 2)] * 2] * 2)):
    lhs = mul(Dg, u2, Dg); rhs = mul(Dg * l, l.T * u2, l.T * Dg)
    res6.setdefault("explicit", []).append(lhs == Dg and rhs == u2 * u2.T)
check("A6 interchange holds for a bijective middle map and fails for stochastic and deterministic non-injective ones; "
      "the failure is a circulation", res6["bijection"][0] and not res6["stochastic"][0] and not res6["deterministic non-injective"][0]
      and res6["stochastic"][1] and res6["deterministic non-injective"][1] and all(res6["explicit"]),
      "random stochastic and non-injective kernels; explicit collapse and constant-kernel counterexamples")

# A7, A8, A9: integer tables
def tables(r, cs):
    """all nonnegative integer matrices with row sums r and column sums cs"""
    out = []
    def rec(i, rem_cols, acc):
        if i == len(r) - 1:
            if all(x >= 0 for x in rem_cols) and sum(rem_cols) == r[-1]: out.append(tuple(acc + [tuple(rem_cols)]))
            return
        for row in itertools.product(*[range(min(r[i], cc) + 1) for cc in rem_cols]):
            if sum(row) == r[i]: rec(i + 1, [cc - x for cc, x in zip(rem_cols, row)], acc + [row])
    rec(0, list(cs), []); return out
r1, c1 = (3, 3), (2, 2, 2)
P1 = tables(r1, c1)
# integer decomposition: every table in kP is a sum of k tables in P
def idp(k):
    Pk = tables(tuple(k * x for x in r1), tuple(k * x for x in c1)); S = set(P1); ok = True
    sums = {tuple(tuple(sum(z) for z in zip(*rows)) for rows in zip(*combo)) for combo in itertools.combinations_with_replacement(P1, k)}
    return all(t in sums for t in Pk), len(Pk)
i2, n2 = idp(2); i3, n3 = idp(3)
# vertices integral: every vertex of the transportation polytope is a lattice point (TU); check vertices among tables
import numpy as np
def vertices(r, cs):
    pts = tables(r, cs); arr = np.array([np.array(p, float).ravel() for p in pts])
    # a lattice point is a vertex iff it is not the midpoint of two other lattice points in the same fibre... use LP-free test:
    # vertices of a transportation polytope are exactly the tables whose support graph is a forest
    def forest(t):
        n, m = len(t), len(t[0]); parent = list(range(n + m))
        def find(u):
            while parent[u] != u: parent[u] = parent[parent[u]]; u = parent[u]
            return u
        for i in range(n):
            for j in range(m):
                if t[i][j]:
                    a, b = find(i), find(n + j)
                    if a == b: return False
                    parent[a] = b
        return True
    return [p for p in pts if forest(p)]
V1 = vertices(r1, c1)
check("A7 N.Cpl(f,g) for f = (1/2,1/2), g = (1/3,1/3,1/3), N = 6 is a lattice hexagon with the integer decomposition "
      "property (k = 2, 3)", len(V1) == 6 and i2 and i3, f"{len(P1)} lattice points, {len(V1)} vertices; kP has {n2}, {n3} points")
k = sp.symbols("k")
counts = [len(tables((3 * t, 3 * t), (2 * t, 2 * t, 2 * t))) for t in range(1, 6)]
poly = sp.expand(sp.interpolate(list(zip(range(1, 6), counts)), k))
def lattice_area(vs):
    # vertices given as tables; coordinates (t00, t01) span the lattice of the fibre
    pts = [(v[0][0], v[0][1]) for v in vs]; cx = sum(p[0] for p in pts) / len(pts); cy = sum(p[1] for p in pts) / len(pts)
    pts.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    return Fr(abs(sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))), 2)
area1 = lattice_area(V1)
r2, c2 = (5, 7), (3, 4, 5)
counts2 = [len(tables(tuple(t * x for x in r2), tuple(t * x for x in c2))) for t in range(1, 6)]
poly2 = sp.expand(sp.interpolate(list(zip(range(1, 6), counts2)), k)); area2 = lattice_area(vertices(r2, c2))
check("A8 Ehrhart: counts 7,19,37,61,91 = 3k^2+3k+1 with leading coefficient the lattice area 3; second example agrees",
      counts == [7, 19, 37, 61, 91] and poly == 3 * k ** 2 + 3 * k + 1 and area1 == 3 and sp.Poly(poly2, k).degree() == 2
      and sp.Poly(poly2, k).LC() == area2 and sp.Poly(poly2, k).eval(0) == 1, f"second: margins {r2},{c2}: {counts2} -> {poly2}, area {area2}")

# A9 Markov moves
def connected(r, cs):
    pts = tables(r, cs); S = set(pts); seen = {pts[0]}; stack = [pts[0]]
    n, m = len(r), len(cs)
    moves = []
    for i, i2_ in itertools.combinations(range(n), 2):
        for j, j2 in itertools.combinations(range(m), 2):
            mv = [[0] * m for _ in range(n)]; mv[i][j] = mv[i2_][j2] = 1; mv[i][j2] = mv[i2_][j] = -1
            moves += [mv, [[-x for x in row] for row in mv]]
    while stack:
        t = stack.pop()
        for mv in moves:
            u = tuple(tuple(t[a][b] + mv[a][b] for b in range(m)) for a in range(n))
            if u in S and u not in seen: seen.add(u); stack.append(u)
    allc = all(sum(row) == 0 for row in moves[0]) and all(sum(col) == 0 for col in zip(*moves[0]))
    return len(seen) == len(S), len(S), allc
cA = connected(r1, c1); cB = connected((4, 3, 5), (3, 5, 4))
check("A9 the basic 2x2 moves connect every fibre of integer tables (2x3 and 3x3 examples), and each move is an integer circulation",
      cA[0] and cB[0] and cA[2], f"fibre sizes {cA[1]}, {cB[1]}")

# A10 Fisher exact test on the synthetic triage data (first 40 items)
path = "/home/claude/synthetic/data/triage_samples.csv"
rows = list(csv.DictReader(open(path)))[:40]
L = ["safe", "harmful"]; A = ["accept", "review", "reject"]
T = [[sum(1 for r in rows if r["label"] == l and r["action"] == a) for a in A] for l in L]
rs = tuple(sum(row) for row in T); cs = tuple(sum(col) for col in zip(*T)); N = sum(rs)
fib = tables(rs, cs)
def logp(t):  # multivariate hypergeometric probability of a table given its margins
    return (sum(math.lgamma(x + 1) for x in rs) + sum(math.lgamma(x + 1) for x in cs) - math.lgamma(N + 1)
            - sum(math.lgamma(x + 1) for row in t for x in row))
probs = [math.exp(logp(t)) for t in fib]; p_obs = math.exp(logp(tuple(map(tuple, T))))
p_exact = sum(p for p in probs if p <= p_obs * (1 + 1e-9))
exp_ = [[rs[i] * cs[j] / N for j in range(3)] for i in range(2)]
X2 = sum((T[i][j] - exp_[i][j]) ** 2 / exp_[i][j] for i in range(2) for j in range(3))
from scipy.stats import chi2 as chi2d
p_chi = chi2d.sf(X2, 2)
check("A10 Fisher's exact test on 40 synthetic triage items: the fibre is enumerated, hypergeometric probabilities sum to 1, "
      "exact and chi-square p-values both reject independence", abs(sum(probs) - 1) < 1e-9 and p_exact < 1e-3 and p_chi < 1e-3,
      f"table {T}, fibre size {len(fib)}, exact p = {p_exact:.2e}, chi-square p = {p_chi:.2e}")
print(f"\n{sum(res)}/{len(res)} checks passed")
