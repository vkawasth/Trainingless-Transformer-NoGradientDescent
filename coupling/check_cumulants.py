"""check_cumulants.py -- checks for the section "Cumulants and the homotopy viewpoint" of the
coupling-polytope companion.  Exact rational arithmetic throughout.

  Q1 a centred coupling alpha - f(x)g is the cross-covariance of the indicator vectors; it is a circulation;
     the product coupling is the unique rank-one point of Cpl(f, g) (Birch point of the independence model)
  Q2 level two: mu*(Pi) = mu P (x) mu Q + Cov_Pi(rho, rho'), the cross-covariance of the random measures
  Q3 level two: mixed moments of (rho, rho') up to order (n-1, n-1) determine Pi (three states on two
     outcomes); flattening keeps only order (1, 1)
  E1 the endomorphism algebra A_f (couplings of f with itself, product = gluing, unit = diagonal) is
     associative and unital, and circulations form a two-sided ideal
  E2 whiskering h_*: A_f -> A_{h#f} is linear and unital; its second Boolean cumulant
     b2(b, a) = h_*(b.a) - h_*(b).h_*(a) is the whiskering defect, and lies in the circulation ideal
  E3 the defects span the whole circulation space (3 points), so the quotient by the ideal they generate
     keeps exactly the marginals
  E4 modulo circulations, whiskering is an algebra map (marginals compose)
  E5 the third Boolean cumulant b3 is a circulation and is nonzero in general; the arity-3 obstruction
     of the A-infinity functor equation is a cycle (an exact identity)
"""
import random, itertools
from fractions import Fraction as F
R = random.Random(9)
res = []
def check(n, c, info=""):
    res.append(bool(c)); print(("PASS " if c else "FAIL ") + n + (f"   [{info}]" if info else ""))

def rd(k, zeros=False):
    w = [F(R.randint(0 if zeros else 1, 6)) for _ in range(k)]
    if sum(w) == 0: w[0] = F(1)
    s = sum(w); return [x / s for x in w]
def mat(n, m, v=F(0)): return [[v] * m for _ in range(n)]
def add(A, B, s=1): return [[a + s * b for a, b in zip(r1, r2)] for r1, r2 in zip(A, B)]
def rows(A): return [sum(r) for r in A]
def cols(A): return [sum(A[i][j] for i in range(len(A))) for j in range(len(A[0]))]
def is_circ(A): return all(x == 0 for x in rows(A)) and all(x == 0 for x in cols(A))
def rcoupling(f, g):
    """random coupling of f and g: mix of product and a random vertex-like perturbation, kept >= 0"""
    P = [[a * b for b in g] for a in f]
    n, m = len(f), len(g)
    for _ in range(3):
        i, k = R.sample(range(n), 2) if n > 1 else (0, 0); j, l = R.sample(range(m), 2) if m > 1 else (0, 0)
        if n < 2 or m < 2: break
        t = min(P[i][l], P[k][j]) * F(R.randint(0, 4), 5)
        P[i][j] += t; P[k][l] += t; P[i][l] -= t; P[k][j] -= t
    return P

# ---------------------------------------------------------------- Q1
ok = True
for _ in range(30):
    f, g = rd(3), rd(4); a = rcoupling(f, g)
    C = [[a[i][j] - f[i] * g[j] for j in range(4)] for i in range(3)]
    # covariance of indicators 1{y=i}, 1{y'=j}: E[1_i 1'_j] - E[1_i]E[1'_j]
    cov = [[a[i][j] - sum(a[i]) * sum(a[k][j] for k in range(3)) for j in range(4)] for i in range(3)]
    ok &= C == cov and is_circ(C)
# rank-one couplings with margins f, g: a = u v^T >= 0 forces a = f g^T
ok2 = True
for _ in range(30):
    u = [F(R.randint(1, 5)) for _ in range(3)]; v = [F(R.randint(1, 5)) for _ in range(4)]
    a = [[x * y for y in v] for x in u]; s = sum(map(sum, a)); a = [[x / s for x in r] for r in a]
    ok2 &= a == [[x * y for y in cols(a)] for x in rows(a)]
check("Q1 alpha - f(x)g is the cross-covariance of the indicators and a circulation; rank-one points are products", ok and ok2)

# ---------------------------------------------------------------- Q2, Q3
def flat(Pi, Ys):
    nb = len(Ys[0])
    return [[sum(Pi[a][b] * Ys[a][i] * Ys[b][j] for a in range(len(Ys)) for b in range(len(Ys))) for j in range(nb)] for i in range(nb)]
ok = True
for _ in range(30):
    Ys = [rd(3, True) for _ in range(3)]; P, Q = rd(3), rd(3); Pi = rcoupling(P, Q)
    muP = [sum(P[a] * Ys[a][i] for a in range(3)) for i in range(3)]; muQ = [sum(Q[b] * Ys[b][i] for b in range(3)) for i in range(3)]
    cov = [[sum(Pi[a][b] * (Ys[a][i] - muP[i]) * (Ys[b][j] - muQ[j]) for a in range(3) for b in range(3)) for j in range(3)] for i in range(3)]
    ok &= flat(Pi, Ys) == add([[x * y for y in muQ] for x in muP], cov) and is_circ(cov)
check("Q2 mu*(Pi) = mu P (x) mu Q + Cov_Pi(rho, rho'), and the covariance term is a circulation", ok)
def rank(M):
    M = [r[:] for r in M]; r = 0
    for c in range(len(M[0])):
        p = next((i for i in range(r, len(M)) if M[i][c] != 0), None)
        if p is None: continue
        M[r], M[p] = M[p], M[r]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                t = M[i][c] / M[r][c]; M[i] = [x - t * y for x, y in zip(M[i], M[r])]
        r += 1
    return r
t = [F(1), F(1, 2), F(0)]                          # rho(0) for three states on two outcomes
basis = []                                         # circulations of 3x3: E_ij - E_i2 - E_2j + E_22
for i in range(2):
    for j in range(2):
        B = mat(3, 3); B[i][j] += 1; B[i][2] -= 1; B[2][j] -= 1; B[2][2] += 1; basis.append(B)
def moment(Pi, a, b): return sum(Pi[x][y] * t[x] ** a * t[y] ** b for x in range(3) for y in range(3))
rk11 = rank([[moment(B, 1, 1) for B in basis]])
rkall = rank([[moment(B, a, b) for (a, b) in ((1, 1), (1, 2), (2, 1), (2, 2))] for B in basis])
check("Q3 on the level-two fibre, flattening sees the (1,1) mixed moment only; moments up to (2,2) determine the coupling",
      rk11 == 1 and rkall == 4, f"rank of order-(1,1) map {rk11}, of orders <= (2,2) map {rkall} on a 4-dim space")

# ---------------------------------------------------------------- E: whiskering as a map of algebras
def glue(a, b, mid):
    return [[sum(a[i][k] * b[k][j] / mid[k] for k in range(len(mid)) if mid[k]) for j in range(len(b[0]))] for i in range(len(a))]
def post(a, h):
    nz = len(h[0]); out = mat(nz, nz)
    for y in range(len(a)):
        for yp in range(len(a)):
            if a[y][yp] == 0: continue
            for z in range(nz):
                if y == yp: out[z][z] += a[y][y] * h[y][z]
                else:
                    for zp in range(nz): out[z][zp] += a[y][yp] * h[y][z] * h[yp][zp]
    return out
def diag(f): return [[f[i] if i == j else F(0) for j in range(len(f))] for i in range(len(f))]
def rmat(n): return [[F(R.randint(-3, 3)) for _ in range(n)] for _ in range(n)]
ok_assoc = ok_unit = ok_ideal = True
for _ in range(30):
    f = rd(3); a, b, c = rmat(3), rmat(3), rmat(3)
    ok_assoc &= glue(glue(a, b, f), c, f) == glue(a, glue(b, c, f), f)
    ok_unit &= glue(diag(f), a, f) == a and glue(a, diag(f), f) == a
    z = add(rmat(3), mat(3, 3))
    # project z to a circulation: subtract row/col means
    zc = [[z[i][j] - sum(z[i]) / 3 - sum(z[k][j] for k in range(3)) / 3 + sum(map(sum, z)) / 9 for j in range(3)] for i in range(3)]
    p = rcoupling(f, f)
    ok_ideal &= is_circ(zc) and is_circ(glue(zc, p, f)) and is_circ(glue(p, zc, f))
check("E1 A_f (gluing over f) is associative and unital; circulations form a two-sided ideal against couplings", ok_assoc and ok_unit and ok_ideal)
def hk(n, m): return [rd(m) for _ in range(n)]
def push(h, f): return [sum(f[y] * h[y][z] for y in range(len(f))) for z in range(len(h[0]))]
def b2(h, f, b, a):
    hf = push(h, f); return add(post(glue(a, b, f), h), glue(post(a, h), post(b, h), hf), -1)
ok_lin = ok_b2 = True; defects = []
for _ in range(40):
    f = rd(3); h = hk(3, 3); a, b = rcoupling(f, f), rcoupling(f, f); s = F(R.randint(1, 5), 7)
    ok_lin &= post(add([[s * x for x in r] for r in a], b), h) == add([[s * x for x in r] for r in post(a, h)], post(b, h))
    ok_lin &= post(diag(f), h) == diag(push(h, f))
    d = b2(h, f, b, a); ok_b2 &= is_circ(d); defects.append(d)
check("E2 whiskering is linear and unital; its second Boolean cumulant (the defect) is a circulation", ok_lin and ok_b2)
vecs = [[d[i][j] for i in range(3) for j in range(3)] for d in defects]
check("E3 the defects span the whole circulation space (dimension 4 for 3 points)", rank(vecs) == 4, f"rank {rank(vecs)}")
ok = True
for _ in range(30):
    f = rd(3); h = hk(3, 3); a, b = rcoupling(f, f), rcoupling(f, f)
    lhs, rhs = post(glue(a, b, f), h), glue(post(a, h), post(b, h), push(h, f))
    ok &= rows(lhs) == rows(rhs) and cols(lhs) == cols(rhs)
check("E4 modulo circulations whiskering is an algebra map: the composites have equal marginals", ok)
def b3(h, f, c, b, a):
    hf = push(h, f); g = lambda x, y: glue(x, y, hf); P = lambda x: post(x, h)
    full = P(glue(glue(a, b, f), c, f))
    t1 = g(g(P(a), P(b)), P(c))
    t2 = g(b2(h, f, b, a), P(c))                    # interval partition {ab}{c}
    t3 = g(P(a), b2(h, f, c, b))                    # {a}{bc}
    return add(add(add(full, t1, -1), t2, -1), t3, -1)
ok_c = True; nonzero = False; ok_cycle = True
for _ in range(30):
    f = rd(3); h = hk(3, 3); a, b, c = rcoupling(f, f), rcoupling(f, f), rcoupling(f, f)
    x3 = b3(h, f, c, b, a); ok_c &= is_circ(x3); nonzero |= any(v != 0 for r in x3 for v in r)
    # arity-3 obstruction: with D(x, y) = F1(x.y) - F1 x . F1 y,
    # F1c.D(b,a)... written for the product order used here: D(b, a) := P(a.b) - P(a).P(b)
    hf = push(h, f); g = lambda x, y: glue(x, y, hf); P = lambda x: post(x, h)
    D = lambda x, y: add(P(glue(x, y, f)), g(P(x), P(y)), -1)
    E3 = add(add(add(g(P(a), D(b, c)), g(D(a, b), P(c)), -1), D(glue(a, b, f), c), -1), D(a, glue(b, c, f)))
    ok_cycle &= all(v == 0 for r in E3 for v in r)
check("E5 the third Boolean cumulant is a circulation and is nonzero in general; the arity-3 obstruction is a cycle",
      ok_c and nonzero and ok_cycle)
print(f"\n{sum(res)}/{len(res)} checks passed")
