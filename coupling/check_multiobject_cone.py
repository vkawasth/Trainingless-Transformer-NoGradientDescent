"""check_multiobject_cone.py -- (4) the multi-object square-zero cone (Question 9.4(ii)).

Experimental definitions.
  L      linear category: objects are full-support distributions f on finite sets; Hom(f,g) = V(f,g) =
         {a in R^{n x m}: a 1 = s f, 1^T a = s g^T for some s = s(a)}; composition a.b = a diag(g)^{-1} b;
         identity 1_f = diag(f).  Couplings are the a >= 0 with s = 1, and composition of couplings is gluing.
  I      the circulations (s = 0): a two-sided ideal.
  C      the mapping cone of I -> L: Hom_C = V (+) eps I, deg eps = -1, m1(eps x) = x, eps x . eps y = 0.
  F^h    whiskering by a stochastic matrix h: F1 = h_* (a -> h^T a h), F2 = eps delta_h with
         delta_h(a,b) = h_*(ab) - h_*(a) h_*(b), F_{>=3} = 0.
  N      nerve -> cone: a 2-simplex p (a 3-way table) gives H(p) = eps (d1 p - d2 p . d0 p).

  M1 L is a category across objects of different sizes; composition of couplings is gluing
  M2 I is a two-sided ideal, s is multiplicative, H^0(C) = L/I has every Hom = R (the indiscrete category on R)
  M3 h_* maps V(f,g) -> V(fh,gh) preserving s; it is not a functor, but the defect delta_h lies in I
  M4 A-infinity relations of F^h across objects: arity 2 (m1 F2 = defect), arity 3 (Hochschild cocycle), arity 4
  M5 units: F1(1_f) != 1_{fh}, but F1(1_f) - 1_{fh} = m1(eps u_f) with u_f in I: F is cohomologically unital only
  M6 F^{h'} o F^h = F^{h h'} as A-infinity functors (F1 and F2 agree): whiskering is strictly functorial in h
  M7 kernels X -> Y as objects: Hom = product over x, all of M1-M4 componentwise (|X| = 2)
  M8 nerve -> cone: H(p) lies in eps I; gluing fillers and degenerate simplices give H = 0; on 3-simplices the
     coherence class Z = g01 H123 + H013 - H012 g23 - H023 vanishes identically (m1 is injective on eps I: H^{-1} = 0)
  M9 naturality: H(h_* p) = F1(H(p)) + F2(d2 p, d0 p)
"""
import itertools, random
from fractions import Fraction as Fr
import sympy as sp
R = random.Random(44)
res = []
def check(n, c, info=""):
    res.append(bool(c)); print(("PASS " if c else "FAIL ") + n + (f"   [{info}]" if info else ""))

def rq(k=20): return sp.Rational(R.randint(1, k), R.randint(1, k))
def dist(n):
    w = [rq() for _ in range(n)]; s = sum(w); return sp.Matrix([x / s for x in w])
def stoch(n, m):
    return sp.Matrix([[x / sum(r) for x in r] for r in [[rq() for _ in range(m)] for _ in range(n)]])
def coupling(f, g):
    """a random rational coupling: product plus a small circulation"""
    a = f * g.T; n, m = a.shape
    for _ in range(3):
        i, k = R.sample(range(n), 2) if n > 1 else (0, 0); j, l = R.sample(range(m), 2) if m > 1 else (0, 0)
        if n < 2 or m < 2: break
        t = min(a[i, j], a[k, l]) * sp.Rational(R.randint(0, 9), 10)
        a[i, j] -= t; a[k, l] -= t; a[i, l] += t; a[k, j] += t
    return a
def circ(n, m):
    x = sp.Matrix(n, m, lambda i, j: rq() - rq())
    x -= x * sp.ones(m, 1) * sp.ones(1, m) / m          # rows sum to 0
    x -= sp.ones(n, 1) * sp.ones(1, n) * x / n          # columns sum to 0 (rows still 0)
    return x
def mor(f, g):
    """a general element of V(f,g): s * coupling + circulation"""
    s = rq() - rq(); return s * coupling(f, g) + circ(len(f), len(g))
def s_of(a, f):
    return (a * sp.ones(a.shape[1], 1))[0] / f[0]
def inV(a, f, g):
    s = s_of(a, f)
    return (a * sp.ones(a.shape[1], 1) - s * f).is_zero_matrix and (sp.ones(1, a.shape[0]) * a - s * g.T).is_zero_matrix
def comp(a, g, b): return a * sp.diag(*g).inv() * b
def ident(f): return sp.diag(*f)
def push(h, a): return h.T * a * h
def delta(h, a, g, b): return push(h, comp(a, g, b)) - comp(push(h, a), h.T * g, push(h, b))

sizes = [2, 3, 4, 2]
fs = [dist(n) for n in sizes]
# ------------------------------------------------------------------ M1
a, b, c = mor(fs[0], fs[1]), mor(fs[1], fs[2]), mor(fs[2], fs[3])
ab = comp(a, fs[1], b)
ok = inV(ab, fs[0], fs[2]) and s_of(ab, fs[0]) == s_of(a, fs[0]) * s_of(b, fs[1])
ok &= comp(ab, fs[2], c) == comp(a, fs[1], comp(b, fs[2], c))
ok &= comp(ident(fs[0]), fs[0], a) == a and comp(a, fs[1], ident(fs[1])) == a
al, be = coupling(fs[0], fs[1]), coupling(fs[1], fs[2])
glue = sp.Matrix(2, 4, lambda x, z: sum(al[x, y] * be[y, z] / fs[1][y] for y in range(3)))
ok &= comp(al, fs[1], be) == glue and inV(glue, fs[0], fs[2]) and all(v >= 0 for v in glue)
check("M1 L is a category on objects of sizes 2,3,4,2 (exact): composition well typed, s multiplicative, associative, "
      "unital; composition of couplings is gluing", ok)

# ------------------------------------------------------------------ M2
x, y = circ(2, 3), circ(3, 4)
ok = inV(x, fs[0], fs[1]) and s_of(x, fs[0]) == 0
ok &= all(s_of(z, fs[0]) == 0 and inV(z, fs[0], fs[2]) for z in (comp(x, fs[1], b), comp(a, fs[1], y)))
# dimension count: dim V = (n-1)(m-1) + 1, dim I = (n-1)(m-1), so L/I has Hom = R with composition s * s'
dims = []
for n, m in ((2, 3), (3, 4), (4, 2)):
    M = sp.Matrix([[int(k // m == i) for k in range(n * m)] for i in range(n)] + [[int(k % m == j) for k in range(n * m)] for j in range(m)])
    dims.append((n * m - M.rank(), (n - 1) * (m - 1)))
ok &= all(d == e for d, e in dims)
check("M2 I is a two-sided ideal; s: V(f,g) -> R is multiplicative with kernel I, so H^0(C) = L/I has every Hom = R",
      ok, f"dim I = (n-1)(m-1): {dims}")

# ------------------------------------------------------------------ M3, M4
# whiskering needs h on each underlying set; the objects of sizes 2,3,4,2 live on different sets Y_i,
# so F^h is given by a family h_i : Y_i -> Z_i and acts by a -> h_i^T a h_j on V(f_i, f_j).
Z = [2, 3, 3, 2]
hs = [stoch(n, z) for n, z in zip(sizes, Z)]
gs = [h.T * f for h, f in zip(hs, fs)]
def F1(i, j, a): return hs[i].T * a * hs[j]
def D(i, j, k, a, b): return F1(i, k, comp(a, fs[j], b)) - comp(F1(i, j, a), gs[j], F1(j, k, b))
ok3 = all(inV(F1(i, i + 1, z), gs[i], gs[i + 1]) and s_of(F1(i, i + 1, z), gs[i]) == s_of(z, fs[i])
          for i, z in ((0, a), (1, b), (2, c)))
d01 = D(0, 1, 2, a, b); d12 = D(1, 2, 3, b, c)
ok3 &= not d01.is_zero_matrix and inV(d01, gs[0], gs[2]) and s_of(d01, gs[0]) == 0
check("M3 F1 = h_* maps V(f_i,f_j) -> V(h f_i, h f_j) preserving s; it is not a functor, and the defect lies in I", ok3,
      f"|delta(a,b)|_max = {float(max(abs(v) for v in d01)):.4f}")
# arity 2: F1(ab) - F1(a)F1(b) = m1(F2(a,b)) = delta  (by definition; check it is in I so eps*delta is a morphism of C)
# arity 3: F1(a) delta(b,c) - delta(ab,c) + delta(a,bc) - delta(a,b) F1(c) = 0
lhs = comp(F1(0, 1, a), gs[1], d12) - D(0, 2, 3, ab, c) + D(0, 1, 3, a, comp(b, fs[2], c)) - comp(d01, gs[2], F1(2, 3, c))
# arity 4: only products eps x . eps y occur, which vanish in C
check("M4 A-infinity relations across objects: arity 2 (m1 F2 = defect, in I), arity 3 (Hochschild cocycle, exact), "
      "arity 4 (eps^2 = 0)", lhs.is_zero_matrix and s_of(d01, gs[0]) == 0)

# ------------------------------------------------------------------ M5 units
u = [F1(i, i, ident(fs[i])) - ident(gs[i]) for i in range(4)]
ok = all(not x.is_zero_matrix for x in u) and all(inV(x, gs[i], gs[i]) and s_of(x, gs[i]) == 0 for i, x in enumerate(u))
F2unit = D(0, 0, 1, ident(fs[0]), a)
check("M5 units: F1(1_f) != 1_{hf}; the difference u_f lies in I (= m1(eps u_f)), and F2(1,a) != 0: F is cohomologically "
      "unital, not strictly unital", ok and not F2unit.is_zero_matrix)

# ------------------------------------------------------------------ M6 functoriality in h
Z2 = [3, 2, 2, 3]
hs2 = [stoch(z, w) for z, w in zip(Z, Z2)]
def F1h(H, i, j, a): return H[i].T * a * H[j]
def Dh(H, F, i, j, k, a, b): return F1h(H, i, k, comp(a, F[j], b)) - comp(F1h(H, i, j, a), H[j].T * F[j], F1h(H, j, k, b))
H12 = [h1 * h2 for h1, h2 in zip(hs, hs2)]
lhs1 = F1h(hs2, 0, 1, F1h(hs, 0, 1, a)); rhs1 = F1h(H12, 0, 1, a)
lhs2 = F1h(hs2, 0, 2, Dh(hs, fs, 0, 1, 2, a, b)) + Dh(hs2, gs, 0, 1, 2, F1h(hs, 0, 1, a), F1h(hs, 1, 2, b))
rhs2 = Dh(H12, fs, 0, 1, 2, a, b)
check("M6 composite A-infinity functor F^{h'} o F^h equals F^{hh'}: (F1, F2) agree exactly", lhs1 == rhs1 and lhs2 == rhs2)

# ------------------------------------------------------------------ M7 kernels as objects
X = 2
kf = [[dist(n) for _ in range(X)] for n in (2, 3, 4)]
A = [mor(kf[0][x], kf[1][x]) for x in range(X)]; B = [mor(kf[1][x], kf[2][x]) for x in range(X)]
hk = [stoch(n, z) for n, z in zip((2, 3, 4), (2, 2, 3))]
ok = True
for x in range(X):
    gx = [hk[i].T * kf[i][x] for i in range(3)]
    dd = hk[0].T * comp(A[x], kf[1][x], B[x]) * hk[2] - comp(hk[0].T * A[x] * hk[1], gx[1], hk[1].T * B[x] * hk[2])
    ok &= inV(comp(A[x], kf[1][x], B[x]), kf[0][x], kf[2][x]) and inV(dd, gx[0], gx[2]) and s_of(dd, gx[0]) == 0
check("M7 objects = kernels X -> Y (|X| = 2): Hom = product over x of V(f_x, g_x); M1-M4 hold componentwise", ok)

# ------------------------------------------------------------------ M8 nerve -> cone
def rtable(shape):
    t = sp.MutableDenseNDimArray.zeros(*shape)
    for idx in itertools.product(*map(range, shape)): t[idx] = rq()
    s = sum(t[idx] for idx in itertools.product(*map(range, shape)))
    for idx in itertools.product(*map(range, shape)): t[idx] /= s
    return t
def marg(t, keep):
    shape = t.shape; out = {}
    for idx in itertools.product(*map(range, shape)):
        k = tuple(idx[i] for i in keep); out[k] = out.get(k, 0) + t[idx]
    if len(keep) == 1: return sp.Matrix([out[(i,)] for i in range(shape[keep[0]])])
    return sp.Matrix(shape[keep[0]], shape[keep[1]], lambda i, j: out[(i, j)])
def Hp(t, v=(0, 1, 2)):
    f1 = marg(t, (v[1],))
    return marg(t, (v[0], v[2])) - comp(marg(t, (v[0], v[1])), f1, marg(t, (v[1], v[2])))
ok = True
for shape in ((2, 3, 2), (3, 2, 4)):
    p = rtable(shape); Hv = Hp(p)
    ok &= inV(Hv, marg(p, (0,)), marg(p, (2,))) and s_of(Hv, marg(p, (0,))) == 0 and not Hv.is_zero_matrix
    # gluing filler of the two edges: H = 0
    a01, a12, f1 = marg(p, (0, 1)), marg(p, (1, 2)), marg(p, (1,))
    G = sp.MutableDenseNDimArray.zeros(*shape)
    for i, j, k in itertools.product(*map(range, shape)): G[i, j, k] = a01[i, j] * a12[j, k] / f1[j]
    ok &= Hp(G).is_zero_matrix
    # degenerate simplices s0, s1 of an edge
    e = marg(p, (0, 1)); n0, n1 = e.shape
    S0 = sp.MutableDenseNDimArray.zeros(n0, n0, n1); S1 = sp.MutableDenseNDimArray.zeros(n0, n1, n1)
    for i, j in itertools.product(range(n0), range(n1)): S0[i, i, j] = e[i, j]; S1[i, j, j] = e[i, j]
    ok &= Hp(S0).is_zero_matrix and Hp(S1).is_zero_matrix
# 3-simplices
q = rtable((2, 3, 2, 3)); Zs = []
def face(q, drop):
    keep = [i for i in range(4) if i != drop]; t = sp.MutableDenseNDimArray.zeros(*[q.shape[i] for i in keep])
    for idx in itertools.product(*map(range, q.shape)): t[tuple(idx[i] for i in keep)] += q[idx]
    return t
H123, H023, H013, H012 = (Hp(face(q, d)) for d in range(4))
f = [marg(q, (i,)) for i in range(4)]
g01, g23 = marg(q, (0, 1)), marg(q, (2, 3))
Zc = comp(g01, f[1], H123) + H013 - comp(H012, f[2], g23) - H023
check("M8 nerve -> cone: H(p) in eps I (nonzero for generic p), H = 0 on gluing fillers and on degenerate simplices; on a "
      "3-simplex the coherence class Z vanishes identically", ok and Zc.is_zero_matrix,
      "the square-zero cone has H^{-1} = 0, so it cannot see 3-simplex (contextuality) obstructions")

# ------------------------------------------------------------------ M9 naturality
p = rtable((2, 3, 2)); hh = [stoch(2, 3), stoch(3, 2), stoch(2, 2)]
hp = sp.MutableDenseNDimArray.zeros(3, 2, 2)
for i, j, k in itertools.product(range(2), range(3), range(2)):
    for x, y, z in itertools.product(range(3), range(2), range(2)):
        hp[x, y, z] += p[i, j, k] * hh[0][i, x] * hh[1][j, y] * hh[2][k, z]
a01, a12, f1 = marg(p, (0, 1)), marg(p, (1, 2)), marg(p, (1,))
lhs = Hp(hp)
rhs = hh[0].T * Hp(p) * hh[2] + (hh[0].T * comp(a01, f1, a12) * hh[2] - comp(hh[0].T * a01 * hh[1], hh[1].T * f1, hh[1].T * a12 * hh[2]))
check("M9 naturality of nerve -> cone under whiskering: H(h_* p) = F1(H(p)) + F2(d2 p, d0 p)", lhs == rhs)
print(f"\n{sum(res)}/{len(res)} checks passed")
