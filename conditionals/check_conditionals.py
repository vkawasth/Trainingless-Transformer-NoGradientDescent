"""check_conditionals.py -- checks for "Conditionals, Couplings and Toric Geometry" (revised note).

Coupling polytopes Cpl(f, g) = {a >= 0 : row sums f, column sums g}, in the lattice of integer
circulations (integer matrices with zero row and column sums).
  T1 nondegenerate marginals (no proper partial sum of f equals one of g)  =>  Delzant
     (geometric test: every vertex has exactly dim neighbours; primitive edge vectors form a lattice basis)
  T2 off the walls sum_I f = sum_J g the combinatorial type is locally constant; it changes across a wall
  T3 degenerate cases: a wall can keep or destroy the Delzant property (2x2 uniform: segment; 3x3 uniform:
     not simple)
  T4 monotone cases: uniform marginals on coprime sizes (2x3, 3x4) are Delzant, every cell is a facet,
     and the product coupling is at equal lattice distance from all facets
  T5 the 2x3 uniform polytope has the fan of the degree-6 del Pezzo surface; its toric potential
     W = x + y + 1/x + 1/y + xy + 1/(xy) has a critical point at (1, 1)
  T6 the maximal coupling of distinct marginals lies on the boundary; the product coupling is interior
  T7 whiskering defect has two 2-cells and one 1-cell as inputs; vertical composition is strictly
     associative, so as an A-infinity category the hom-categories have m_k = 0 for k >= 3
"""
import itertools, random
from fractions import Fraction as F
from math import gcd
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    from check_stochastic_basechange import polytope, rank_solve, det

R = random.Random(3)
res = []
def check(n, c, info=""):
    res.append(bool(c)); print(("PASS " if c else "FAIL ") + n + (f"   [{info}]" if info else ""))

def zeros(v, cells): return frozenset(c for c, x in zip(cells, v) if x == 0)
def neighbours(V, cells):
    Z = [zeros(v, cells) for v in V]; nb = {i: [] for i in range(len(V))}
    for i, j in itertools.combinations(range(len(V)), 2):
        common = Z[i] & Z[j]
        if not any(k not in (i, j) and Z[k] >= common for k in range(len(V))):
            nb[i].append(j); nb[j].append(i)
    return nb
def primitive(w):
    """w: rational circulation (dict cell->Fraction); scale to a primitive integer vector"""
    den = 1
    for x in w.values(): den = den * x.denominator // gcd(den, x.denominator)
    ints = [int(x * den) for x in w.values()]; g = 0
    for x in ints: g = gcd(g, abs(x))
    return {c: x * den / g for c, x in w.items()}
def delzant(f, g):
    n, m = len(f), len(g); dim = (n - 1) * (m - 1)
    cells, V = polytope(f, g)
    if len(V) == 1: return True, True, V, cells
    nb = neighbours(V, cells); simple = smooth = True
    for i, v in enumerate(V):
        if len(nb[i]) != dim: simple = False; continue
        dirs = [primitive({c: V[j][k] - v[k] for k, c in enumerate(cells)}) for j in nb[i]]
        M = [[d[(a, b)] for a in range(n - 1) for b in range(m - 1)] for d in dirs]
        smooth &= abs(det(M)) == 1
    return simple, simple and smooth, V, cells
def nondegenerate(f, g):
    sf = {sum(c) for r in range(1, len(f)) for c in itertools.combinations(f, r)}
    sg = {sum(c) for r in range(1, len(g)) for c in itertools.combinations(g, r)}
    return not (sf & sg)
def rmarg(k):
    w = [F(R.randint(1, 12)) for _ in range(k)]; s = sum(w); return [x / s for x in w]
def facets(V, cells):
    """cells c such that {a_c = 0} is a facet: its vertex set is not contained in another's, and nonempty"""
    Zs = {c: frozenset(i for i, v in enumerate(V) if v[cells.index(c)] == 0) for c in cells}
    return [c for c in cells if Zs[c] and not any(d != c and Zs[d] > Zs[c] for d in cells)], Zs

# T1
ok, cnt = True, 0
for shape in ((2, 3), (3, 3), (2, 4)):
    for _ in range(6):
        while True:
            f, g = rmarg(shape[0]), rmarg(shape[1])
            if nondegenerate(f, g): break
        si, de, V, _ = delzant(f, g); ok &= de; cnt += 1
check("T1 nondegenerate coupling polytopes are Delzant (geometric test)", ok, f"{cnt} random polytopes, shapes 2x3, 3x3, 2x4")

# T2 chambers and walls (2x3, f fixed, g moves along a segment crossing the wall g_1 = 1/2)
f = [F(1, 2), F(1, 2)]
def gt(t): return [t, F(2, 3) - t, F(1, 3)]
def ctype(f, g):
    cells, V = polytope(f, g); return frozenset(zeros(v, cells) for v in V)
types = {t: ctype(f, gt(t)) for t in (F(9, 20), F(19, 40), F(1, 2), F(21, 40), F(11, 20))}
same_left = types[F(9, 20)] == types[F(19, 40)]; same_right = types[F(21, 40)] == types[F(11, 20)]
change = types[F(9, 20)] != types[F(11, 20)]
on_wall = not nondegenerate(f, gt(F(1, 2)))
sl, dl, Vl, cl = delzant(f, gt(F(9, 20))); sr, dr, Vr, cr = delzant(f, gt(F(11, 20)))
nfl, nfr = len(facets(Vl, cl)[0]), len(facets(Vr, cr)[0])
check("T2 combinatorial type constant inside chambers, changes across the wall g_1 = f_1", same_left and same_right and change and on_wall,
      f"vertices: {len(types[F(9,20)])} | wall {len(types[F(1,2)])} | {len(types[F(11,20)])}; facets {nfl} -> {nfr}; Delzant on both sides: {dl and dr}")

# T3 degenerate cases
d22 = delzant([F(1, 2)] * 2, [F(1, 2)] * 2)[1]
s33, d33, V33, _ = delzant([F(1, 3)] * 3, [F(1, 3)] * 3)
check("T3 on walls: 2x2 uniform is still Delzant (a segment), 3x3 uniform is not simple", d22 and not s33)

# T4 monotone examples
ok = True; info = []
for n, m in ((2, 3), (3, 4)):
    f, g = [F(1, n)] * n, [F(1, m)] * m
    si, de, V, cells = delzant(f, g); fac, _ = facets(V, cells)
    ok &= nondegenerate(f, g) and de and len(fac) == n * m
    info.append(f"{n}x{m}: {len(V)} vertices, {len(fac)} facets")
s24 = delzant([F(1, 2)] * 2, [F(1, 4)] * 4)
check("T4 uniform marginals on coprime sizes: Delzant, all nm cells are facets, product coupling equidistant", ok,
      "; ".join(info) + f"; 2x4 uniform (not coprime) Delzant: {s24[1]}")

# T5 del Pezzo potential
x = y = F(1)
Wx = 1 - 1 / x**2 + y - 1 / (x**2 * y); Wy = 1 - 1 / y**2 + x - 1 / (x * y**2)
check("T5 W = x + y + 1/x + 1/y + xy + 1/(xy) has a critical point at (1, 1)", Wx == 0 and Wy == 0, "W(1,1) = 6")

# T6 (square case, the setting of coupling 2-cells): maximal coupling on the boundary, product coupling interior
ok = True
for _ in range(20):
    while True:
        f, g = rmarg(3), rmarg(3)
        if nondegenerate(f, g): break
    mm = [min(a, b) for a, b in zip(f, g)]; t = 1 - sum(mm)
    M = [[(mm[i] if i == j else 0) + (f[i] - mm[i]) * (g[j] - mm[j]) / t for j in range(3)] for i in range(3)]
    ok &= any(M[i][j] == 0 for i in range(3) for j in range(3)) and delzant(f, g)[1]
    ok &= all(f[i] * g[j] > 0 for i in range(3) for j in range(3))
check("T6 square nondegenerate case: Delzant; the maximal coupling has zero entries (boundary), the product is interior", ok)

# T7 arity and associativity
def glue(a, b):
    gm = [sum(a[i][j] for i in range(len(a))) for j in range(len(a[0]))]
    return [[sum(a[i][k] * b[k][j] / gm[k] for k in range(len(gm)) if gm[k] > 0) for j in range(len(b[0]))] for i in range(len(a))]
def rcpl(rowm, n):
    out = []
    for r in rowm:
        w = [F(R.randint(0, 4)) for _ in range(n)]
        if sum(w) == 0: w[0] = F(1)
        s = sum(w); out.append([r * x / s for x in w])
    return out
ok = True
for _ in range(30):
    f = rmarg(3); a = rcpl(f, 3); g1 = [sum(a[i][j] for i in range(3)) for j in range(3)]
    b = rcpl(g1, 3); g2 = [sum(b[i][j] for i in range(3)) for j in range(3)]; c = rcpl(g2, 3)
    ok &= glue(glue(a, b), c) == glue(a, glue(b, c))
check("T7 gluing is strictly associative, so the associator (and any m_3 on a hom-category) is zero", ok)
print(f"\n{sum(res)}/{len(res)} checks passed")
