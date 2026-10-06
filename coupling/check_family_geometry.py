"""check_family_geometry.py -- the coupling polygons of the 2x3 family as a family over the marginal space.

  G1 within a chamber every vertex is an affine function of the marginals (f1, g1, g2): explicit formulas for the
     hexagon chamber, checked exactly at random points of the chamber
  G2 the product coupling is a global section through the interior of every polygon, at every nondegenerate point
  G3 along the path from (0.4, 0.5, 0.3) (pentagon) to (1/2, 1/3, 1/3) (hexagon), which crosses only the wall
     f1 = g1 at t = 3/8: the new edge {alpha_10 = 0} has length proportional to (t - 3/8) on the hexagon side,
     the area is a different quadratic on each side, continuous and C^1 at the wall with a jump in the second
     derivative, and the Fisher metric at the product coupling is smooth across the wall
"""
import itertools, math, random
from fractions import Fraction as F
import sympy as sp
R = random.Random(8)
res = []
def check(n, c, info=""):
    res.append(bool(c)); print(("PASS " if c else "FAIL ") + n + (f"   [{info}]" if info else ""))

NORMAL = {(0, 0): (1, 0), (0, 1): (0, 1), (0, 2): (-1, -1), (1, 0): (-1, 0), (1, 1): (0, -1), (1, 2): (1, 1)}
CELLS = list(NORMAL)
def alpha(x, y, f1, g1, g2):
    return {(0, 0): x, (0, 1): y, (0, 2): f1 - x - y, (1, 0): g1 - x, (1, 1): g2 - y, (1, 2): x + y - (f1 + g1 + g2 - 1)}
def verts(f1, g1, g2):
    V = set(); k0 = alpha(0, 0, f1, g1, g2)
    for c1, c2 in itertools.combinations(CELLS, 2):
        n1, n2 = NORMAL[c1], NORMAL[c2]; d = n1[0] * n2[1] - n1[1] * n2[0]
        if d == 0: continue
        x = (-k0[c1] * n2[1] + k0[c2] * n1[1]) / d; y = (-n1[0] * k0[c2] + n2[0] * k0[c1]) / d
        if all(v >= 0 for v in alpha(x, y, f1, g1, g2).values()): V.add((x, y))
    return V
def signs(f1, g1, g2):
    f = (f1, 1 - f1); g = (g1, g2, 1 - g1 - g2)
    return tuple((f[i] > g[j]) - (f[i] < g[j]) for i in range(2) for j in range(3))
def area(V):
    cx = sum(v[0] for v in V) / len(V); cy = sum(v[1] for v in V) / len(V)
    P = sorted(V, key=lambda v: math.atan2(float(v[1] - cy), float(v[0] - cx)))
    return abs(sum(P[i][0] * P[(i + 1) % len(P)][1] - P[(i + 1) % len(P)][0] * P[i][1] for i in range(len(P)))) / 2
def edge_len(c, p, V):
    pts = sorted(v for v in V if alpha(v[0], v[1], *p)[c] == 0)
    if len(pts) < 2: return F(0)
    (x0, y0), (x1, y1) = pts[0], pts[-1]; n = NORMAL[c]; a, b = -n[1], n[0]
    return abs((x1 - x0) / a) if a else abs((y1 - y0) / b)

# G1 hexagon chamber: c = f1 + g1 + g2 - 1
hexsig = signs(F(1, 2), F(1, 3), F(1, 3))
def hex_formula(f1, g1, g2):
    c = f1 + g1 + g2 - 1
    return {(F(0), c), (F(0), g2), (f1 - g2, g2), (g1, f1 - g1), (g1, F(0)), (c, F(0))}
ok = True; n = 0
while n < 200:
    p = tuple(F(R.randint(1, 999), 1000) for _ in range(3))
    if p[1] + p[2] >= 1 or signs(*p) != hexsig: continue
    n += 1; ok &= verts(*p) == hex_formula(*p)
check("G1 in the hexagon chamber the vertices are (0,c), (0,g2), (f1-g2,g2), (g1,f1-g1), (g1,0), (c,0) with c = f1+g1+g2-1",
      ok, f"{n} random points of the chamber")

# G2 product coupling is interior at every nondegenerate point
ok = True
for _ in range(2000):
    p = tuple(F(R.randint(1, 999), 1000) for _ in range(3))
    if p[1] + p[2] >= 1: continue
    f = (p[0], 1 - p[0]); g = (p[1], p[2], 1 - p[1] - p[2])
    ok &= all(v > 0 for v in alpha(f[0] * g[0], f[0] * g[1], *p).values())
check("G2 the product coupling lies in the interior of every polygon (a global section of the family)", ok)

# G3 path crossing the wall f1 = g1
t = sp.symbols("t")
P0 = (F(2, 5), F(1, 2), F(3, 10)); P1 = (F(1, 2), F(1, 3), F(1, 3))
def at(tt): return tuple(a + tt * (b - a) for a, b in zip(P0, P1))
walls = []
for i, j in itertools.product(range(2), range(3)):
    fi = lambda q: q[0] if i == 0 else 1 - q[0]; gj = lambda q: (q[1], q[2], 1 - q[1] - q[2])[j]
    a0, a1 = fi(P0) - gj(P0), fi(P1) - gj(P1)
    if a0 * a1 < 0: walls.append(((i, j), a0 / (a0 - a1)))
check("G3a the straight path crosses exactly one wall, f1 = g1, at t = 3/8", walls == [((0, 0), F(3, 8))], str(walls))
rows = []
for tt in (F(0), F(1, 4), F(3, 8) - F(1, 100), F(3, 8), F(3, 8) + F(1, 100), F(1, 2), F(3, 4), F(1)):
    p = at(tt); V = verts(*p); rows.append((tt, len(V), edge_len((1, 0), p, V), area(V)))
for r in rows: print("     t=%-8s vertices=%d  len{alpha_10=0}=%-10s area=%s" % (r[0], r[1], r[2], r[3]))
ok_len = all((r[2] == (at(r[0])[0] - at(r[0])[1]) if r[0] > F(3, 8) else r[2] == 0) for r in rows)
ok_v = [r[1] for r in rows] == [5, 5, 5, 5, 6, 6, 6, 6]
check("G3b the edge {alpha_10 = 0} appears at the wall with length f1 - g1, growing linearly in t; vertices 5 -> 6",
      ok_len and ok_v)
# symbolic area on each side
def sym_area(side):
    tt = F(1, 5) if side == 0 else F(3, 4)
    p = at(tt); V = verts(*p)
    # identify each vertex as an intersection of two lines and rebuild it symbolically in t
    f1s, g1s, g2s = [sp.Rational(a.numerator, a.denominator) + t * sp.Rational((b - a).numerator, (b - a).denominator) for a, b in zip(P0, P1)]
    xs, ys = sp.symbols("x y")
    def al(x, y):
        return {(0, 0): x, (0, 1): y, (0, 2): f1s - x - y, (1, 0): g1s - x, (1, 1): g2s - y, (1, 2): x + y - (f1s + g1s + g2s - 1)}
    k0 = alpha(0, 0, *p); svs = []
    for v in V:
        act = [c for c in CELLS if alpha(v[0], v[1], *p)[c] == 0]
        sol = sp.solve([al(xs, ys)[act[0]], al(xs, ys)[act[1]]], [xs, ys], dict=True)[0]
        svs.append((sol[xs], sol[ys], math.atan2(float(v[1] - sum(w[1] for w in V) / len(V)), float(v[0] - sum(w[0] for w in V) / len(V)))))
    svs.sort(key=lambda z: z[2])
    A = sum(svs[i][0] * svs[(i + 1) % len(svs)][1] - svs[(i + 1) % len(svs)][0] * svs[i][1] for i in range(len(svs))) / 2
    return sp.expand(sp.Abs(A) if False else A)
A0, A1 = sym_area(0), sym_area(1)
w = sp.Rational(3, 8)
vals = [sp.simplify(A0.subs(t, w) - A1.subs(t, w)), sp.simplify(sp.diff(A0, t).subs(t, w) - sp.diff(A1, t).subs(t, w)),
        sp.simplify(sp.diff(A0, t, 2) - sp.diff(A1, t, 2))]
ok_sym = all(abs(float(A0.subs(t, sp.Rational(r[0].numerator, r[0].denominator)))) == float(r[3]) for r in rows if r[0] <= F(3, 8)) and \
         all(abs(float(A1.subs(t, sp.Rational(r[0].numerator, r[0].denominator)))) == float(r[3]) for r in rows if r[0] >= F(3, 8))
check("G3c the area is a different quadratic in t on each side; continuous and C^1 at the wall, with a jump in the second derivative",
      ok_sym and vals[0] == 0 and vals[1] == 0 and vals[2] != 0,
      f"pentagon side {sp.factor(abs(A0))}, hexagon side {sp.factor(abs(A1))}, jump in A'' = {abs(vals[2])}")
# Fisher metric at the product coupling, in the lattice basis (x, y)
def fisher_det(p):
    f = (p[0], 1 - p[0]); g = (p[1], p[2], 1 - p[1] - p[2])
    al = alpha(f[0] * g[0], f[0] * g[1], *p)
    v1 = {(0, 0): 1, (0, 2): -1, (1, 0): -1, (1, 2): 1}; v2 = {(0, 1): 1, (0, 2): -1, (1, 1): -1, (1, 2): 1}
    G = [[sum(a.get(c, 0) * b.get(c, 0) / al[c] for c in CELLS) for b in (v1, v2)] for a in (v1, v2)]
    return G[0][0] * G[1][1] - G[0][1] * G[1][0]
fs = sp.symbols("s")
f1s, g1s, g2s = [sp.Rational(a.numerator, a.denominator) + fs * sp.Rational((b - a).numerator, (b - a).denominator) for a, b in zip(P0, P1)]
fsym = (f1s, 1 - f1s); gsym = (g1s, g2s, 1 - g1s - g2s)
prod = [fsym[i] * gsym[j] for i, j in CELLS]
v1 = [1, 0, -1, -1, 0, 1]; v2 = [0, 1, -1, 0, -1, 1]
Gs = sp.Matrix([[sum(a[k] * b[k] / prod[k] for k in range(6)) for b in (v1, v2)] for a in (v1, v2)])
detG = sp.simplify(Gs.det())
smooth = all(sp.simplify(sp.denom(sp.together(detG)).subs(fs, w)) != 0 for _ in [0])
ok_num = all(abs(float(detG.subs(fs, sp.Rational(r[0].numerator, r[0].denominator))) - float(fisher_det(at(r[0])))) < 1e-12 for r in rows)
check("G3d the Fisher metric at the product coupling is a rational function of t with no pole on [0,1]: smooth across the wall",
      smooth and ok_num and all(sp.denom(sp.together(detG)).subs(fs, sp.Rational(k, 20)) != 0 for k in range(21)),
      f"det G at t = 0, 3/8, 1: {float(detG.subs(fs, 0)):.2f}, {float(detG.subs(fs, w)):.2f}, {float(detG.subs(fs, 1)):.2f}")
print(f"\n{sum(res)}/{len(res)} checks passed")
