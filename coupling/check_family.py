"""check_family.py -- the family of coupling polytopes over the space of marginals (2x3 case).

  C1 within a chamber the vertices are affine functions of the marginals (one formula per feasible spanning tree)
  C2 the product coupling is a polynomial section, interior to every fibre; edge lengths are affine in a chamber
  C3 along the path from (0.4, 0.5, 0.3) to (1/2, 1/3, 1/3), which crosses the single wall f1 = g1 at t = 3/8:
     the area (Duistermaat-Heckman volume) is piecewise quadratic, continuous and C^1 at the wall, with a jump in
     the second derivative; the new edge grows linearly from length 0
  C4 the Fisher metric at the product coupling is a rational function of t, smooth through the wall:
     information geometry at the product coupling does not see the walls; toric geometry does
"""
import itertools, math
from fractions import Fraction as F
res = []
def check(n, c, info=""):
    res.append(bool(c)); print(("PASS " if c else "FAIL ") + n + (f"   [{info}]" if info else ""))

NORMAL = {(0, 0): (1, 0), (0, 1): (0, 1), (0, 2): (-1, -1), (1, 0): (-1, 0), (1, 1): (0, -1), (1, 2): (1, 1)}
CELLS = list(NORMAL)
def alpha(x, y, f1, g1, g2):
    return {(0, 0): x, (0, 1): y, (0, 2): f1 - x - y, (1, 0): g1 - x, (1, 1): g2 - y, (1, 2): x + y - (f1 + g1 + g2 - 1)}
def tagged_vertices(p):
    """vertices with the pair of tight cells that defines them (the complement of a spanning tree)"""
    k0 = alpha(F(0), F(0), *p); out = {}
    for c1, c2 in itertools.combinations(CELLS, 2):
        n1, n2 = NORMAL[c1], NORMAL[c2]; d = n1[0] * n2[1] - n1[1] * n2[0]
        if d == 0: continue
        x = (-k0[c1] * n2[1] + k0[c2] * n1[1]) / d; y = (-n1[0] * k0[c2] + n2[0] * k0[c1]) / d
        if all(v >= 0 for v in alpha(x, y, *p).values()): out[(c1, c2)] = (x, y)
    return out
def area(V):
    V = list(V); cx = sum(v[0] for v in V) / len(V); cy = sum(v[1] for v in V) / len(V)
    P = sorted(V, key=lambda v: math.atan2(float(v[1] - cy), float(v[0] - cx)))
    return abs(sum(P[i][0] * P[(i + 1) % len(P)][1] - P[(i + 1) % len(P)][0] * P[i][1] for i in range(len(P)))) / 2
def signs(f1, g1, g2):
    f = (f1, 1 - f1); g = (g1, g2, 1 - g1 - g2)
    return tuple((f[i] > g[j]) - (f[i] < g[j]) for i in range(2) for j in range(3))
def lerp(p, q, t): return tuple(a + t * (b - a) for a, b in zip(p, q))

# C1, C2: affine dependence inside a chamber
import random
R = random.Random(3); ok1 = ok2 = True; tested = 0
for _ in range(400):
    p = tuple(F(R.randint(1, 99), 100) for _ in range(3)); q = tuple(x + F(R.randint(-20, 20), 2000) for x in p)
    if p[1] + p[2] >= 1 or q[1] + q[2] >= 1 or signs(*p) != signs(*q) or 0 in signs(*p): continue
    tested += 1; t = F(R.randint(1, 9), 10); m = lerp(p, q, t)
    if signs(*m) != signs(*p): continue
    Vp, Vq, Vm = tagged_vertices(p), tagged_vertices(q), tagged_vertices(m)
    ok1 &= set(Vp) == set(Vq) == set(Vm) and all(Vm[k] == lerp(Vp[k], Vq[k], t) for k in Vm)
    for pt in (p, q, m):
        f = (pt[0], 1 - pt[0]); g = (pt[1], pt[2], 1 - pt[1] - pt[2])
        ok2 &= all(f[i] * g[j] > 0 for i in range(2) for j in range(3))
check("C1 inside a chamber each vertex is an affine function of (f1, g1, g2), indexed by its tree", ok1 and tested > 20,
      f"{tested} pairs of points in common chambers")
check("C2 the product coupling is a polynomial section in the interior of every fibre", ok2)

# C3 the path across the wall f1 = g1
p0, p1 = (F(2, 5), F(1, 2), F(3, 10)), (F(1, 2), F(1, 3), F(1, 3))
tw = F(3, 8)
walls = [t for t in (F(k, 800) for k in range(1, 800)) if 0 in signs(*lerp(p0, p1, t))]
A = lambda t: area(tagged_vertices(lerp(p0, p1, t)).values())
def quad(ts):
    (a, fa), (b, fb), (c, fc) = [(t, A(t)) for t in ts]
    # second divided difference = leading coefficient
    return ((fc - fb) / (c - b) - (fb - fa) / (b - a)) / (c - a)
left = [quad((F(1, 10), F(2, 10), F(3, 10))), quad((F(1, 20), F(1, 5), F(7, 20)))]
right = [quad((F(5, 10), F(6, 10), F(7, 10))), quad((F(9, 20), F(7, 10), F(19, 20)))]
h = F(1, 10**6)
dl = (A(tw) - A(tw - h)) / h; dr = (A(tw + h) - A(tw)) / h
check("C3 the path crosses exactly one wall, f1 = g1, at t = 3/8; the area is quadratic on each side with different "
      "leading coefficients, continuous and C^1 at the wall",
      walls == [tw] and left[0] == left[1] and right[0] == right[1] and left[0] != right[0] and abs(dl - dr) < F(1, 10**4),
      f"leading coefficients {left[0]} | {right[0]}; one-sided slopes {float(dl):.6f}, {float(dr):.6f}")
lens = []
for t in (tw + F(1, 100), tw + F(2, 100), tw + F(4, 100)):
    V = tagged_vertices(lerp(p0, p1, t)); f1, g1, g2 = lerp(p0, p1, t)
    pts = sorted(v for v in V.values() if alpha(v[0], v[1], f1, g1, g2)[(1, 0)] == 0)
    lens.append(abs(pts[-1][1] - pts[0][1]))
check("C3' the edge {alpha_10 = 0} appears at the wall and grows linearly", lens[1] == 2 * lens[0] and lens[2] == 4 * lens[0],
      f"lengths {[str(l) for l in lens]}")

# C4 Fisher metric at the product coupling along the path, on the circulation basis e1, e2
def fisher(t):
    f1, g1, g2 = lerp(p0, p1, t); f = (f1, 1 - f1); g = (g1, g2, 1 - g1 - g2)
    E = [{(0, 0): 1, (0, 2): -1, (1, 0): -1, (1, 2): 1}, {(0, 1): 1, (0, 2): -1, (1, 1): -1, (1, 2): 1}]
    return [[sum(F(a.get(c, 0) * b.get(c, 0)) / (f[c[0]] * g[c[1]]) for c in CELLS) for b in E] for a in E]
Gm, G0, Gp = fisher(tw - h), fisher(tw), fisher(tw + h)
second = [[(Gp[i][j] - 2 * G0[i][j] + Gm[i][j]) / h**2 for j in range(2)] for i in range(2)]
second2 = [[(fisher(tw + 2 * h)[i][j] - 2 * G0[i][j] + fisher(tw - 2 * h)[i][j]) / (4 * h**2) for j in range(2)] for i in range(2)]
check("C4 the Fisher metric at the product coupling is smooth through the wall (no jump in any derivative tested)",
      all(abs(second[i][j] - second2[i][j]) < F(1, 10**3) for i in range(2) for j in range(2)),
      f"G(3/8) = {[[str(x) for x in r] for r in G0]}")
print(f"\n{sum(res)}/{len(res)} checks passed")
