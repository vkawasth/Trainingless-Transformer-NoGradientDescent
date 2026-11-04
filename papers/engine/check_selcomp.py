"""Checks for section 12.9 of the companion note: composition by Markov gluing against composition by selection (module pce/selcomp.py).
  AS1  Level 1 (exact rationals): Markov gluing g1 D^-1 g2 is associative and unital on 20 random rational chains; along every face of K_4 it
       gives one composite; the Markov multi-coupling has larger entropy than 200 random points of its fibre (the chain-rule bound)
  AS2  the faces of K_n as dissections of the (n+1)-gon: 3, 11, 45 faces for n = 3, 4, 5, with 2, 5, 14 vertices (Catalan) and one top cell
  AS3  the binary K_4 example (exact): the 11 selected composites (5 bracketings, 5 edges, the top) are distinct rationals; Markov gives the single
       value 2371469/12672000; each edge composite lies strictly between the composites of its two vertices, and the top inside their range
  AS4  the 2 -> 3 state K_4 example (exact): 11 distinct points of the two-dimensional polytope Pi(p0, p4); Markov gives one point; the top
       lies inside the convex hull of the five bracketings
  AS5  selection is unital and commutes with transposition (exact): the identity coupling on either side returns the other coupling, and
       select(g2^T, g1^T) = select(g1, g2)^T
  AS6  non-associativity census: on 30 random rational chains of three couplings (2 or 3 states) the two bracketings differ in at least 24;
       Markov never
  AS7  finite T (T = e^-L, selection = the positive critical point of Theorem 27 on each fibre): on the binary K_3 example the two bracketings
       differ by -41/960 for L = 25, 50, 100, 200 (to 1e-9), so the associator has valuation 0; each binary composite equals its leximin value to
       1e-9, and the top's distance to its leximin value decays faster than 1/L
"""
import os, sys, itertools
from fractions import Fraction as Fr
import numpy as np
import mpmath as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import selcomp as S
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(5)
def rcpl(p, q):
    m, n = len(p), len(q); g = [[p[i] * q[j] for j in range(n)] for i in range(m)]
    for _ in range(3):
        i, k = rng.choice(m, 2, replace=False); j, l = rng.choice(n, 2, replace=False)
        d = min(g[i][l], g[k][j]) * Fr(int(rng.integers(2, 9)), 10); g[i][j] += d; g[k][l] += d; g[i][l] -= d; g[k][j] -= d
    return g
def rmarg(n): w = [int(x) for x in rng.integers(1, 9, n)]; s = sum(w); return tuple(Fr(x, s) for x in w)
def ent(x): return -sum(float(v) * np.log(float(v)) for v in x if v > 0)

# AS1
ok = True
for _ in range(20):
    ps = [rmarg(int(rng.integers(2, 4))) for _ in range(5)]; gs = [rcpl(ps[i], ps[i + 1]) for i in range(4)]
    ok &= np.all(S.markov(S.markov(gs[0], gs[1]), gs[2]) == S.markov(gs[0], S.markov(gs[1], gs[2])))
    I = [[ps[1][i] if i == j else Fr(0) for j in range(len(ps[1]))] for i in range(len(ps[1]))]; ok &= np.all(S.markov(gs[0], I) == np.array(gs[0], dtype=object))
    vals = {tuple(np.ravel(S.composite(gs, d, S.markov_rule))) for d in S.dissections(4)}; ok &= len(vals) == 1
gs3 = gs[:3]; shape, cells, A, b = S.fibre_constraints(gs3); _, mk = S.markov_chain(gs3); Hm = ent(mk.values())
import scipy.linalg as sl
Ns = sl.null_space(np.array(A, float)); x0 = np.array([float(mk[c]) for c in cells]); worse = 0
for _ in range(200):
    d = Ns @ rng.normal(size=Ns.shape[1]); tmax = min([-x0[i] / d[i] for i in range(len(d)) if d[i] < 0] + [1.0]); x = x0 + rng.uniform(0, 0.95) * tmax * d
    worse += ent(x) < Hm
check("AS1 Markov gluing: associative, unital, one composite on every face of K_4; the Markov point has the largest entropy of its fibre", ok and worse == 200)

# AS2
cnt = {n: (len(S.dissections(n)), sum(1 for d in S.dissections(n) if len(d) == n - 2)) for n in (3, 4, 5)}
check("AS2 faces of K_3, K_4, K_5 as dissections: 3, 11, 45 (vertices 2, 5, 14)", cnt == {3: (3, 2), 4: (11, 5), 5: (45, 14)}, f"{cnt}")

# AS3
def cpl(p, q, t): return [[Fr(t), p[0] - Fr(t)], [q[0] - Fr(t), 1 - p[0] - q[0] + Fr(t)]]
PB = [(Fr(2, 5), Fr(3, 5)), (Fr(1, 3), Fr(2, 3)), (Fr(3, 7), Fr(4, 7)), (Fr(1, 4), Fr(3, 4)), (Fr(5, 11), Fr(6, 11))]
TB = [Fr(13, 60), Fr(7, 30), Fr(1, 80), Fr(3, 80)]; GB = [cpl(PB[i], PB[i + 1], TB[i]) for i in range(4)]
D4 = S.dissections(4); VB = {d: S.composite(GB, d)[0][0] for d in D4}
expect = {frozenset({(0, 2), (0, 3)}): Fr(173, 960), frozenset({(0, 2), (2, 4)}): Fr(93, 385), frozenset({(0, 3), (1, 3)}): Fr(61, 384),
          frozenset({(1, 3), (1, 4)}): Fr(61, 320), frozenset({(1, 4), (2, 4)}): Fr(38, 165), frozenset({(0, 2)}): Fr(15403, 73920),
          frozenset({(0, 3)}): Fr(997, 5760), frozenset({(1, 3)}): Fr(197, 1152), frozenset({(1, 4)}): Fr(14941, 73920), frozenset({(2, 4)}): Fr(1249, 5280),
          frozenset(): Fr(28573, 147840)}
ok = VB == expect and len(set(VB.values())) == 11 and {S.composite(GB, d, S.markov_rule)[0][0] for d in D4} == {Fr(2371469, 12672000)}
verts = [d for d in D4 if len(d) == 2]
for e in [d for d in D4 if len(d) == 1]:
    ends = [v for v in verts if e <= v]; lo, hi = sorted(VB[v] for v in ends); ok &= len(ends) == 2 and lo < VB[e] < hi
ok &= min(VB[v] for v in verts) < VB[frozenset()] < max(VB[v] for v in verts)
check("AS3 binary K_4: 11 distinct exact composites; Markov one value; every edge between its vertices; the top inside", ok,
      f"bracketings {[str(VB[v]) for v in verts]}, top {VB[frozenset()]}")

# AS4
P3 = [(Fr(2, 5), Fr(3, 5)), (Fr(1, 6), Fr(1, 3), Fr(1, 2)), (Fr(3, 7), Fr(4, 7)), (Fr(1, 5), Fr(3, 10), Fr(1, 2)), (Fr(1, 4), Fr(1, 3), Fr(5, 12))]
G3 = [[[Fr(1, 150), Fr(19, 150), Fr(4, 15)], [Fr(4, 25), Fr(31, 150), Fr(7, 30)]],
      [[Fr(7, 50), Fr(2, 75)], [Fr(18, 175), Fr(121, 525)], [Fr(13, 70), Fr(11, 35)]],
      [[Fr(3, 35), Fr(81, 1400), Fr(57, 200)], [Fr(4, 35), Fr(339, 1400), Fr(43, 200)]],
      [[Fr(9, 125), Fr(23, 375), Fr(1, 15)], [Fr(209, 3000), Fr(79, 750), Fr(1, 8)], [Fr(13, 120), Fr(1, 6), Fr(9, 40)]]]
ok = all(sum(r) == P3[i][k] for i, g in enumerate(G3) for k, r in enumerate(g)) and all(sum(g[k][j] for k in range(len(g))) == P3[i + 1][j] for i, g in enumerate(G3) for j in range(len(g[0])))
V3 = {d: S.composite(G3, d) for d in D4}; key = lambda c: (c[0][0], c[0][1]); pts = {d: key(c) for d, c in V3.items()}
ok &= len(set(pts.values())) == 11 and len({key(S.composite(G3, d, S.markov_rule)) for d in D4}) == 1
from scipy.spatial import Delaunay
hull = Delaunay(np.array([[float(x) for x in pts[v]] for v in verts])); ok &= hull.find_simplex(np.array([float(x) for x in pts[frozenset()]])) >= 0
check("AS4 2 -> 3 states K_4: 11 distinct exact points in the 2-d endpoint polytope; Markov one point; the top inside the pentagon of bracketings", ok,
      f"top {[str(x) for x in pts[frozenset()]]}")

# AS5
ok = True
for _ in range(8):
    p, q, r = rmarg(2), rmarg(3), rmarg(2); g1 = rcpl(p, q); g2 = rcpl(q, r)
    Ip = [[p[i] if i == j else Fr(0) for j in range(2)] for i in range(2)]; Iq = [[q[i] if i == j else Fr(0) for j in range(3)] for i in range(3)]
    ok &= np.all(S.select([Ip, g1]) == np.array(g1, dtype=object)) and np.all(S.select([g1, Iq]) == np.array(g1, dtype=object))
    T = lambda g: [list(c) for c in zip(*g)]
    ok &= np.all(S.select([T(g2), T(g1)]) == S.select([g1, g2]).T)
check("AS5 selection is unital and commutes with transposition (exact)", ok)

# AS6
diff = 0; mk_diff = 0
for t in range(30):
    n = 2 + t % 2; ps = [rmarg(n) for _ in range(4)]; gs = [rcpl(ps[i], ps[i + 1]) for i in range(3)]
    a = S.select([S.select(gs[:2]), gs[2]]); b_ = S.select([gs[0], S.select(gs[1:])]); diff += not np.all(a == b_)
    mk_diff += not np.all(S.markov(S.markov(gs[0], gs[1]), gs[2]) == S.markov(gs[0], S.markov(gs[1], gs[2])))
check("AS6 selection composition is not associative on most random chains (>= 24 of 30); Markov never", diff >= 24 and mk_diff == 0, f"{diff}/30 differ")

# AS7
G = GB[:3]; D3 = S.dissections(3); lex = {d: S.composite(G, d)[0][0] for d in D3}; ok = lex[frozenset({(0, 2)})] - lex[frozenset({(1, 3)})] == Fr(-41, 960); info = []
for L in (25.0, 50.0, 100.0, 200.0):
    def rL(sides, L=L): return S.softmin(sides, L)
    v = {d: S.composite(G, d, rL)[0][0] for d in D3}; lx = {d: mp.mpf(lex[d].numerator) / lex[d].denominator for d in D3}
    assoc = v[frozenset({(0, 2)})] - v[frozenset({(1, 3)})]; ok &= abs(assoc + mp.mpf(41) / 960) < 1e-9
    ok &= all(abs(v[d] - lx[d]) < 1e-9 for d in D3 if d); info.append((L, float(L * abs(v[frozenset()] - lx[frozenset()]))))
ok &= info[-1][1] < info[0][1] / 10
check("AS7 finite T: the associator is -41/960 at every L (valuation 0); binary composites equal their leximin values; the top converges faster than 1/L", ok,
      f"L * |top - lex| {[(L, round(x, 6)) for L, x in info]}")
print(f"{sum(results)}/{len(results)} passed")
