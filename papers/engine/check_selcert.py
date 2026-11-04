"""Exact certificates for sections 12.9-12.10 of the companion note (module pce/selcomp.py).
  SX1  a full-dimensional cell of associativity (exact): for the margins (2/5, 1/3, 3/7, 1/4) of Theorem 45, at (t1, t2, t3) = (13/60, 7/30, 29/240)
       every max and min of the Frechet-midpoint formula is strict, so the branches are constant on the open polyhedron cut out by the listed affine
       inequalities (all positive at the point), and there A is affine with all four coefficients zero: A = 0 on a set of positive volume.
       The same holds for the fully non-uniform margins (3/20, 1/4, 3/4, 17/20) at (1/40, 1/48, 51/80)
  SX2  every selection used for the composites of Theorem 45 (binary, 11 faces), of the ternary-intermediate example (11 faces) and of the K_3
       example carries an exact certificate: feasibility; the forced zeros and the fibre dimension; a witness of a face of dimension dim - 1 for
       each facet cell, and for each other cell either an exact lower bound x_c >= m > 0 or a certified extra zero; a balanced law q > 0 on each
       level set, with sum q_c e_c in the row space of the margins and earlier fixed cells (which bounds the level and forces the whole level set);
       and strictly increasing levels
  SX3  the ternary configuration, exactly (rational orientation tests): the top composite lies strictly inside the convex hull of the five
       bracketed composites, the Markov composite strictly outside, and two non-adjacent edges of the image of the pentagon cross properly
"""
import os, sys, itertools
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import selcomp as S
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)

# SX1
ok = True; info = []
for ps, t in (([Fr(2, 5), Fr(1, 3), Fr(3, 7), Fr(1, 4)], [Fr(13, 60), Fr(7, 30), Fr(29, 240)]),
              ([Fr(3, 20), Fr(1, 4), Fr(3, 4), Fr(17, 20)], [Fr(1, 40), Fr(1, 48), Fr(51, 80)])):
    A, ineq = S.assoc_cell(ps, t); ok &= all(c == 0 for c in A.c) and all(q.val(t) > 0 for q in ineq)
    print(f"  cell for margins {[str(x) for x in ps]} around {[str(x) for x in t]}: c0 + c1 t1 + c2 t2 + c3 t3 > 0 for (c0, c1, c2, c3) in")
    for q in sorted({tuple(q.c) for q in ineq}): print("    ", tuple(str(x) for x in q))
    ok &= S.associator_bin(ps, t) == 0; info.append(len({tuple(q.c) for q in ineq}))
check("SX1 a full-dimensional cell where A = 0: strict branches at a rational point, A affine with zero coefficients (two margin tuples)", ok,
      f"distinct inequalities {info}")

# SX2
def cpl(p, q, t): return [[t, p - t], [q - t, 1 - p - q + t]]
PB = [Fr(2, 5), Fr(1, 3), Fr(3, 7), Fr(1, 4), Fr(5, 11)]; TB = [Fr(13, 60), Fr(7, 30), Fr(1, 80), Fr(3, 80)]
GB = [cpl(PB[i], PB[i + 1], TB[i]) for i in range(4)]
G3 = [[[Fr(1, 150), Fr(19, 150), Fr(4, 15)], [Fr(4, 25), Fr(31, 150), Fr(7, 30)]],
      [[Fr(7, 50), Fr(2, 75)], [Fr(18, 175), Fr(121, 525)], [Fr(13, 70), Fr(11, 35)]],
      [[Fr(3, 35), Fr(81, 1400), Fr(57, 200)], [Fr(4, 35), Fr(339, 1400), Fr(43, 200)]],
      [[Fr(9, 125), Fr(23, 375), Fr(1, 15)], [Fr(209, 3000), Fr(79, 750), Fr(1, 8)], [Fr(13, 120), Fr(1, 6), Fr(9, 40)]]]
log = []
def certified(sides):
    good, rep = S.certify_leximin(sides); log.append((good, rep)); return S.select(sides)
for chain in (GB, G3, GB[:3]):
    for d in S.dissections(len(chain)): S.composite(chain, d, certified)
check("SX2 every selection behind Theorem 45, the ternary example and the K_3 example has an exact certificate", all(g for g, _ in log) and len(log) > 50,
      f"{sum(g for g, _ in log)}/{len(log)} selections certified")

# SX3
D4 = S.dissections(4); pts = {d: (S.composite(G3, d)[0][0], S.composite(G3, d)[0][1]) for d in D4}
M3 = S.composite(G3, frozenset(), S.markov_rule); mk = (M3[0][0], M3[0][1])
CYC = [frozenset({(0, 2), (0, 3)}), frozenset({(0, 3), (1, 3)}), frozenset({(1, 3), (1, 4)}), frozenset({(1, 4), (2, 4)}), frozenset({(0, 2), (2, 4)})]
def orient(a, b, c): return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
V = [pts[d] for d in CYC]
def hull(P):
    P = sorted(set(P)); lo = []; up = []
    for p in P:
        while len(lo) >= 2 and orient(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and orient(up[-2], up[-1], p) <= 0: up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]
H = hull(V); inside = lambda z: all(orient(H[i], H[(i + 1) % len(H)], z) > 0 for i in range(len(H)))
segs = []
for k in range(5):
    e = CYC[k] & CYC[(k + 1) % 5]; segs += [(k, pts[CYC[k]], pts[e]), (k, pts[e], pts[CYC[(k + 1) % 5]])]
def cross(s, t):
    (_, a, b), (_, c, d) = s, t
    return orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0
crossings = [(s[0], t[0]) for s, t in itertools.combinations(segs, 2) if s[0] != t[0] and (s[0] - t[0]) % 5 not in (1, 4) and cross(s, t)]
ok = inside(pts[frozenset()]) and not inside(mk) and len(crossings) >= 1
check("SX3 ternary configuration (exact orientation tests): top strictly inside the hull of the bracketings, Markov outside, edges of the pentagon cross", ok,
      f"hull has {len(H)} vertices; crossing edge pairs {crossings}")
print(f"{sum(results)}/{len(results)} passed")
