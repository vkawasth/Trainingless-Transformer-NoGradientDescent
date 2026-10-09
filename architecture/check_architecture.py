"""check_architecture.py -- every number in Sections 5 and 6 of the architecture note.
  A1  the example f = (0.4, 0.6), g = (0.3, 0.43, 0.27): pentagon, vertices, area, forced cell, wall values, tear and break margins
  A2  band tolerance: an elementwise band of half-width delta meets no wall iff delta < 3/200 (tightened extremes, Bands Prop 3.4)
  A3  product coupling and its entropy; surface, self-intersections, Kaehler coefficients
  A4  Floer: five distinct branes, 3 + 1 + 1, leximin carries 3, product none (unique smallest facet cell)
  A5  real locus: chi = -1, non-orientable, three (-1)-curves
  A6  rigidity: Demazure roots, h^0(T) = chi(T) = 14 - 2r, so h^1 = 0 for the five 2x3 surfaces
  A7  ranks of margins -> Kaehler classes: 1, 2, 2, 3, 3 against dim H^2 = 1, 2, 2, 3, 4; for dP6, c1.[omega] = 1
"""
import subprocess, sys, itertools
from fractions import Fraction as Fr
res = []
def check(n, c, info=""): res.append(bool(c)); print(("PASS " if c else "FAIL ") + n + (f"   [{info}]" if info else ""))
out = subprocess.run([sys.executable, "arch_example.py"], capture_output=True, text=True).stdout
has = lambda *xs: all(x in out for x in xs)
check("A1 fibre and base", has("facet cells ['11', '12', '13', '21', '23'] | forced lower bounds {'22': '3/100'}",
      "vertices 5 [('0', '13/100'), ('0', '2/5'), ('13/100', '0'), ('3/10', '0'), ('3/10', '1/10')]", "area (symplectic volume) 1331/20000",
      "wall values f1 - g(J): ['1/10', '-3/100', '13/100', '-33/100', '-17/100', '-3/10'] | tear margin m_T = 3/100 | break margin = 27/100"))
# A2: band of half-width delta on every coordinate of f and g, tightened; wall function ranges
f = [Fr(2, 5), Fr(3, 5)]; g = [Fr(3, 10), Fr(43, 100), Fr(27, 100)]
def tight(c, d):
    lo = [max(Fr(0), x - d) for x in c]; hi = [x + d for x in c]
    lo2 = [max(lo[i], 1 - sum(hi[j] for j in range(len(c)) if j != i)) for i in range(len(c))]
    hi2 = [min(hi[i], 1 - sum(lo[j] for j in range(len(c)) if j != i)) for i in range(len(c))]
    return lo2, hi2
def safe(d):
    (flo, fhi), (glo, ghi) = tight(f, d), tight(g, d)
    for J in [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]:
        # f1 - g(J): extremes of g(J) over the tightened band are the sums of tightened extremes clipped by the complement
        gl = max(sum(glo[j] for j in J), 1 - sum(ghi[j] for j in range(3) if j not in J))
        gh = min(sum(ghi[j] for j in J), 1 - sum(glo[j] for j in range(3) if j not in J))
        if not (flo[0] - gh > 0 or fhi[0] - gl < 0): return False
    return True
check("A2 band tolerance: safe for delta < 3/200, not at 3/200", safe(Fr(3, 200) - Fr(1, 10**6)) and not safe(Fr(3, 200)) and safe(Fr(1, 100)))
check("A3 entropy and toric data", has("product (max-entropy) coupling (a11, a12) = 3/25 43/250 | H = 1.7506 nats = 2.5256 bits",
      "surface Bl_2 CP^2 | self-intersections {'11': 0, '12': -1, '13': 0, '21': -1, '23': -1}",
      "Kaehler class coefficients f_i g_j on facets: {'11': '3/25', '12': '43/250', '13': '27/250', '21': '9/50', '23': '81/500'}"))
check("A4 Floer: 3 + 1 + 1 distinct branes, leximin 3, product none", has("3 branes over (2/15, 2/15), smallest cells ['11', '12', '13'] = 2/15, level 1",
      "1 branes over (13/100, 13/100), smallest cells ['11', '12', '23'] = 13/100, level 1", "1 branes over (1/5, 1/10), smallest cells ['12', '13', '21'] = 1/10, level 1",
      "leximin ('2/15', '2/15') carries 3 | product carries 0 | smallest facet product ('27/250', '13') unique: True", "distinct critical points at T=e^-40: 5"))
check("A5 real locus", has("chi -1 orientable False Maslov-1 half-discs from 3 (-1)-curves"))
check("A6 rigidity of the five 2x3 surfaces", has("CP^2         n=3: Demazure roots 6, h0(T)=8, chi(T)=14-2n=8, h1(T)=h0-chi=0",
      "CP^1 x CP^1  n=4: Demazure roots 4, h0(T)=6, chi(T)=14-2n=6, h1(T)=h0-chi=0", "F_1          n=4: Demazure roots 4, h0(T)=6, chi(T)=14-2n=6, h1(T)=h0-chi=0",
      "Bl_2         n=5: Demazure roots 2, h0(T)=4, chi(T)=14-2n=4, h1(T)=h0-chi=0", "Bl_3         n=6: Demazure roots 0, h0(T)=2, chi(T)=14-2n=2, h1(T)=h0-chi=0"))
check("A7 margins -> Kaehler classes", has("CP^2         dim H^2 = 1, rank of margins -> [omega] = 1", "CP^1 x CP^1  dim H^2 = 2, rank of margins -> [omega] = 2",
      "F_1          dim H^2 = 2, rank of margins -> [omega] = 2", "Bl_2         dim H^2 = 3, rank of margins -> [omega] = 3", "Bl_3         dim H^2 = 4, rank of margins -> [omega] = 3",
      "dP6: sum of normals (0, 0) | c1.[omega] = 1"))
print(f"\n{sum(res)}/{len(res)} checks passed")
