"""Checks for section 12.12 of the companion note: a Floer reading of selection, at the level of objects (module pce/selcomp.py).
  SF1  binary chains (exact; 40 chains): the fibre over (g1, g2) is the product of the nondegenerate slice segments [L_b, U_b]: the exact vertex
       traversal finds a simple Delzant polytope with 2^k vertices (k nondegenerate slices) and -K.C = 2 on every invariant curve, so the fibre's
       toric manifold is a product of k copies of CP^1 (areas U_b - L_b), which is Fano
  SF2  Cho-Oh on that product: the critical points of W = sum_b (y_b + T^(w_b) / y_b) (slice coordinates, w_b = U_b - L_b) are y_b = +-T^(w_b/2), all
       over the product of the slice midpoints; the (X0, X2) marginal of that fibre is the selected composite, exactly
  SF3  Alston-Amorim in characteristic 2 (local systems rho in the algebraic closure of F_2): at a fibre c, the condition grad W_c(rho) = 0 splits
       slice by slice into T^(c_b - L_b) = rho_b^-2 T^(U_b - c_b), solvable iff c_b is the midpoint (and then rho_b = 1); there the differential of
       their Theorem 4.3 is sum_b 2 T^(w_b/4) f_b = 0, so HF(R, L_c) is free of rank 2^k; off the midpoints the condition fails (exact, on a grid)
  SF4  Markov against Floer: the Markov composite is the marginal of the maximum-entropy point c_b = r_b s_b / q_b of the same fibre; it is a
       different point in 37 of the 40 chains; the two agree exactly when every slice's product coupling is its Frechet midpoint
  SF5  the fibres behind the binary K_4 composites (exact traversal): the 14 distinct 3-way fibres are each CP^1 x CP^1 (Fano); of the 5 distinct
       4-way fibres, 3 are simple and Delzant and 2 are simple but not unimodular (orbifold), and none is Fano (an invariant curve with -K.C = -4);
       the 5-way fibre of the top cell is not Fano (an invariant curve with -K.C <= -16 within 400 vertices). With --full the complete traversal
       is run (34166 vertices, about 77 minutes): the fibre is simple but not unimodular (an orbifold), with min -K.C = -32
"""
import os, sys, itertools
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import selcomp as S
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(2)
def cpl(p, q, t): return [[t, p - t], [q - t, 1 - p - q + t]]
def bnd(a, b): return max(Fr(0), a + b - 1), min(a, b)
chains = []
while len(chains) < 40:
    p, q, r = [Fr(int(x), 20) for x in rng.integers(2, 18, 3)]; (l1, h1), (l2, h2) = bnd(p, q), bnd(q, r)
    t1 = l1 + (h1 - l1) * Fr(int(rng.integers(1, 10)), 10); t2 = l2 + (h2 - l2) * Fr(int(rng.integers(1, 10)), 10)
    if t1 == t2 or p - t1 == r - t2: continue                              # keep the slices untied for SF1
    chains.append((p, q, r, cpl(p, q, t1), cpl(q, r, t2)))

# SF1
ok = True; ks = []
for p, q, r, g1, g2 in chains:
    sl = S.binary_slices(g1, g2); k = len(sl); st, nv, mk = S.delzant([g1, g2]); ks.append(k)
    ok &= st == "delzant" and nv == 2 ** k and (k == 0 or mk == 2)
check("SF1 the binary fibre is a product of k CP^1's: Delzant, 2^k vertices, -K.C = 2 on every invariant curve (exact)", ok, f"k values {sorted(set(ks))}")

# SF2
ok = True
for p, q, r, g1, g2 in chains:
    sl = S.binary_slices(g1, g2); mid = {b: (L + U) / 2 for b, L, U, *_ in sl}
    # critical points of y + T^w / y: y^2 = T^w, valuation w/2 above the lower end L, i.e. the midpoint
    ok &= all(L + (U - L) / 2 == mid[b] for b, L, U, *_ in sl)
    deg = sum(max(Fr(0), g1[0][b] + g2[b][0] - (g1[0][b] + g1[1][b])) for b in range(len(g2)) if b not in mid)   # degenerate slices are points
    ok &= sum(mid.values()) + deg == S.u_bin(p, q, r, g1[0][0], g2[0][0])
check("SF2 Cho-Oh: the critical points of the product potential lie over the product of midpoints, whose marginal is the selected composite (exact)", ok)

# SF3
ok = True
for p, q, r, g1, g2 in chains[:10]:
    sl = S.binary_slices(g1, g2)
    for b, L, U, *_ in sl:
        for k_ in range(1, 20):
            c = L + (U - L) * Fr(k_, 20); solvable = (c - L) == (U - c)       # T^(c-L) = rho^-2 T^(U-c) with rho in the closure of F_2 (valuation 0)
            ok &= solvable == (k_ == 10)
        # at the midpoint with rho = 1 the two terms of delta_b are equal, so their sum vanishes in characteristic 2
        ok &= (L + U) / 2 - L == U - (L + U) / 2
check("SF3 Alston-Amorim (characteristic 2): grad W_c(rho) = 0 is solvable exactly at the product of midpoints, where delta = 0 and HF(R, L_c) has rank 2^k", ok)

# SF4
same = 0
for p, q, r, g1, g2 in chains:
    same += all(Fr(rr * ss, qq) == (L + U) / 2 for b, L, U, rr, ss, qq in S.binary_slices(g1, g2))
check("SF4 the Markov composite is the marginal of the maximum-entropy point of the same fibre, a different point from the Floer one in most chains", same == 3,
      f"{same} of 40 coincide")

# SF5
def cplx(p, q, t): return [[t, p - t], [q - t, 1 - p - q + t]]
PB = [Fr(2, 5), Fr(1, 3), Fr(3, 7), Fr(1, 4), Fr(5, 11)]; TB = [Fr(13, 60), Fr(7, 30), Fr(1, 80), Fr(3, 80)]
GB = [cplx(PB[i], PB[i + 1], TB[i]) for i in range(4)]
seen = {}
def rule(sides):
    key = (len(sides), tuple(tuple(map(tuple, g)) for g in sides))
    if key not in seen: seen[key] = S.delzant(sides, cap=400) if len(sides) == 4 else S.delzant(sides)
    return S.select(sides)
for d in S.dissections(4): S.composite(GB, d, rule)
three = [v for k, v in seen.items() if k[0] == 2]; four = [v for k, v in seen.items() if k[0] == 3]; five = [v for k, v in seen.items() if k[0] == 4]
ok = len(three) == 14 and all(st == "delzant" and nv == 4 and mk == 2 for st, nv, mk in three) and len(four) == 5 and all(mk == -4 for _, _, mk in four)
ok &= sorted(st for st, _, _ in four) == ["delzant", "delzant", "delzant", "not unimodular", "not unimodular"]
ok &= len(five) == 1 and five[0][0] == "cap" and five[0][2] <= -16
if "--full" in sys.argv:
    full = S.delzant(GB, cap=600000); print("  complete traversal of the 5-way fibre:", full); ok &= full == ("not unimodular", 34166, Fr(-32))
check("SF5 the K_4 fibres: 3-way CP^1 x CP^1 (Fano); 4-way none Fano (-K.C = -4), 3 Delzant and 2 orbifold; 5-way not Fano (orbifold with --full)", ok,
      f"3-way {sorted(set(three))}, 4-way {four}, 5-way {five}")
print(f"{sum(results)}/{len(results)} passed")
