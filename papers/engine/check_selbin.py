"""Checks for section 12.10 of the companion note: selection on binary chains, continuity, and the associativity locus (module pce/selcomp.py).
  SB1  Frechet-midpoint gluing (exact): for binary X0, X2 the selected composite has 00 entry sum_b (max(0, r_b + s_b - q_b) + min(r_b, s_b)) / 2;
       it agrees with the leximin computation on 40 binary chains and 20 chains with a ternary middle, including tied (degenerate) slices
  SB2  uniform margins (exact): with rho = 4 t - 1, selection composes correlations by rho = sign(rho1 rho2) min(|rho1|, |rho2|) and Markov gluing by
       rho1 rho2, on the 21 x 21 grid of rationals; selection is then associative on the 21^3 grid, with unit rho = 1
  SB3  affine on a fixed type (exact): on a 3 x 2 x 3 chain, along 4 random segments of the input, whenever the facet set and the level sets
       agree at two points and their midpoint, the selected composite at the midpoint is the average of the two
  SB4  continuity across facet-set changes (numerical): along the same 4 segments (step 1/200; at least 10 changes of the facet set) the largest change of
       the composite in one step is below 2e-3, and the steps that cross a facet change are no larger than the largest step elsewhere
  SB5  which margins make selection associative: on all 9^4 binary margin tuples with denominator 10, a 5^3 screen of couplings leaves 17
       candidates, and on a 13^3 grid only the uniform tuple (1/2, 1/2, 1/2, 1/2) has A = 0 everywhere
  SB6  the associativity locus has positive measure (exact grids): for six non-uniform margin tuples, A = 0 on between 10% and 75% of a 13^3 grid;
       it is not determined by the signs of the dependences t_i - p_i p_(i+1)
  SB7  one parameter (exact): margins (2/5, 1/3, 3/7, 1/4), t1 = 13/60, t2 = 7/30, t3 in [0, 1/4]: A is piecewise linear with kinks at 1/30, 13/120,
       2/15, 3/20, 143/840, 41/210, 7/30, slopes in {0, +-1/4, 1/2}; A = 0 exactly on [13/120, 2/15] and at 5/28; A(1/80) = -41/960
"""
import os, sys, itertools
from fractions import Fraction as Fr
from collections import Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import selcomp as S
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(2)
def cpl(p, q, t): return [[t, p - t], [q - t, 1 - p - q + t]]
def bnd(a, b): return max(Fr(0), a + b - 1), min(a, b)
def rc(pp, qq, moves=3):
    g = [[pp[i] * qq[j] for j in range(len(qq))] for i in range(len(pp))]
    for _ in range(moves):
        i, k = rng.choice(len(pp), 2, replace=False); j, l = rng.choice(len(qq), 2, replace=False); d = min(g[i][l], g[k][j]) * Fr(int(rng.integers(2, 10)), 10)
        g[i][j] += d; g[k][l] += d; g[i][l] -= d; g[k][j] -= d
    return g

# SB1
bad = 0; ties = 0
for _ in range(40):
    p, q, r = [Fr(int(x), 20) for x in rng.integers(2, 18, 3)]; (l1, h1), (l2, h2) = bnd(p, q), bnd(q, r)
    t1 = l1 + (h1 - l1) * Fr(int(rng.integers(1, 10)), 10); t2 = l2 + (h2 - l2) * Fr(int(rng.integers(1, 10)), 10)
    ties += t1 == t2 or p - t1 == r - t2
    bad += S.select([cpl(p, q, t1), cpl(q, r, t2)])[0][0] != S.u_bin(p, q, r, t1, t2)
for _ in range(20):
    w = [int(x) for x in rng.integers(1, 9, 3)]; qq = tuple(Fr(x, sum(w)) for x in w); p = Fr(int(rng.integers(3, 17)), 20); r = Fr(int(rng.integers(3, 17)), 20)
    g1 = rc((p, 1 - p), qq); g2 = rc(qq, (r, 1 - r)); bad += S.select([g1, g2])[0][0] != S.select_binary(g1, g2)
check("SB1 binary selection is Frechet-midpoint gluing, slice by slice (exact; 60 chains, ternary middles, tied slices)", bad == 0, f"{ties} tied slices among the binary chains")

# SB2
h = Fr(1, 2); grid = [Fr(k, 40) for k in range(0, 21)]; ok = True
rho = lambda t: 4 * t - 1; sgn = lambda x: (x > 0) - (x < 0)
for t1, t2 in itertools.product(grid, repeat=2):
    a, b = rho(t1), rho(t2)
    ok &= rho(S.u_bin(h, h, h, t1, t2)) == sgn(a * b) * min(abs(a), abs(b)) and rho(S.u_markov(h, h, h, t1, t2)) == a * b
ok &= all(S.associator_bin([h] * 4, ts) == 0 for ts in itertools.product(grid, repeat=3)) and S.u_bin(h, h, h, Fr(1, 2), Fr(3, 20)) == Fr(3, 20)
check("SB2 uniform margins: selection composes correlations by sign-times-min (Markov by product); associative on the 21^3 grid; unit rho = 1", ok)

# SB3, SB4
pA = (Fr(1, 5), Fr(3, 10), Fr(1, 2)); pB = (Fr(2, 5), Fr(3, 5)); pC = (Fr(1, 4), Fr(1, 3), Fr(5, 12)); rng = np.random.default_rng(12)
ok3 = True; nmid = 0; maxj = 0; maxj_change = 0; nchange = 0
for trial in range(4):
    ga = rc(pA, pB, 4); gb = rc(pA, pB, 4); g2 = rc(pB, pC, 4); n = 200; prev = None; prevF = None; types = {}; vals = {}
    for k in range(n + 1):
        s = Fr(k, n); g1 = [[(1 - s) * ga[i][j] + s * gb[i][j] for j in range(2)] for i in range(3)]
        F = tuple(S.facet_cells([g1, g2])); _, _, x, lev = S.leximin([g1, g2]); c = S.select([g1, g2])
        types[k] = (F, tuple(S_ for _, S_ in lev)); vals[k] = c
        if prev is not None:
            jmp = max(abs(float(a - b)) for a, b in zip(np.ravel(c), np.ravel(prev)))
            if F != prevF: nchange += 1; maxj_change = max(maxj_change, jmp)
            else: maxj = max(maxj, jmp)
        prev, prevF = c, F
    for k in range(0, n - 1, 2):
        if types[k] == types[k + 1] == types[k + 2]:
            nmid += 1; ok3 &= np.all(vals[k + 1] == (vals[k] + vals[k + 2]) / 2)
check("SB3 on a fixed facet set and fixed level sets, selection is affine in the input (exact midpoint test)", ok3 and nmid > 100, f"{nmid} midpoint tests")
check("SB4 no jump at facet-set changes: largest one-step change < 2e-3, and no larger at a change than elsewhere", nchange >= 10 and max(maxj, maxj_change) < 2e-3 and maxj_change <= maxj,
      f"{nchange} changes; max step {maxj:.2e} elsewhere, {maxj_change:.2e} at changes")

# SB5
def frac0(ps, g):
    n = z = 0
    for ks in itertools.product(range(g + 1), repeat=3):
        ts = []
        for i, k in enumerate(ks): lo, hi = bnd(ps[i], ps[i + 1]); ts.append(lo + (hi - lo) * Fr(k, g))
        n += 1; z += S.associator_bin(ps, ts) == 0
    return Fr(z, n)
cand = [ps for ps in itertools.product([Fr(k, 10) for k in range(1, 10)], repeat=4) if frac0(ps, 4) == 1]
full = [ps for ps in cand if frac0(ps, 12) == 1]
check("SB5 only the uniform margins make selection associative (6561 tuples; 17 survive the screen; 1 survives the 13^3 grid)", len(cand) == 17 and full == [(h, h, h, h)],
      f"{len(cand)} candidates")

# SB6
rng = np.random.default_rng(4); fr = []; bysign_nontrivial = False
for _ in range(6):
    ps = [Fr(int(x), 20) for x in rng.integers(3, 18, 4)]; st = {}
    for ks in itertools.product(range(1, 13), repeat=3):
        ts = []; sg = []
        for i, k in enumerate(ks): lo, hi = bnd(ps[i], ps[i + 1]); t = lo + (hi - lo) * Fr(k, 13); ts.append(t); sg.append(t > ps[i] * ps[i + 1])
        st.setdefault(tuple(sg), []).append(S.associator_bin(ps, ts) == 0)
    allz = [z for v in st.values() for z in v]; fr.append(sum(allz) / len(allz))
    bysign_nontrivial |= any(0 < sum(v) < len(v) for v in st.values())
check("SB6 the associativity locus has positive measure (10%-75% of the grid for six margin tuples) and is not a union of dependence-sign classes",
      all(0.1 <= f <= 0.75 for f in fr) and bysign_nontrivial, f"fractions {[round(f, 2) for f in fr]}")

# SB7
ps = [Fr(2, 5), Fr(1, 3), Fr(3, 7), Fr(1, 4)]; t1, t2 = Fr(13, 60), Fr(7, 30); Af = lambda t: S.associator_bin(ps, [t1, t2, t])
kinks = [Fr(1, 30), Fr(13, 120), Fr(2, 15), Fr(3, 20), Fr(143, 840), Fr(41, 210), Fr(7, 30)]; pts = [Fr(0)] + kinks + [Fr(1, 4)]; ok = True; slopes = []
for a, b in zip(pts, pts[1:]):
    sl = (Af(b) - Af(a)) / (b - a); slopes.append(sl); m = (a + b) / 2; ok &= Af(m) == Af(a) + sl * (m - a)        # affine on each piece
    for q in (Fr(1, 3), Fr(2, 3)): ok &= Af(a + q * (b - a)) == Af(a) + sl * q * (b - a)
ok &= all(s1 != s2 for s1, s2 in zip(slopes, slopes[1:])) and set(slopes) <= {Fr(0), Fr(1, 4), Fr(-1, 4), Fr(1, 2)}
ok &= Af(Fr(13, 120)) == 0 == Af(Fr(2, 15)) and Af(Fr(5, 28)) == 0 and Af(Fr(1, 80)) == Fr(-41, 960)
zeros = [Fr(k, 840) for k in range(0, 211) if Af(Fr(k, 840)) == 0]; ok &= all(Fr(13, 120) <= z <= Fr(2, 15) or z == Fr(5, 28) for z in zeros)
check("SB7 one parameter: A is piecewise linear with 7 exact kinks; A = 0 exactly on [13/120, 2/15] and at 5/28; A(1/80) = -41/960", ok,
      f"slopes {[str(s) for s in slopes]}")
print(f"{sum(results)}/{len(results)} passed")
