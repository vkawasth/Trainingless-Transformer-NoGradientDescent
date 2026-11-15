"""The 3 x 3 test, continued (section sec:threebythree). New script; it modifies nothing; uses pce/cpl33.py, pce/fsgram.py.
Declared before the first run:
  TT8 IDENTITY  on the 600 searched margins of check_threebythree.py (seed 20261122), the facet-leximin point equals the all-cell leximin
                point exactly (both certified): cor:leximin in four dimensions
  TT9 IDENTITY  facets are exactly the cells with f_i + g_j <= 1 (nondegenerate margins), on the 600 searched margins (exact)
  TT10 RECORD   a larger census: 10000 margins (seed 20261124, the same distribution), facets by the rule of TT9, vertex sets in floating
                point with tolerance 1e-9 (the margins are nondegenerate rationals with denominator 120): normal-fan types up to symmetry,
                how many beyond the 13 of the 2000-margin sample, Fano types, and vertex counts against De Loera-Kim-Onn-Santos
                (9, 12, 15, 18)
  TT11 IDENTITY the Fano chamber is P^2 x P^2: its six normals split into two triples, each summing to 0 and spanning complementary
                planes; Kouchnirenko's count 4! vol(conv normals) = 9 = number of vertices = rank K_0; the product (Beilinson) collection
                O(a, b), 0 <= a, b <= 2, is exceptional with Gram matrix the Kronecker square of the P^2 Gram matrix; mutations preserve
                its isometry invariants
"""
import os, sys, math, itertools
from collections import Counter
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import cpl33 as C, fsgram as FS
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
def nondegenerate(f, g):
    fs = {sum(c) for k in (1, 2) for c in itertools.combinations(f, k)}; gs = {sum(c) for k in (1, 2) for c in itertools.combinations(g, k)}
    return not (fs & gs)
def rmarg(rng, den=120):
    v = rng.dirichlet([2, 2, 2]) * den; k = np.maximum(1, np.floor(v)).astype(int); k[int(np.argmax(k))] += den - k.sum()
    return [Fr(int(x), den) for x in k] if min(k) >= 1 else None
rng = np.random.default_rng(20261122); n = 0; diff = 0; rule_bad = 0
while n < 600:
    f = rmarg(rng); g = rmarg(rng)
    if f is None or g is None or not nondegenerate(f, g): continue
    n += 1; F = C.exact_facets(f, g); rule = [c for c in C.CELLS if f[c[0]] + g[c[1]] <= 1]; rule_bad += sorted(F) != sorted(rule)
    _, xf = C.exact_leximin(f, g, F); _, xa = C.exact_leximin(f, g, C.CELLS); diff += xf != xa
check("TT8 IDENTITY facet-leximin equals all-cell leximin exactly on 600 nondegenerate 3x3 margins", diff == 0, f"{n} margins; disagreements {diff}")
check("TT9 IDENTITY the facets are exactly the cells with f_i + g_j <= 1", rule_bad == 0, f"{n} margins; mismatches {rule_bad}")
N = C.normals(); NV = {c: np.array(N[c], float) for c in C.CELLS}
def fvertex_sets(f, g, facets):
    x0 = C.base_point(f, g); out = set()
    for Sq in itertools.combinations(facets, 4):
        M = np.array([NV[c] for c in Sq])
        if abs(np.linalg.det(M)) < 1e-9: continue
        u = np.linalg.solve(M, [-float(x0[c]) for c in Sq])
        if all(NV[c] @ u >= -float(x0[c]) - 1e-9 for c in facets): out.add(frozenset(Sq))
    return out
def ffano(facets, V):
    out = set()
    for Sq in itertools.combinations(facets, 4):
        M = np.array([NV[c] for c in Sq])
        if abs(np.linalg.det(M)) < 1e-9: continue
        u = np.linalg.solve(M, [-1.0] * 4)
        if all(NV[c] @ u >= -1 - 1e-9 for c in facets): out.add(frozenset(Sq))
    return out == V
rng = np.random.default_rng(20261124); m = 0; types = Counter(); info = {}; first_seen = {}; vcounts = Counter()
while m < 10000:
    f = rmarg(rng); g = rmarg(rng)
    if f is None or g is None or not nondegenerate(f, g): continue
    m += 1; F = [c for c in C.CELLS if f[c[0]] + g[c[1]] <= 1]; V = fvertex_sets(f, g, F); key = C.canon_fan(V); types[key] += 1; vcounts[len(V)] += 1
    if key not in info: info[key] = (len(F), len(V), ffano(F, V)); first_seen[key] = m
late = sum(1 for k in types if first_seen[k] > 2000)
check("TT10 RECORD census of 10000 margins: normal-fan types, Fano types, vertex counts", True,
      f"{len(types)} fan types (first seen after margin 2000: {late}); by facets " + str(dict(Counter(v[0] for v in info.values())))
      + f"; Fano types {sum(1 for v in info.values() if v[2])}; vertex counts {dict(sorted(vcounts.items()))} (De Loera et al.: 9, 12, 15, 18)")
# TT11 the Fano chamber
fano_keys = [k for k, v in info.items() if v[2]]; ok = len(fano_keys) == 1
from scipy.spatial import ConvexHull
ex = None; rng = np.random.default_rng(5)
while ex is None:
    f = rmarg(rng); g = rmarg(rng)
    if f is None or g is None or not nondegenerate(f, g): continue
    F = [c for c in C.CELLS if f[c[0]] + g[c[1]] <= 1]
    if len(F) == 6: ex = (f, g, F)
f, g, F = ex; vecs = [NV[c] for c in F]; split = None
for T3 in itertools.combinations(range(6), 3):
    A = [vecs[i] for i in T3]; B = [vecs[i] for i in range(6) if i not in T3]
    if np.allclose(sum(A), 0) and np.allclose(sum(B), 0) and np.linalg.matrix_rank(np.array(A + B)) == 4 and np.linalg.matrix_rank(np.array(A)) == 2: split = (T3,); break
ok &= split is not None
vol = ConvexHull(np.array(vecs)).volume; kouch = round(24 * vol); nv = len(fvertex_sets(f, g, F)); ok &= kouch == 9 and nv == 9
P2 = np.array([[1, 3, 6], [0, 1, 3], [0, 0, 1]]); chi = lambda d: (d + 1) * (d + 2) // 2
objs = [(a, b) for a in range(3) for b in range(3)]; G = np.array([[chi(a2 - a) * chi(b2 - b) for (a2, b2) in objs] for (a, b) in objs])
ok &= np.array_equal(G, np.kron(P2, P2)) and np.array_equal(np.tril(G, -1), 0 * G) and all(G[i, i] == 1 for i in range(9))
inv0 = FS.invariants(G); X = G; rngm = np.random.default_rng(7); inv_bad = 0
for step in range(30):
    X = FS.mutate(X, int(rngm.integers(8)))
    if step % 10 == 9: inv_bad += FS.invariants(X) != inv0
ok &= inv_bad == 0
check("TT11 IDENTITY the Fano chamber is P2 x P2: split normals, Kouchnirenko 9 = vertices, product collection with Gram = Kronecker square, mutation invariants",
      ok, f"Fano fan types {len(fano_keys)}; example f={tuple(str(x) for x in f)}, g={tuple(str(x) for x in g)}; 4! vol = {24 * vol:.3f}; vertices {nv}; "
      + f"Smith form of G + G^T {inv0['snf_sym']}; invariant changes under 30 mutations {inv_bad}")
print(f"{sum(results)}/{len(results)} passed")
