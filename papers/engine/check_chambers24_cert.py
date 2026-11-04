"""Independent verifier for the certificate kahler/chambers24.json (Theorem 41 of the companion note). It reads only that file and uses only
exact rational arithmetic from the standard library: it imports nothing from pce/ and recomputes everything it checks.
  CC1  148 records with pairwise distinct sign vectors over the 14 nonempty proper column sets J; each record's rational point is strictly inside
       the margin simplex and has exactly the stated signs of f1 - g(J) (so the 148 chambers are realised and distinct)
  CC2  from the point alone: the vertices of the fibre (all triples of the eight cell functions, solved exactly, kept if feasible) and their tight
       cells agree with the record; the facets (cells vanishing on three affinely independent vertices) agree; every vertex has exactly three
       tight facets and |det| = 1 (simple and smooth)
  CC3  on every edge (two vertices sharing two facets): the stated integers a, b satisfy v_k + v_l + a v_i + b v_j = 0, -K.C = 2 + a + b, and
       the Kahler degree -lam_k - lam_l - a lam_i - b lam_j equals the lattice length of the edge, both recomputed; the number of edges is 3V/2
  CC4  the class: Fano iff every -K.C > 0, weak Fano iff all >= 0 with some = 0 (toric Kleiman); the counts are 2 + 8 + 4 Fano and 24 + 64 + 46
       weak Fano by facet number, with 0 / 1 / 3 / 6 curves of -K.C = 0 as in the theorem
"""
import json, os, itertools, math
from fractions import Fraction as Fr
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); D = json.load(open(os.path.join(HERE, "..", "kahler", "chambers24.json")))["chambers"]
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
CELLS = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
NORM = {(0, 0): (1, 0, 0), (0, 1): (0, 1, 0), (0, 2): (0, 0, 1), (0, 3): (-1, -1, -1), (1, 0): (-1, 0, 0), (1, 1): (0, -1, 0), (1, 2): (0, 0, -1), (1, 3): (1, 1, 1)}
def lam(f1, g):
    return {(0, 0): Fr(0), (0, 1): Fr(0), (0, 2): Fr(0), (0, 3): -f1, (1, 0): -g[0], (1, 1): -g[1], (1, 2): -g[2], (1, 3): f1 + g[0] + g[1] + g[2] - 1}
def det3(M): return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
def solve(A, b):
    d = det3(A)
    if d == 0: return None
    out = []
    for i in range(3):
        M = [list(r) for r in A]
        for r in range(3): M[r][i] = b[r]
        out.append(Fr(det3(M)) / d)
    return out
def dot(v, u): return sum(Fr(a) * b for a, b in zip(v, u))
def rank(vecs):
    M = [list(map(Fr, v)) for v in vecs]; r = 0
    for c in range(3):
        p = next((i for i in range(r, len(M)) if M[i][c] != 0), None)
        if p is None: continue
        M[r], M[p] = M[p], M[r]
        for i in range(len(M)):
            if i != r and M[i][c] != 0: fac = M[i][c] / M[r][c]; M[i] = [a - fac * b for a, b in zip(M[i], M[r])]
        r += 1
    return r
tup = lambda x: tuple(x)

# CC1
sv = set(); ok = len(D) == 148
for R in D:
    f = [Fr(x) for x in R["f"]]; g = [Fr(x) for x in R["g"]]
    ok &= f[0] + f[1] == 1 and sum(g) == 1 and 0 < f[0] < 1 and min(g) > 0
    for J, s in R["signs"]: ok &= s * (f[0] - sum(g[j] for j in J)) > 0
    sv.add(tuple(s for _, s in R["signs"]))
ok &= len(sv) == 148 and all(len(R["signs"]) == 14 for R in D)
check("CC1 148 distinct sign vectors, each realised by its exact interior point", ok, f"{len(sv)} distinct")

# CC2 and CC3
ok2 = ok3 = True; nedges = 0; minusK = []
for R in D:
    f1 = Fr(R["f"][0]); g = [Fr(x) for x in R["g"]]; L = lam(f1, g); verts = {}
    for S in itertools.combinations(CELLS, 3):
        x = solve([NORM[c] for c in S], [L[c] for c in S])
        if x is None: continue
        vals = {c: dot(NORM[c], x) - L[c] for c in CELLS}
        if min(vals.values()) < 0: continue
        verts[tuple(x)] = frozenset(c for c in CELLS if vals[c] == 0)
    facets = [c for c in CELLS if sum(1 for t in verts.values() if c in t) >= 3 and
              rank([[a - b for a, b in zip(x, next(iter(xx for xx, t in verts.items() if c in t)))] for x, t in verts.items() if c in t]) == 2]
    ok2 &= sorted(map(tuple, R["facets"])) == sorted(facets)
    rv = {tuple(Fr(c) for c in K["vertex"]): (frozenset(map(tuple, K["facets"])), K["det"]) for K in R["cones"]}
    ok2 &= set(rv) == set(verts)
    for x, t in verts.items():
        tf = frozenset(c for c in t if c in facets); ok2 &= len(tf) == 3 and rv.get(x, (None,))[0] == tf
        d = det3([NORM[c] for c in sorted(tf)]); ok2 &= abs(d) == 1 and abs(rv[x][1]) == 1
    E = {}
    for (x, s), (y, t) in itertools.combinations([(x, frozenset(c for c in t if c in facets)) for x, t in verts.items()], 2):
        com = s & t
        if len(com) == 2: E[frozenset(com)] = (x, y, next(iter(s - com)), next(iter(t - com)))
    ok3 &= len(E) == 3 * len(verts) // 2 == len(R["curves"]); nedges += len(E)
    for C in R["curves"]:
        on = frozenset(map(tuple, C["on"])); x, y, k, l = E[on]; i, j = map(tuple, C["on"]); a, b = C["a"], C["b"]
        ok3 &= {k, l} == set(map(tuple, C["ends"]))
        ok3 &= all(NORM[k][r] + NORM[l][r] + a * NORM[i][r] + b * NORM[j][r] == 0 for r in range(3)) and C["minusK"] == 2 + a + b
        om = -L[k] - L[l] - a * L[i] - b * L[j]; dvec = [q - p for p, q in zip(x, y)]
        den = 1
        for c in dvec: den = den * c.denominator // math.gcd(den, c.denominator)
        gg = 0
        for c in dvec: gg = math.gcd(gg, abs(int(c * den)))
        ok3 &= om == Fr(gg, den) == Fr(C["kahler_degree"]) == Fr(C["lattice_length"])
check("CC2 vertices, tight sets and facets recomputed from each point agree; simple and smooth (|det| = 1)", ok2)
check("CC3 wall relations, -K.C = 2 + a + b, Kahler degree = lattice length on every edge (recomputed)", ok3, f"{nedges} edges")

# CC4
cnt = Counter(); ok4 = True
for R in D:
    ks = [C["minusK"] for C in R["curves"]]; cls = "Fano" if min(ks) > 0 else ("weak Fano" if min(ks) == 0 else "not nef")
    ok4 &= cls == R["class"]; cnt[(cls, len(R["facets"]), sum(k == 0 for k in ks))] += 1
ok4 &= dict(cnt) == {("Fano", 4, 0): 2, ("Fano", 5, 0): 8, ("Fano", 6, 0): 4, ("weak Fano", 6, 1): 24, ("weak Fano", 7, 3): 64, ("weak Fano", 8, 6): 46}
check("CC4 toric Kleiman: 14 Fano (4, 5, 6 facets: 2, 8, 4) and 134 weak Fano (6, 7, 8 facets: 24, 64, 46; 1, 3, 6 curves of -K.C = 0)", ok4, f"{dict(cnt)}")
print(f"{sum(results)}/{len(results)} passed")
