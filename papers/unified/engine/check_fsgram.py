"""Checks for the categorical layer per chamber (prop:fsgram, section sec:fsgram). New script; it modifies nothing; uses pce/fsgram.py.
  FG1 IDENTITY  census of 2 x 3 coupling polygons (4000 random margins): every facet set gives a smooth complete fan whose rays are the
                surface is Fano (D_i^2 >= -1 on every invariant curve: a toric del Pezzo surface); the number of facets equals Kouchnirenko's count 2 Area(conv
                rays), the number of critical points of W; five surface types occur (P^2, P^1 x P^1, F_1, the blow-ups of P^2 at 2 and
                at 3 points)
  FG2 IDENTITY  the cohomology routine: h^0 - h^1 + h^2 equals Riemann-Roch on 300 random line bundles over the five surfaces, and
                h^0(O(d)) = C(d+2, 2) on P^2
  FG3 IDENTITY  the toric collection O, O(D_1), ..., O(D_1 + ... + D_{m-1}) is exceptional on each surface (all Ext between later and
                earlier objects vanish; End = k); its length is the rank of K_0, so it is full (Kuleshov-Orlov). Its Gram matrix is
                unipotent upper-triangular with every off-diagonal entry positive
  FG4 IDENTITY  mutations: 40 random left mutations per surface (invariants every 10; entries grow like Markov numbers, so longer walks only cost time) keep the Gram matrix integral and unipotent upper-triangular, satisfy
                the braid relations, and preserve the isometry invariants (Coxeter characteristic polynomial, Smith normal forms of
                G + G^T and G - G^T)
  (first run: FG1-FG3 passed; FG4 at 2000, then 300, mutations did not finish in 30 minutes because entries grow without bound; reduced)
  FG5 IDENTITY  the invariants separate the five types; in particular F_1 and P^1 x P^1 (both rank 4, K^2 = 8) have Smith forms of
                G + G^T equal to (1, 1, 4, 4) and (2, 2, 2, 2)
  FG6 IDENTITY  the facet type changes along a margin path only where three constraint lines meet at a vertex (a support wall, codimension
                one in the margins); F_1 keeps its facet set for every alpha in (0, 1), so the outcome wall alpha = 1/3 is not a support wall
"""
import os, sys, math, itertools
import numpy as np
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import fsgram as F
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(20261116)
def area(pts):
    pts = sorted(pts, key=lambda v: math.atan2(v[1], v[0])); return 0.5 * abs(sum(pts[i][0] * pts[i - 1][1] - pts[i - 1][0] * pts[i][1] for i in range(len(pts))))
def is_fano(S):
    """-K is ample iff -K . D_i = 2 + D_i^2 > 0 for every invariant curve (toric Nakai criterion), i.e. D_i^2 >= -1."""
    return all(int(S["I"][i, i]) >= -1 for i in range(S["m"]))
def name(S):
    if S["m"] == 4: return "P1xP1" if all(int(x) == 0 for x in np.diag(S["I"])) else "F1"
    return {3: "P2", 5: "dP7", 6: "dP6"}[S["m"]]
cnt = Counter(); bad = 0; surfaces = {}
for _ in range(4000):
    f = rng.dirichlet([1, 1]); g = rng.dirichlet([1, 1, 1]) if rng.random() < 0.6 else rng.dirichlet([0.3, 0.3, 0.3])
    fac = F.cpl23_facets(f, g); rays = [F.NORMALS[c] for c in fac]; S = F.surface(rays)
    ok = S["smooth"] and is_fano(S) and abs(2 * area(rays) - len(rays)) < 1e-9
    bad += not ok; nm = name(S); cnt[nm] += 1; surfaces.setdefault(nm, S)
check("FG1 IDENTITY every 2x3 coupling polygon is a smooth toric del Pezzo surface; facets = Kouchnirenko count of critical points", bad == 0 and len(cnt) == 5,
      f"4000 margins; violations {bad}; types {dict(cnt)}")
bad = 0
for nm, S in surfaces.items():
    for _ in range(60):
        a = rng.integers(-3, 4, size=S["m"]); h = F.cohomology(S, a); bad += h[0] - h[1] + h[2] != F.chi(S, a)
P2 = surfaces["P2"]; bad += any(F.cohomology(P2, np.array([d, 0, 0]))[0] != math.comb(d + 2, 2) for d in range(0, 6))
check("FG2 IDENTITY line-bundle cohomology agrees with Riemann-Roch and with h^0(O_P2(d)) = C(d+2,2)", bad == 0, f"300 bundles + 6; mismatches {bad}")
G = {}; bad = 0; info = []
for nm, S in surfaces.items():
    C = F.toric_collection(S); ok = F.cohomology(S, np.zeros(S["m"], int)) == [1, 0, 0]
    for i, j in itertools.combinations(range(len(C)), 2):
        ok &= F.cohomology(S, C[i] - C[j]) == [0, 0, 0]; ok &= F.cohomology(S, C[j] - C[i])[0] >= 0
    Gm = F.gram(S, C); ok &= np.array_equal(np.tril(Gm, -1), 0 * Gm) and all(Gm[i, i] == 1 for i in range(len(Gm))) and all(Gm[i, j] > 0 for i in range(len(Gm)) for j in range(i + 1, len(Gm)))
    bad += not ok; G[nm] = Gm; info.append(f"{nm}: " + "/".join(",".join(str(int(x)) for x in Gm[i, i + 1:]) for i in range(len(Gm) - 1)))
check("FG3 IDENTITY the toric collection is exceptional (hence full) on each surface; its Gram matrix is unipotent with positive entries", bad == 0, "; ".join(info))
bad = 0; br = 0
for nm, Gm in G.items():
    inv0 = F.invariants(Gm); n = len(Gm); X = Gm
    for step in range(40):
        i = int(rng.integers(n - 1)); X = F.mutate(X, i)
        ok = all(X[r][c] == 0 for r in range(n) for c in range(r)) and all(X[r][r] == 1 for r in range(n))
        if not ok or (step % 10 == 0 and F.invariants(X) != inv0): bad += 1; break
    for i in range(n - 2):
        L = F.mutate(F.mutate(F.mutate(Gm, i), i + 1), i); R_ = F.mutate(F.mutate(F.mutate(Gm, i + 1), i), i + 1)
        br += L != R_
check("FG4 IDENTITY mutations keep the Gram matrix unipotent and integral, satisfy the braid relation, and preserve the isometry invariants", bad == 0 and br == 0,
      f"failures {bad}; braid-relation failures {br}")
inv = {nm: F.invariants(Gm) for nm, Gm in G.items()}; keys = {nm: (v["coxeter_charpoly"], v["snf_sym"], v["snf_anti"]) for nm, v in inv.items()}
check("FG5 IDENTITY the isometry invariants separate the five types (F1 and P1xP1 by the Smith form of G + G^T)", len(set(keys.values())) == 5 and inv["F1"]["snf_sym"] == (1, 1, 4, 4) and inv["P1xP1"]["snf_sym"] == (2, 2, 2, 2),
      "; ".join(f"{nm}: sym {v['snf_sym']}, anti {v['snf_anti']}" for nm, v in inv.items()))
def triple_vertex(f, g):
    x0 = F._x0(f, g); cons = [(np.array(F.NORMALS[c], float), -x0[c]) for c in F.NORMALS]; best = 1e9
    for (n1, b1), (n2, b2) in itertools.combinations(cons, 2):
        A = np.array([n1, n2])
        if abs(np.linalg.det(A)) < 1e-12: continue
        u = np.linalg.solve(A, [b1, b2])
        if all(n @ u >= b - 1e-7 for n, b in cons):
            tight = sorted(abs(n @ u - b) for n, b in cons); best = min(best, tight[2])
    return best
nchg = 0; badw = 0
for _ in range(150):
    f0, f1 = rng.dirichlet([1, 1]), rng.dirichlet([1, 1]); g0, g1 = rng.dirichlet([1, 1, 1]), rng.dirichlet([1, 1, 1])
    ts = np.linspace(0, 1, 201); fs = [tuple(sorted(F.cpl23_facets((1 - t) * f0 + t * f1, (1 - t) * g0 + t * g1))) for t in ts]
    for k in range(200):
        if fs[k] != fs[k + 1]:
            lo, hi = ts[k], ts[k + 1]
            for _ in range(50):
                mid = (lo + hi) / 2; fm = tuple(sorted(F.cpl23_facets((1 - mid) * f0 + mid * f1, (1 - mid) * g0 + mid * g1)))
                if fm == fs[k]: lo = mid
                else: hi = mid
            t = (lo + hi) / 2; nchg += 1; badw += triple_vertex((1 - t) * f0 + t * f1, (1 - t) * g0 + t * g1) > 1e-6
from pce import kahlerbridge as KB
f1set = set()
for al in np.linspace(0.02, 0.98, 49):
    v, lam = KB.F1(al); f1set.add(len(v))
check("FG6 IDENTITY the facet type changes only at support walls (three constraint lines through a vertex); F1 keeps its facets for all alpha",
      badw == 0 and nchg > 50 and f1set == {4}, f"{nchg} changes on 150 margin paths, {badw} not at a triple vertex; F1 facet counts over alpha {sorted(f1set)}")
print(f"{sum(results)}/{len(results)} passed")
