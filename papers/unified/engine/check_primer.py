"""Checks for the reader's guide to spectral scale valuations (sec:primer, before def:scales). New script; it modifies nothing.
  PM1 IDENTITY  the hand example: normals (1,0), (-1,0) at valuation 0, (0,1) at a, (1,1) at b > a. Exactly (symbolic T): the scale
                valuations of Q are (0, a); the greedy picks one normal at level 0 and one at level a; the level b adds nothing
  PM2 IDENTITY  positivity matters: normals (1,0), (0,1), (1,1) with weights 1, 1, -1/2 at valuation 0 and (1,1) again... at valuation a:
                the leading part has rank 1 although its normals span 2, and the scales are (0, a), not (0, 0) as the positive rule
                would predict (exact, symbolic)
  PM3 IDENTITY  the Newton polygon: for random positive Gram matrices Q = sum T^(h_k) v_k v_k^T (T = e^(-L), 200 digits), the eigenvalue
                valuations equal the successive slopes val e_k - val e_(k-1) of the elementary symmetric functions, and the greedy weights
  (first run: PM3 2 and PM4 3 mismatches above 0.02, because a single L = 400 leaves an error log(c)/L from the leading coefficients c;
  now two values of L, 400 and 1600, extrapolated linearly in 1/L; at 800 digits the smallest eigenvalues, near e^-2000, came out
  complex from rounding, so 1300 digits)
  PM4 IDENTITY  the classical case: a weighted graph Laplacian sum_e T^(w_e) (e_i - e_j)(e_i - e_j)^T (grounded at one vertex) has scale
                valuations equal to the edge weights of Kruskal's minimum spanning tree (the graphic-matroid instance of the theorem)
"""
import os, sys, itertools
import numpy as np
import sympy as sp
import mpmath as mp
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
T, a, b = sp.symbols("T a b", positive=True)
def val_of(expr, subs):
    """valuation in T of a positive expression: substitute T = e^(-L) for large L and read the exponent (exact via series in T)."""
    e = sp.expand(expr.subs(subs)); terms = sp.Add.make_args(e); exps = []
    for t in terms:
        c, pw = t.as_coeff_exponent(T); exps.append((pw, c))
    m = min(p for p, c in exps); lead = sum(c for p, c in exps if p == m)
    return m if lead != 0 else None
sub = {a: sp.Rational(1, 3), b: sp.Rational(3, 4)}
v = [sp.Matrix([1, 0]), sp.Matrix([-1, 0]), sp.Matrix([0, 1]), sp.Matrix([1, 1])]; w = [1, 1, T ** a, T ** b]
Q = sum((wk * vk * vk.T for wk, vk in zip(w, v)), sp.zeros(2, 2))
e1 = Q.trace(); e2 = sp.expand(Q.det()); d1 = val_of(e1, sub); d2 = val_of(e2, sub)
check("PM1 IDENTITY hand example: scales (0, a); greedy takes one normal at level 0 and one at level a; level b adds nothing",
      d1 == 0 and d2 == sub[a], f"Q = {Q.subs(sub)}; det Q = {e2}; val e1 = {d1}, val e2 = {d2}")
v2 = [sp.Matrix([1, 0]), sp.Matrix([0, 1]), sp.Matrix([1, 1]), sp.Matrix([1, 1])]; w2 = [1, 1, -sp.Rational(1, 2), T ** a]
Q2 = sum((wk * vk * vk.T for wk, vk in zip(w2, v2)), sp.zeros(2, 2)); M0 = Q2.subs(T, 0)
d2b = val_of(sp.expand(Q2.det()), sub)
check("PM2 IDENTITY with mixed signs the leading part loses rank (1 < span 2) and the second scale becomes a, not 0", M0.rank() == 1 and d2b == sub[a],
      f"leading part {M0} (rank {M0.rank()}); det Q = {sp.expand(Q2.det())}; val e2 = {d2b}")
mp.mp.dps = 1300; rng = np.random.default_rng(20261125); bad = 0
def rank(vs): return int(np.linalg.matrix_rank(np.array(vs, float))) if vs else 0
def scales(vs, hs, d):
    out = []
    for L in (mp.mpf(400), mp.mpf(1600)):
        Qm = mp.matrix(d, d)
        for vk, hk in zip(vs, hs):
            for i in range(d):
                for j in range(d): Qm[i, j] += mp.e ** (-L * hk) * float(vk[i]) * float(vk[j])
        out.append(sorted(float(-mp.log(abs(x)) / L) for x in mp.eigsy(Qm)[0]))
    return [(4 * y - x) / 3 for x, y in zip(*out)]                 # linear extrapolation in 1/L from 1/400 and 1/1600
def greedy(vs, hs):
    out = []; ch = []
    for i in sorted(range(len(vs)), key=lambda i: hs[i]):
        if rank(ch + [vs[i]]) > len(ch): ch.append(vs[i]); out.append(hs[i])
    return out
for _ in range(60):
    d = int(rng.integers(2, 5)); m = int(rng.integers(d, d + 4)); vs = [rng.integers(-2, 3, size=d) for _ in range(m)]
    if rank(vs) < d: continue
    hs = [float(x) for x in rng.integers(0, 6, size=m) / 4]; ev = scales(vs, hs, d)
    bad += max(abs(x - y) for x, y in zip(ev, sorted(greedy(vs, hs)))) > 0.02
check("PM3 IDENTITY eigenvalue valuations = Newton-polygon slopes = greedy weights for positive Gram matrices", bad == 0, f"random cases; mismatches {bad}")
bad = 0; tot = 0
for _ in range(60):
    n = int(rng.integers(3, 7)); edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.7]
    if len(edges) < n - 1: continue
    wts = {e: float(rng.integers(0, 8)) / 4 for e in edges}; vs = []; hs = []
    for (i, j) in edges:
        x = np.zeros(n); x[i], x[j] = 1, -1; vs.append(x[1:]); hs.append(wts[(i, j)])       # grounded at vertex 0
    if rank(vs) < n - 1: continue
    tot += 1; par = list(range(n))
    def find(x):
        while par[x] != x: x = par[x]
        return x
    mst = []
    for e in sorted(edges, key=lambda e: wts[e]):
        r1, r2 = find(e[0]), find(e[1])
        if r1 != r2: par[r1] = r2; mst.append(wts[e])
    ev = scales(vs, hs, n - 1); bad += max(abs(x - y) for x, y in zip(ev, sorted(mst))) > 0.02
check("PM4 IDENTITY grounded weighted graph Laplacians: scale valuations = Kruskal minimum-spanning-tree weights", bad == 0 and tot > 30, f"{tot} graphs; mismatches {bad}")
print(f"{sum(results)}/{len(results)} passed")
