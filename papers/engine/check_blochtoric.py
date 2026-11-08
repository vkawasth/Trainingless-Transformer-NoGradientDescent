"""Checks for the Bloch picture of the Floer objects of Part IV (prop:bloch, module pce/blochtoric.py).
IDENTITY = an exact or numerical instance of a statement proved in the text; NUMERICAL = floating-point, with tolerances stated.
  BL1  IDENTITY     z = sqrt(p) e^{i theta}: the Fubini-Study line element is (1/4) sum dp^2 / p + sum p dtheta^2 - (sum p dtheta)^2 (3 and 8
                    components); a qubit's Bloch vector has length 1 with z-component 1 - 2p and (x, y) = 2 sqrt(p(1-p)) (cos, sin) theta
  BL2  IDENTITY     example A is three Bloch spheres: the 8 objects map to the product states of columns 1, 3, 4 with p = P(row 2 | column) = 1/2
                    and longitudes in {0, pi}: the eigenbasis of sigma_x x sigma_x x sigma_x, Gram matrix the identity; every lattice
                    identification (50 random unimodular matrices) maps the holonomy set onto the full 2-torsion group, so the basis is canonical
  BL3  NUMERICAL    on every fibre the leading holonomies form a coset of a finite subgroup of the torus, of order the number of objects there;
                    on the leximin fibre it is a subgroup (it contains the positive object): A (Z/2)^3, B Z/6, C Z/2 x Z/4, and C's second fibre a
                    Z/2-coset not containing 0; objects on one fibre carry one coupling, so every decision agrees on them
  BL4  NUMERICAL    for B and C a lattice identification is a choice: random unimodular identifications give different image sets of holonomies
                    (so different Bloch placements and Gram matrices), unlike A
"""
import os, sys
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pce import blochtoric as B
from pce import cpl24 as C
results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)
rng = np.random.default_rng(20261108)

# BL1
dev = 0.0; bv = 0.0
for n in (3, 8):
    for _ in range(200):
        p = rng.dirichlet(np.ones(n)); dp = rng.normal(size=n); dp -= dp.mean(); dth = rng.normal(size=n)
        ref = 0.25 * np.sum(dp ** 2 / p) + np.sum(p * dth ** 2) - np.sum(p * dth) ** 2; dev = max(dev, abs(B.fs_metric(p, dp, dth) - ref) / ref)
for _ in range(200):
    p, th = rng.uniform(0, 1), rng.uniform(0, 2 * np.pi); q = B.qubit(p, th); b = B.bloch(p, th)
    ex = np.array([2 * (np.conj(q[0]) * q[1]).real, 2 * (np.conj(q[0]) * q[1]).imag, abs(q[0]) ** 2 - abs(q[1]) ** 2])
    bv = max(bv, abs(np.linalg.norm(b) - 1), np.abs(b - ex).max())
check("BL1 IDENTITY Fubini-Study = (1/4) Fisher + phase term; Bloch vector of a qubit", dev < 1e-12 and bv < 1e-12,
      f"400 random points (3 and 8 components): largest relative deviation {dev:.1e}; 200 qubits: Bloch deviation {bv:.1e}")

def unimodular(rng, steps=12):
    U = np.eye(3, dtype=int)
    for _ in range(steps):
        i, j = rng.choice(3, 2, replace=False); E = np.eye(3, dtype=int); E[i, j] = rng.choice([-1, 1]); U = U @ E
    return U

data = {}
for nm, ex, L in (("A", C.EX_A, 200.0), ("B", C.EX_B, 400.0), ("C", C.EX_C, 400.0)):
    objs, bound = B.objects(ex, L); data[nm] = (ex, B.rounded(objs), bound)

# BL2
(f, g), R, _ = data["A"]; states = []; probs = set()
for u, t in R:
    x11, x12, x13 = u; x14 = f[0] - x11 - x12 - x13
    cols = [(x11, g[0] - x11, t[0]), (x13, g[2] - x13, t[2]), (x14, g[3] - x14, (-(t[0] + t[1] + t[2])) % 2)]   # (row 1, row 2, longitude / pi)
    q = np.array([1.0 + 0j])
    for r1, r2, lon in cols:
        p = r2 / (r1 + r2); probs.add(p); q = np.kron(q, B.qubit(float(p), float(lon) * np.pi))
    states.append(q)
S = np.array(states); Gm = np.abs(S.conj() @ S.T); hol = [t for _, t in R]
full = set((a, b, c) for a in (Fr(0), Fr(1)) for b in (Fr(0), Fr(1)) for c in (Fr(0), Fr(1)))
canon = all(B.identify(hol, unimodular(rng)) == full for _ in range(50))
check("BL2 IDENTITY example A: the 8 objects are the sigma_x product basis of three Bloch spheres (columns 1, 3, 4), canonically",
      probs == {Fr(1, 2)} and np.abs(Gm - np.eye(8)).max() < 1e-12 and set(hol) == full and canon,
      f"column-conditional probabilities {sorted(str(p) for p in probs)}; largest deviation of the Gram matrix from the identity {np.abs(Gm - np.eye(8)).max():.1e}; "
      f"holonomies = all of (Z/2)^3: {set(hol) == full}; 50 random unimodular identifications all give the full 2-torsion group: {canon}")

# BL3
rows = []; ok = True
for nm in "ABC":
    ex, R, bound = data[nm]; fibs = {}
    for u, t in R: fibs.setdefault(u, []).append(t)
    tot = 0
    for u, Sx in fibs.items():
        cl, n, D = B.holonomy_group(Sx); has0 = (0, 0, 0) in [tuple(x) for x in Sx]; inv = B.invariant_factors(D)
        ok &= cl and n == len(Sx); tot += n
        rows.append(f"{nm} fibre ({', '.join(str(x) for x in u)}): {n} objects, coset closed {cl}, contains 0 {has0}, invariant factors {inv}")
    ok &= tot == bound
lex = {nm: C.leximin_exact(*data[nm][0])["point"] for nm in "ABC"}
groups = {}
for nm in "ABC":
    ex, R, _ = data[nm]; Sx = [t for u, t in R if tuple(u) == tuple(lex[nm])]; groups[nm] = ((0, 0, 0) in [tuple(x) for x in Sx], B.invariant_factors(B.holonomy_group(Sx)[2]))
ok &= groups == {"A": (True, [2, 2, 2]), "B": (True, [6]), "C": (True, [2, 4])}
check("BL3 NUMERICAL leading holonomies on each fibre form a coset of a finite group; a subgroup on the leximin fibre", ok, "; ".join(rows))

# BL4
cnt = {}
for nm in "ABC":
    ex, R, _ = data[nm]; Sx = [t for u, t in R if tuple(u) == tuple(lex[nm])]
    imgs = set(frozenset(B.identify(Sx, unimodular(rng))) for _ in range(50)); cnt[nm] = len(imgs)
check("BL4 NUMERICAL the Bloch placement needs a lattice identification: canonical for A, a choice for B and C",
      cnt["A"] == 1 and cnt["B"] > 1 and cnt["C"] > 1, f"distinct image sets of the leximin-fibre holonomies under 50 random unimodular identifications: {cnt}")
print(f"{sum(results)}/{len(results)} passed")
