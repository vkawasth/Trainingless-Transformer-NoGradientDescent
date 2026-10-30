"""check_csd.py -- testing the CSD framework's descent claims on the lab's systems (lab paper, Proposition "the lab inside CSD").

The CSD paper (Chang, hal-05711275) asserts, under its hypotheses, a canonical class Obs_CSD(A) in H^2 of a linearised sheaf of local
realizations with Obs_CSD(A) = 0 iff every compatible family of local realizations glues (its Theorem 7), the primary contextual
obstruction in degree 2, and higher contextuality in Obs^n, n >= 3. Its construction of Spec(A) needs an operator-semantic system; the
lab supplies the shared data only: the context category of windows, the covering by windows and the presheaf of local laws. Here we
compute what any Cech-type obstruction theory on those data can and cannot do.

  CX1 R-linearisation (signed measures): linear Cech cohomology vanishes in degrees >= 1 on the lab's covers, yet the liar cycle is a
      compatible family that does not glue: "obstruction = 0 iff glues" fails for this linearisation
  CX2 degrees: the nerve of the pair cover at m = 3 is a hollow triangle (reduced Betti b1 = 1, b2 = 0), so Cech H^2 vanishes for every
      coefficient group; at m = 4 the nerve of the six pair windows has b1 = 3, b2 = 0, b3 = 0
  CX3 Z/2 support linearisation (all-versus-nothing): the liar cycle's supports are affine parity conditions a+b = 0, b+c = 0, a+c = 1,
      inconsistent; the edge-parity cochain (0, 0, 1) on the graph Gamma of windows is not a coboundary: a non-zero class in H^1(Gamma; Z/2),
      degree 1 (the same group as the latent sign class of Theorem M)
  CX4 a contextual family invisible to both linearisations: the noisy liar (1 - q) liar + q (independent uniform) for 0 < q < 2/3 is
      compatible, has full support (so every support-based class vanishes), and is contextual (lambda* < 1; the Bell value 3 - 1.5 q > 2);
      it glues exactly from q = 2/3. Only the positivity LP detects it
  CX5 the latent layer: the label-switch stack's obstruction (the sign class) has degree 1 on the graph of pairs and H^2 = 0 there; no
      degree >= 2 class arises on the lab's covers
"""
import itertools, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pce import descent as DE
from pce import Window
from pce.cohomology import separate, hodge
from pce.integrate import gluable_fraction
from pce.complex import nerve, reduced_betti

results = []
def check(name, cond, info=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else ""), flush=True)

D3 = {"A": 2, "B": 2, "C": 2}; eq = np.array([[0.5, 0], [0, 0.5]]); ne = np.array([[0, 0.5], [0.5, 0]])
liar = [Window(("A", "B"), eq), Window(("B", "C"), eq), Window(("A", "C"), ne)]

# CX1
h3 = DE.level0_cohomology(3, [(0, 1), (0, 2), (1, 2)]); h4 = DE.level0_cohomology(4, list(itertools.combinations(range(4), 2)))
pr = separate(D3, liar)
check("CX1 R-linear Cech cohomology vanishes in degrees >= 1, yet the liar cycle is compatible and does not glue", all(x == 0 for x in h3[1:]) and all(x == 0 for x in h4[1:])
      and pr.kind == "positivity", f"H^q (m = 3) {[int(x) for x in h3]}, (m = 4) {[int(x) for x in h4]}; liar: {pr.kind}, lambda* {pr.lam:.3f}")

# CX2
ctx3 = {"AB": {0, 1}, "BC": {1, 2}, "AC": {0, 2}}; N3 = nerve(ctx3); b3 = reduced_betti([tuple(sorted(J)) for J in N3])
ctx4 = {f"{i}{j}": {i, j} for i, j in itertools.combinations(range(4), 2)}; N4 = nerve(ctx4); b4 = reduced_betti([tuple(sorted(J)) for J in N4])
check("CX2 nerves: the m = 3 pair cover is a hollow triangle (b1 = 1, b2 = 0); the m = 4 pair cover has b1 = 3 and no b2, b3",
      b3.get(1) == 1 and b3.get(2, 0) == 0 and b4.get(1) == 3 and b4.get(2, 0) == 0 and b4.get(3, 0) == 0, f"m = 3: {b3}; m = 4: {b4}")

# CX3
def supports(ws): return [{(a, b) for a in (0, 1) for b in (0, 1) if w.table[a, b] > 1e-12} for w in ws]
sup = supports(liar); parity = [{a ^ b for a, b in s} for s in sup]
affine = all(len(p) == 1 for p in parity); cochain = [next(iter(p)) for p in parity]          # edge parities on AB, BC, AC
edges = [(0, 1), (1, 2), (0, 2)]; B = np.zeros((3, 3), np.uint8)
for k, (i, j) in enumerate(edges): B[k, i] = B[k, j] = 1
nonzero = not DE.in_image_gf2(B, np.array(cochain)) and sum(cochain) % 2 == 1
no_global = not any(all((x[i], x[j]) in s for (i, j), s in zip(edges, sup)) for x in itertools.product((0, 1), repeat=3))
check("CX3 the liar cycle: affine parity supports, edge-parity cochain (0, 0, 1) is not a coboundary: a non-zero class in H^1(Gamma; Z/2), the group of the latent sign class",
      affine and nonzero and no_global, f"parities {cochain}; no global assignment is consistent with the supports")

# CX4
rows = []; thr = None
for q in np.linspace(0, 1, 101):
    mix = lambda T: (1 - q) * T + q * np.full((2, 2), 0.25)
    W = [Window(("A", "B"), mix(eq)), Window(("B", "C"), mix(eq)), Window(("A", "C"), mix(ne))]
    lam = gluable_fraction(D3, W).lam; bell = np.trace(W[0].table) + np.trace(W[1].table) + W[2].table[0, 1] + W[2].table[1, 0]
    full = all(len(s) == 4 for s in supports(W)); en = hodge(D3, W).energy
    rows.append((q, lam, bell, full, en))
    if thr is None and lam > 1 - 1e-7: thr = q
mid = [r for r in rows if 0 < r[0] < 2 / 3 - 1e-9]
check("CX4 the noisy liar: compatible, full support, contextual for 0 < q < 2/3 (lambda* < 1, Bell value 3 - 1.5q > 2), glues from q = 2/3",
      all(r[3] and r[4] < 1e-12 and r[1] < 1 - 1e-7 and r[2] > 2 + 1e-9 and abs(r[2] - (3 - 1.5 * r[0])) < 1e-12 for r in mid) and abs(thr - 0.67) < 0.011,
      f"first glued q on the grid {thr:.2f}; lambda* at q = 0.3: {rows[30][1]:.3f}")

# CX5
e3, B3 = DE.pair_graph(3); e4, B4 = DE.pair_graph(4)
g3 = reduced_betti([(i,) for i in range(3)] + [tuple(e) for e in e3]); g4 = reduced_betti([(i,) for i in range(4)] + [tuple(e) for e in e4])
check("CX5 the latent layer: the sign class lives in H^1 of the graph of pairs (rank 1 on the triangle, 3 on K4); the graph has no H^2",
      g3.get(1) == 1 and g4.get(1) == 3 and g3.get(2, 0) == 0 and g4.get(2, 0) == 0, f"triangle {g3}; K4 {g4}")

print(f"{sum(results)}/{len(results)} passed")
