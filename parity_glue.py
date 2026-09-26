#!/usr/bin/env python3
"""PARITY_GLUE -- an obstruction that is a theorem, not an accident.

    python3 parity_glue.py                # control, obstruction, noise sweep
    python3 parity_glue.py --alpha 3

WHY THIS REPLACES THE RANDOM-GRAMMAR VERSION
--------------------------------------------
glue.py found a real obstruction from two randomly generated tree corpora, but
its contextual fraction was 0.0028 and it died under 0.1% smoothing: the
obstruction sat on the support pattern of a few sampled cells, which is noise
in everything but name. A case that small cannot carry a cohomology argument.

Here the impossibility is designed and provable. Arrange the nine leaves as a
3x3 grid, so the row contexts are the grid rows and the column contexts are its
columns -- the same cyclic K_3,3 cover as before. Impose

    every ROW sums to 0   (mod 3)
    every COLUMN sums to s_j (mod 3)

Summing all nine cells two ways gives 0 from the rows and sum_j s_j from the
columns. So whenever sum_j s_j != 0 (mod 3) NO assignment of the nine cells
satisfies all six constraints, while every context on its own is perfectly
satisfiable and all single-leaf marginals are uniform, so the family is exactly
consistent on every overlap. This is the mod-3 analogue of the Mermin-Peres
magic square, and it is strongly contextual: not merely unglueable on average
but with no global assignment at all.

  CONTROL      all column sums 0: total is 0 both ways, a global joint exists,
               contextual fraction must be 0.
  OBSTRUCTION  one column sum 1: total is 0 and 1, contextual fraction must be
               large.

NOISE ROBUSTNESS
----------------
Each table is mixed with the uniform distribution at weight eps and the
contextual fraction is recomputed. A designed obstruction should degrade
linearly and survive percent-level noise, unlike the sampled one which
vanished at 0.001.
"""
import argparse, itertools
import numpy as np
from scipy.optimize import linprog

ap = argparse.ArgumentParser()
ap.add_argument("--alpha", type=int, default=3)
ap.add_argument("--eps-sweep", default="0,0.01,0.05,0.1,0.2,0.4,0.6")
ap.add_argument("--quiet", action="store_true")
a = ap.parse_args()
A = a.alpha
ROWS = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
COLS = [(0, 3, 6), (1, 4, 7), (2, 5, 8)]
OUT = np.array(list(itertools.product(range(A), repeat=9)), dtype=np.int64)
TRI = np.array(list(itertools.product(range(A), repeat=3)), dtype=np.int64)

def parity_table(target, eps):
    """uniform over the triples summing to `target` mod A, then mixed with the
    uniform distribution at weight eps"""
    ok = (TRI.sum(1) % A) == target
    t = ok.astype(float); t /= t.sum()
    t = (1-eps)*t + eps*np.full(A**3, 1.0/A**3)
    return t

def single(T, k):
    return T.reshape(A, A, A).sum(axis=tuple(i for i in range(3) if i != k))

def overlap_consistency(rowT, colT):
    worst = 0.0
    for ri, R in enumerate(ROWS):
        for ci, C in enumerate(COLS):
            leaf = (set(R) & set(C)).pop()
            worst = max(worst, float(np.abs(single(rowT[ri], R.index(leaf))
                                            - single(colT[ci], C.index(leaf))).max()))
    return worst

def ctx_codes():
    out = []
    for ctxs in (ROWS, COLS):
        for c in ctxs:
            out.append(OUT[:, c[0]]*A*A + OUT[:, c[1]]*A + OUT[:, c[2]])
    return out
CODES = ctx_codes()

def rows_for(tabs):
    Aeq = []; beq = []
    for code, T in zip(CODES, tabs):
        for v in range(A**3):
            Aeq.append((code == v).astype(float)); beq.append(T[v])
    return np.array(Aeq), np.array(beq)

def feasible(tabs):
    Aeq, beq = rows_for(tabs)
    Aeq = np.vstack([Aeq, np.ones(len(OUT))]); beq = np.append(beq, 1.0)
    r = linprog(np.zeros(len(OUT)), A_eq=Aeq, b_eq=beq, bounds=(0, None),
                method="highs")
    return r.status == 0

def contextual_fraction(tabs):
    Aub, bub = rows_for(tabs)
    r = linprog(-np.ones(len(OUT)), A_ub=Aub, b_ub=bub, bounds=(0, None),
                method="highs")
    ncf = -r.fun if r.status == 0 else 0.0
    return max(0.0, 1.0 - float(ncf))

def family(col_sums, eps):
    rowT = [parity_table(0, eps) for _ in ROWS]
    colT = [parity_table(s, eps) for s in col_sums]
    return rowT + colT, rowT, colT

print(f"  parity_glue: 3x3 grid, alphabet {A}, {A**9} outcomes")
print(f"  rows {ROWS}\n  columns {COLS}   (cyclic K_3,3 cover)")
print(f"  rows all sum to 0 mod {A}; column sums are the design parameter.")
print(f"  Summing the grid two ways forces  0 == sum of column sums (mod {A}),")
print(f"  so any other choice is globally impossible while every context is")
print(f"  individually satisfiable.\n")

for name, sums in (("CONTROL      column sums (0,0,0)", (0, 0, 0)),
                   ("OBSTRUCTION  column sums (0,0,1)", (0, 0, 1)),
                   ("OBSTRUCTION  column sums (1,1,1)", (1, 1, 1))):
    tabs, rowT, colT = family(sums, 0.0)
    cons = overlap_consistency(rowT, colT)
    fz = feasible(tabs); cf = contextual_fraction(tabs)
    tot = sum(sums) % A
    print(f"  {name}")
    print(f"    sum of column sums mod {A} = {tot}  "
          f"({'consistent' if tot == 0 else 'CONTRADICTS the row total of 0'})")
    print(f"    overlap consistency {cons:.2e}   "
          f"LP {'FEASIBLE' if fz else 'INFEASIBLE'}   CF {cf:.4f}")

print(f"\n  NOISE ROBUSTNESS of the (0,0,1) family: each table mixed with")
print(f"  uniform at weight eps.")
print(f"    {'eps':>7}{'CF':>10}{'LP':>13}")
for e in (float(x) for x in a.eps_sweep.split(",")):
    tabs, rowT, colT = family((0, 0, 1), e)
    print(f"    {e:>7.3g}{contextual_fraction(tabs):>10.4f}"
          f"{'FEASIBLE' if feasible(tabs) else 'INFEASIBLE':>13}")
print(f"\n  A designed obstruction degrades smoothly with noise instead of")
print(f"  vanishing at the first smoothing, which is what makes it usable as")
print(f"  the positive case for a cohomological invariant: the class must be")
print(f"  nonzero where CF is large, and the interesting question is how far")
print(f"  down the noise sweep it stays nonzero.")
