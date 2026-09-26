#!/usr/bin/env python3
"""GLUE -- does a family of local tables on a CYCLIC cover glue?

    python3 glue.py                 # same-source control + cross-source test
    python3 glue.py --alpha 3 --n 200000

THE COVER IS CYCLIC, WHICH IS THE POINT
---------------------------------------
Nine leaves, two ternary bracketings:

    rows     R1={0,1,2}  R2={3,4,5}  R3={6,7,8}
    columns  C1={0,3,6}  C2={1,4,7}  C3={2,5,8}

Every row meets every column in exactly one leaf, so the incidence between the
six contexts is K_{3,3}. That hypergraph fails the running intersection
property, so Vorob'ev's acyclicity condition FAILS and a family of consistent
marginals on these six contexts need not extend to a global joint.

Every corpus earlier in this work was a fixed tree, hence acyclic, hence
extension was guaranteed and any gluing obstruction was vacuous. Here it is not
vacuous -- which is the precondition for the obstruction to be measurable at
all.

WHAT IS TESTED
--------------
A global joint exists iff the linear program

    find p >= 0 over A^9 outcomes
    subject to  sum over the other six leaves of p = T_R  for each row R
                sum over the other six leaves of p = T_C  for each column C
                sum p = 1

is feasible. With A = 3 that is 3^9 = 19683 variables and 6*27 + 1 = 163
equality constraints -- decided exactly, not approximately.

TWO CASES, AND ONLY THE SECOND CAN FAIL
---------------------------------------
  SAME SOURCE   all six tables estimated from one corpus. A global joint
                exists by construction (the empirical joint), so the LP must
                be feasible. This is the control: if it fails, the code is
                wrong, not the mathematics.

  CROSS SOURCE  row tables from a corpus generated with the ROW bracketing,
                column tables from a corpus generated with the COLUMN
                bracketing, with the single-leaf marginals matched so the
                family is consistent on every overlap. Nothing guarantees a
                global joint. If the LP is infeasible, the obstruction is
                real: six locally valid, pairwise consistent tables that no
                distribution on nine leaves carries.

This is the setting of Wright's merging of conflicting tables and of
Abramsky-Mansfield-Barbosa's contextuality. The LP is ground truth; a
cohomological class would be an invariant that detects some of it (sufficient,
not necessary -- Hardy-type models are infeasible with a vanishing class).
"""
import argparse, itertools, math
import numpy as np
from scipy.optimize import linprog

ap = argparse.ArgumentParser()
ap.add_argument("--alpha", type=int, default=3, help="leaf alphabet")
ap.add_argument("--nsym", type=int, default=3)
ap.add_argument("--nrules", type=int, default=2)
ap.add_argument("--n", type=int, default=200000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tol", type=float, default=1e-7)
ap.add_argument("--smooth", type=float, default=1e-6,
                help="uniform mass added to every table cell. Needed because "
                     "empirical tables have structural zeros and the iterative "
                     "proportional fitting that matches overlaps divides by "
                     "them. NOTE this biases toward FEASIBILITY: a positive "
                     "table is easier to glue than one with zeros, so an "
                     "infeasible verdict is safe and a feasible one should be "
                     "rechecked at a smaller value.")
a = ap.parse_args()

ROWS = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
COLS = [(0, 3, 6), (1, 4, 7), (2, 5, 8)]
A = a.alpha

def make_grammar(rng, V, A, m):
    top = {s: [tuple(rng.randint(0, V, 3)) for _ in range(m)] for s in range(V)}
    bot = {s: [tuple(rng.randint(0, A, 3)) for _ in range(m)] for s in range(V)}
    return top, bot

def generate(n, groups, top, bot, rng, V):
    X = np.zeros((n, 9), dtype=np.int64)
    for i in range(n):
        r = rng.randint(V)
        kids = top[r][rng.randint(len(top[r]))]
        for g, sym in zip(groups, kids):
            leaves = bot[sym][rng.randint(len(bot[sym]))]
            for pos, lf in zip(g, leaves): X[i, pos] = lf
    return X

def table(X, ctx):
    """empirical distribution over the three leaves of a context"""
    idx = X[:, ctx[0]]*A*A + X[:, ctx[1]]*A + X[:, ctx[2]]
    t = np.bincount(idx, minlength=A**3).astype(float)
    t = t/t.sum()
    t = t + a.smooth
    return t/t.sum()

def single(T, which):
    """single-leaf marginal of a context table; which in {0,1,2}"""
    t = T.reshape(A, A, A)
    return t.sum(axis=tuple(i for i in range(3) if i != which))

def overlap_consistency(rowT, colT):
    """every row meets every column in one leaf, so consistency of the family
    is exactly agreement of those nine single-leaf marginals"""
    worst = 0.0
    for ri, R in enumerate(ROWS):
        for ci, C in enumerate(COLS):
            shared = set(R) & set(C)
            leaf = shared.pop()
            mr = single(rowT[ri], R.index(leaf))
            mc = single(colT[ci], C.index(leaf))
            worst = max(worst, float(np.abs(mr-mc).max()))
    return worst

def match_singles(colT, rowT):
    """rescale the column tables so their single-leaf marginals match the row
    tables', by iterative proportional fitting on each column table. This makes
    the family CONSISTENT on every overlap without making it realisable -- the
    whole question is whether consistency implies a global joint, and on a
    cyclic cover it need not."""
    out = []
    for ci, C in enumerate(COLS):
        T = colT[ci].reshape(A, A, A).copy()
        for _ in range(200):
            for k, leaf in enumerate(C):
                ri = next(i for i, R in enumerate(ROWS) if leaf in R)
                target = single(rowT[ri], ROWS[ri].index(leaf))
                cur = T.sum(axis=tuple(i for i in range(3) if i != k))
                scale = target/np.maximum(cur, 1e-300)
                shape = [1, 1, 1]; shape[k] = A
                T = T*scale.reshape(shape)
            T = np.maximum(T, 1e-300); T = T/T.sum()
        out.append(T.reshape(-1))
    return out

def glue_lp(rowT, colT):
    """feasibility of a global joint with the given context marginals"""
    N = A**9
    outcomes = np.array(list(itertools.product(range(A), repeat=9)),
                        dtype=np.int64)          # N x 9
    rows_eq = []; b_eq = []
    for ctxs, tabs in ((ROWS, rowT), (COLS, colT)):
        for ctx, T in zip(ctxs, tabs):
            code = (outcomes[:, ctx[0]]*A*A + outcomes[:, ctx[1]]*A
                    + outcomes[:, ctx[2]])
            for v in range(A**3):
                rows_eq.append((code == v).astype(float)); b_eq.append(T[v])
    rows_eq.append(np.ones(N)); b_eq.append(1.0)
    Aeq = np.array(rows_eq); beq = np.array(b_eq)
    res = linprog(c=np.zeros(N), A_eq=Aeq, b_eq=beq,
                  bounds=(0, None), method="highs")
    return res

def contextual_fraction(rowT, colT):
    """Abramsky-Barbosa-Mansfield contextual fraction. Binary feasibility is
    too brittle: the obstruction here lives on the SUPPORT (structural zeros),
    so 0.1% of uniform smoothing destroys it. The graded version asks for the
    largest weight of the family that IS explained by a global distribution,

        NCF = max  sum p   subject to   marginal_ctx(p) <= T_ctx  elementwise,
                                        p >= 0

    and reports CF = 1 - NCF. CF = 0 means the family gluesures entirely; CF > 0
    is the fraction that no global distribution can carry, and it degrades
    smoothly with noise instead of collapsing."""
    N = A**9
    outcomes = np.array(list(itertools.product(range(A), repeat=9)),
                        dtype=np.int64)
    rows_ub = []; b_ub = []
    for ctxs, tabs in ((ROWS, rowT), (COLS, colT)):
        for ctx, T in zip(ctxs, tabs):
            code = (outcomes[:, ctx[0]]*A*A + outcomes[:, ctx[1]]*A
                    + outcomes[:, ctx[2]])
            for v in range(A**3):
                rows_ub.append((code == v).astype(float)); b_ub.append(T[v])
    res = linprog(c=-np.ones(N), A_ub=np.array(rows_ub), b_ub=np.array(b_ub),
                  bounds=(0, None), method="highs")
    ncf = -res.fun if res.status == 0 else 0.0
    return max(0.0, 1.0 - float(ncf))

rng = np.random.RandomState(a.seed)
top, bot = make_grammar(rng, a.nsym, A, a.nrules)
print(f"  glue: 9 leaves, alphabet {A}, {A**9} outcomes, "
      f"{6*A**3+1} equality constraints")
print(f"  rows {ROWS}   columns {COLS}   smoothing {a.smooth:g}")
print(f"  incidence is K_3,3 -- the cover is CYCLIC, so Vorob'ev gives no")
print(f"  extension guarantee and the question is genuinely open.\n")

Xr = generate(a.n, ROWS, top, bot, rng, a.nsym)
Xc = generate(a.n, COLS, top, bot, rng, a.nsym)

# ---- control: all six tables from one corpus
rowT = [table(Xr, c) for c in ROWS]; colT = [table(Xr, c) for c in COLS]
cons = overlap_consistency(rowT, colT)
res = glue_lp(rowT, colT)
print(f"  SAME SOURCE (control)")
print(f"    overlap consistency (max |difference| on shared leaves) {cons:.2e}")
print(f"    LP: {'FEASIBLE' if res.status == 0 else 'INFEASIBLE'}"
      f"   -> {'PASS' if res.status == 0 else 'FAIL -- the code is wrong'}")
print(f"    a global joint exists by construction here, so feasibility is the")
print(f"    only correct answer and this row tests the implementation.\n")

# ---- test: rows from one source, columns from another
rowT2 = [table(Xr, c) for c in ROWS]
colT2 = [table(Xc, c) for c in COLS]
before = overlap_consistency(rowT2, colT2)
colT2 = match_singles(colT2, rowT2)
after = overlap_consistency(rowT2, colT2)
res2 = glue_lp(rowT2, colT2)
print(f"  CROSS SOURCE (rows from the row-bracketed corpus, columns from the")
print(f"  column-bracketed one)")
print(f"    overlap consistency before matching {before:.2e}, after {after:.2e}")
print(f"    LP: {'FEASIBLE' if res2.status == 0 else 'INFEASIBLE'}")
if res2.status == 0:
    print(f"    The two families glue: consistency on overlaps was enough here,")
    print(f"    even though the cover is cyclic and gave no guarantee. The")
    print(f"    obstruction is zero for THIS family, not vacuously but as a")
    print(f"    computed fact.")
else:
    print(f"    No distribution on nine leaves carries all six tables. Six")
    print(f"    locally valid, pairwise consistent contexts with no global")
    print(f"    section: a genuine gluing obstruction, of the kind that cannot")
    print(f"    occur on an acyclic cover.")
cf_same = contextual_fraction(rowT, colT)
cf_cross = contextual_fraction(rowT2, colT2)
print(f"\n  CONTEXTUAL FRACTION (graded, not binary)")
print(f"    same source  CF = {cf_same:.4f}")
print(f"    cross source CF = {cf_cross:.4f}")
print(f"    CF is the weight of the family that no global distribution can")
print(f"    carry. Binary feasibility flips to FEASIBLE under 0.1% smoothing,")
print(f"    because the obstruction sits on the support; CF degrades smoothly")
print(f"    and is the measure to use on anything noisy.")
print(f"\n  The LP is ground truth. A cohomological class would be an invariant")
print(f"  detecting some infeasible families and not others (sufficient, not")
print(f"  necessary), so the pairing LP-vs-class is what makes the cohomology")
print(f"  falsifiable rather than decorative.")
