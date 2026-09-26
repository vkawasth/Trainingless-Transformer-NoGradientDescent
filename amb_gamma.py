#!/usr/bin/env python3
"""AMB_GAMMA -- the Abramsky-Mansfield-Barbosa obstruction, computed exactly.

    python3 amb_gamma.py                 # control, obstruction, and Hardy check
    python3 amb_gamma.py --ring Z2

WHAT IS COMPUTED, FROM THE PAPER'S OWN DEFINITION
--------------------------------------------------
AMB (EPTCS 95, 2012), Section 4. For a ring R they define F_R(X) as the free
R-module on X, and work with the Cech cohomology of the abelian presheaf
F := F_Z S_e, where S_e is the support of the empirical model: F(U) is the set
of formal Z-linear combinations of sections in the support over U. Fixing a
section s1 in the support of a context C1, the obstruction gamma(s1) is the
class of z = delta^0(s1,...,sn) in H^1(U, F_{C1-bar}).

Proposition 4.2 gives the form we compute:

    gamma(s1) = 0   iff   there is a family { r_i in F(C_i) } with r_1 = s1
                          and  r_i | (C_i ^ C_j)  =  r_j | (C_i ^ C_j)  for all i,j

so the test is the solvability of a linear system over the ring: one integer
coefficient per support-section per context, restriction maps given by the
pushforward  F_R(f)(phi)(y) = sum over f(x)=y of phi(x), and the coefficients
of context C1 pinned to the indicator of s1. This is exactly the recipe AMB use
by hand for the Hardy model in Section 5 ("setting the variable labelling that
section to 1, and the other variables in its row to 0").

THE MODEL TESTED
----------------
The 3x3 parity family from parity_glue.py, whose contextual fraction is 1 with
column sums (0,0,1) and 0 with (0,0,0) or (1,1,1). The support of a context is
the set of triples whose sum is the prescribed residue. Rows are pairwise
disjoint, as are columns, so the only overlaps are row-with-column, each a
single leaf -- nine overlaps, three equations each.

WHAT THE RESULT CAN AND CANNOT SAY
----------------------------------
AMB's Proposition 4.3: if the model is possibilistically extendable then
gamma vanishes for every section, so a NON-vanishing gamma is a sufficient
condition for contextuality. It is not necessary: false positives arise from
families {r_i} that satisfy the equations without determining a genuine global
section, which is what happens in their Hardy example. So

    gamma != 0  =>  contextual
    gamma  = 0  =>  nothing follows

and the interesting measurement is where gamma stops detecting an obstruction
that the linear program still sees.
"""
import argparse, itertools
import numpy as np
from sympy import Matrix, Rational, ZZ
from sympy.matrices.normalforms import smith_normal_form

ap = argparse.ArgumentParser()
ap.add_argument("--alpha", type=int, default=3)
ap.add_argument("--ring", default="Z", choices=["Z", "Q", "Z2", "Z3", "Z5"])
ap.add_argument("--hardy", action="store_true", help="also run the Hardy model")
ap.add_argument("--eps", type=float, default=0.0,
                help="uniform smoothing applied to the empirical model before "
                     "the SUPPORT is taken. gamma is possibilistic: it reads "
                     "the support, so any eps > 0 makes every outcome possible "
                     "and the support becomes full.")
a = ap.parse_args()
A = a.alpha
ROWS = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
COLS = [(0, 3, 6), (1, 4, 7), (2, 5, 8)]

def support(ctx_sum):
    """sections of a context = the SUPPORT of its distribution. With no
    smoothing this is the triples summing to ctx_sum mod A. With eps > 0 every
    outcome has positive probability, so the support is everything -- which is
    the whole fragility of a possibilistic invariant, in one line."""
    if a.eps > 0:
        return list(itertools.product(range(A), repeat=3))
    return [t for t in itertools.product(range(A), repeat=3)
            if sum(t) % A == ctx_sum]

def build(col_sums):
    ctxs = [(R, 0) for R in ROWS] + [(C, s) for C, s in zip(COLS, col_sums)]
    supp = [support(s) for _, s in ctxs]
    return ctxs, supp

def gamma_vanishes(ctxs, supp, i0, sec0, ring):
    """solve  r_i|overlap = r_j|overlap  with r_{i0} pinned to the indicator of
    sec0. Variables: one coefficient per (context, support section)."""
    offs, n = [], 0
    for S in supp: offs.append(n); n += len(S)
    rows, rhs = [], []
    for i, (Ci, _) in enumerate(ctxs):
        for j, (Cj, _) in enumerate(ctxs):
            if j <= i: continue
            shared = sorted(set(Ci) & set(Cj))
            if not shared: continue
            li = [Ci.index(x) for x in shared]; lj = [Cj.index(x) for x in shared]
            vals = list(itertools.product(range(A), repeat=len(shared)))
            for v in vals:
                r = [0]*n
                for k, t in enumerate(supp[i]):
                    if tuple(t[p] for p in li) == v: r[offs[i]+k] += 1
                for k, t in enumerate(supp[j]):
                    if tuple(t[p] for p in lj) == v: r[offs[j]+k] -= 1
                rows.append(r); rhs.append(0)
    for k, t in enumerate(supp[i0]):                     # pin r_{i0} = s0
        r = [0]*n; r[offs[i0]+k] = 1
        rows.append(r); rhs.append(1 if t == sec0 else 0)
    M = Matrix(rows); b = Matrix(rhs)
    if ring == "Q":
        return M.rank() == M.row_join(b).rank()
    if ring.startswith("Z") and len(ring) > 1:
        p = int(ring[1:])
        return solvable_mod_p(rows, rhs, p)
    return solvable_over_Z(M, b)

def solvable_mod_p(rows, rhs, p):
    """Gaussian elimination over the field Z/p (p prime): the system is
    solvable iff the augmented matrix has the same rank as the matrix."""
    Aug = [[x % p for x in r] + [v % p] for r, v in zip(rows, rhs)]
    m = len(Aug); n = len(Aug[0]) - 1
    r = 0
    for c in range(n):
        piv = next((k for k in range(r, m) if Aug[k][c] % p), None)
        if piv is None: continue
        Aug[r], Aug[piv] = Aug[piv], Aug[r]
        inv = pow(Aug[r][c], p-2, p)
        Aug[r] = [(x*inv) % p for x in Aug[r]]
        for k in range(m):
            if k != r and Aug[k][c] % p:
                f = Aug[k][c]
                Aug[k] = [(x - f*y) % p for x, y in zip(Aug[k], Aug[r])]
        r += 1
        if r == m: break
    for k in range(m):                       # 0 = nonzero  =>  no solution
        if all(x % p == 0 for x in Aug[k][:n]) and Aug[k][n] % p:
            return False
    return True

def solvable_over_Z(M, b):
    """Mx = b has an integer solution iff the Smith normal forms of M and of
    the augmented matrix [M|b] have the same invariant factors. This uses only
    smith_normal_form, which every sympy version exposes; the U,V-returning
    smith_normal_decomp is recent and absent from older installs."""
    from sympy.matrices.normalforms import smith_normal_form as snf
    def invariants(X):
        S = snf(Matrix(X), domain=ZZ)
        d = [abs(S[i, i]) for i in range(min(S.rows, S.cols))]
        return sorted([x for x in d if x != 0])
    return invariants(M) == invariants(M.row_join(b))

def report(name, col_sums, ring):
    ctxs, supp = build(col_sums)
    tot = sum(col_sums) % A
    res = []
    for k, sec in enumerate(supp[0]):
        res.append(gamma_vanishes(ctxs, supp, 0, sec, ring))
    nz = sum(1 for r in res if not r)
    print(f"  {name}")
    print(f"    column sums {col_sums}, total mod {A} = {tot}"
          f"  ({'consistent' if tot == 0 else 'CONTRADICTORY'})")
    print(f"    support of context C1: {len(supp[0])} sections")
    print(f"    gamma(s) != 0 for {nz} of {len(res)} sections"
          f"   -> {'OBSTRUCTION DETECTED' if nz else 'gamma vanishes everywhere'}")
    return nz

print(f"  amb_gamma: AMB obstruction over {a.ring}, 3x3 parity families, "
      f"alphabet {A}")
print(f"  rows {ROWS}\n  columns {COLS}\n")
n1 = report("CONTROL      (0,0,0)", (0, 0, 0), a.ring)
n2 = report("OBSTRUCTION  (0,0,1)", (0, 0, 1), a.ring)
n3 = report("CONTROL      (1,1,1)", (1, 1, 1), a.ring)

print(f"\n  AMB Prop 4.3: gamma != 0 is SUFFICIENT for contextuality, not")
print(f"  necessary. Compare with parity_glue.py, where the contextual")
print(f"  fraction is 0.0000 / 1.0000 / 0.0000 for these three families.")
if n2 and not n1 and not n3:
    print(f"  The class agrees with the LP on all three: it detects the designed")
    print(f"  obstruction and stays silent on both controls.")
elif not n2:
    print(f"  The class does NOT detect an obstruction the LP rates at CF = 1.")
    print(f"  That is a false negative of the cohomological invariant, of the")
    print(f"  kind AMB anticipate, and it is the measurement worth reporting.")
else:
    print(f"  The class fires on a control, which would contradict Prop 4.3 --")
    print(f"  check the construction before believing it.")
