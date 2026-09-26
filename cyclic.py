#!/usr/bin/env python3
"""CYCLIC -- local cohomology on overlapping window covers inside tree data.

    python3 cyclic.py --data /tmp/cH
    python3 cyclic.py --data /tmp/cD6 --width 2 --patch 5 --classes 3

WHY THIS CAN BE NON-TRIVIAL WHERE EVERYTHING ELSE WAS NOT
---------------------------------------------------------
Every previous attempt computed cohomology on a site that forces vanishing:

  the tree poset has a terminal object (the root), so global sections is
  evaluation there, which is exact, so H^k = 0 for k > 0 with ANY coefficients;

  the information site gives I_k as a (k-1)-coboundary, so its classes vanish
  whatever the data;

  a single tree is an acyclic hypergraph, so Vorob'ev makes every consistent
  family extend.

The common cause is acyclicity. The cure is not a different coefficient system
but a different COVER. Take a patch of P consecutive leaf positions and cover
it with overlapping windows

    W_i = { i, i+1, ..., i+w-1 }   (mod P),    i = 0, ..., P-1

which wrap around. For w >= 2 and P >= 3 the windows form a cycle: consecutive
ones share w-1 positions and the last meets the first. That hypergraph is NOT
acyclic -- it fails the running intersection property -- so Vorob'ev gives no
extension guarantee and the obstruction is free to be non-zero. This is the
n-cycle measurement scenario, and it lives entirely inside the tree: no global
sections, no root, just a patch.

WHAT IS COMPUTED, ON REAL CORPUS DATA
-------------------------------------
  support    the possibilistic model: which window assignments actually occur
             in the corpus. The RHM grammar is sparse, so most assignments do
             not occur and the support is a proper subset -- which is what
             makes a possibilistic invariant have anything to read.
  gamma      AMB Proposition 4.2 over Z: is there a family of Z-linear
             combinations of support sections agreeing on all overlaps, with
             the chosen section pinned?
  CF         contextual fraction by linear program over the patch's c^P
             outcomes -- small enough to solve exactly.

THREE CONTROLS
--------------
  acyclic cover     the same patch covered by DISJOINT windows. Vorob'ev then
                    guarantees extension, so gamma must vanish and CF must be
                    0. If not, the code is wrong.
  shuffled data     each position permuted independently, destroying the
                    grammar while preserving the marginals.
  full support      a patch where every assignment occurs, where gamma must
                    vanish for the same reason smoothing killed it before.
"""
import json, argparse, itertools
import numpy as np
from sympy import Matrix, ZZ
from sympy.matrices.normalforms import smith_normal_form as snf
from scipy.optimize import linprog

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/tmp/cH")
ap.add_argument("--patch", type=int, default=5, help="consecutive positions")
ap.add_argument("--width", type=int, default=2, help="window width")
ap.add_argument("--classes", type=int, default=3,
                help="leaf values are coarse-grained to this many classes, so "
                     "the patch has c^P outcomes and the LP stays exact")
ap.add_argument("--starts", default="", help="patch start positions")
ap.add_argument("--n", type=int, default=200000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

M = json.load(open(f"{a.data}/rhm_meta.json"))
SEQ, NLEAF = M["seq_len"], M["nleaf"]
tr = np.array(json.load(open(f"{a.data}/train_ids.json")), dtype=np.int64)
X = tr[:(len(tr)//SEQ)*SEQ].reshape(-1, SEQ)[:a.n]
C = a.classes
P, W = a.patch, a.width
rng = np.random.RandomState(a.seed)

def windows(cyclic):
    if cyclic:
        return [tuple((i+k) % P for k in range(W)) for i in range(P)]
    return [tuple(range(i, min(i+W, P))) for i in range(0, P, W)]

def is_acyclic(ctxs):
    """running intersection / graham reduction: repeatedly delete a vertex in
    only one context, then a context contained in another. Acyclic iff this
    empties the hypergraph."""
    S = [set(c) for c in ctxs]
    changed = True
    while changed:
        changed = False
        cnt = {}
        for s in S:
            for v in s: cnt[v] = cnt.get(v, 0)+1
        for s in S:
            ear = {v for v in s if cnt[v] == 1}
            if ear:
                s -= ear; changed = True
        S2 = [s for s in S if s]
        for i, s in enumerate(S2):
            if any(i != j and s <= t for j, t in enumerate(S2)):
                S2.pop(i); changed = True; break
        S = S2
    return len(S) == 0

def patch_data(start, shuffle=False):
    pos = [(start+k) % SEQ for k in range(P)]
    D = (X[:, pos] % C)
    if shuffle:
        D = D.copy()
        for k in range(P): D[:, k] = D[rng.permutation(len(D)), k]
    return D

def supports(D, ctxs):
    out = []
    for c in ctxs:
        seen = set(map(tuple, D[:, list(c)].tolist()))
        out.append(sorted(seen))
    return out

def gamma_nonzero_count(ctxs, supp):
    """AMB Prop 4.2 over Z, for every section of the first context"""
    offs, n = [], 0
    for S in supp: offs.append(n); n += len(S)
    base = []
    for i, Ci in enumerate(ctxs):
        for j, Cj in enumerate(ctxs):
            if j <= i: continue
            sh = sorted(set(Ci) & set(Cj))
            if not sh: continue
            li = [Ci.index(x) for x in sh]; lj = [Cj.index(x) for x in sh]
            for v in itertools.product(range(C), repeat=len(sh)):
                r = [0]*n
                for k, t in enumerate(supp[i]):
                    if tuple(t[p] for p in li) == v: r[offs[i]+k] += 1
                for k, t in enumerate(supp[j]):
                    if tuple(t[p] for p in lj) == v: r[offs[j]+k] -= 1
                base.append(r)
    def invariants(Mx):
        S = snf(Matrix(Mx), domain=ZZ)
        d = [abs(S[i, i]) for i in range(min(S.rows, S.cols))]
        return sorted([x for x in d if x != 0])
    nz = 0
    for k0 in range(len(supp[0])):
        rows = [r[:] for r in base]; rhs = [0]*len(base)
        for k in range(len(supp[0])):
            r = [0]*n; r[offs[0]+k] = 1
            rows.append(r); rhs.append(1 if k == k0 else 0)
        Mm = Matrix(rows); bb = Matrix(rhs)
        if invariants(Mm) != invariants(Mm.row_join(bb)): nz += 1
    return nz, len(supp[0])

def cf(D, ctxs):
    """contextual fraction over the patch: max weight of the family carried by
    a global distribution on C^P, from the empirical window tables"""
    OUT = np.array(list(itertools.product(range(C), repeat=P)), dtype=np.int64)
    rows, b = [], []
    for c in ctxs:
        idx = np.zeros(len(D), dtype=np.int64); code = np.zeros(len(OUT), dtype=np.int64)
        for k, p in enumerate(c):
            idx = idx*C + D[:, p]; code = code*C + OUT[:, p]
        t = np.bincount(idx, minlength=C**len(c)).astype(float); t /= t.sum()
        for v in range(C**len(c)):
            rows.append((code == v).astype(float)); b.append(t[v])
    r = linprog(-np.ones(len(OUT)), A_ub=np.array(rows), b_ub=np.array(b),
                bounds=(0, None), method="highs")
    return max(0.0, 1.0 - (-r.fun if r.status == 0 else 0.0))

starts = ([int(x) for x in a.starts.split(",")] if a.starts
          else [0, 1, SEQ//2 - 1])
cyc, acy = windows(True), windows(False)
print(f"  cyclic: corpus {a.data}, {len(X)} sequences, patch of {P} positions,")
print(f"  windows of width {W}, leaf values coarse-grained to {C} classes")
print(f"    cyclic cover  {cyc}   acyclic by Graham reduction: {is_acyclic(cyc)}")
print(f"    disjoint cover{acy}   acyclic by Graham reduction: {is_acyclic(acy)}")
print(f"  A cover that is NOT acyclic is the precondition for a non-zero class;")
print(f"  Vorob'ev forces vanishing on the disjoint one.\n")
print(f"    {'start':>6}{'cover':>10}{'data':>10}{'|supp C1|':>11}"
      f"{'gamma!=0':>10}{'CF':>9}")
for st in starts:
    for tag, ctxs in (("cyclic", cyc), ("disjoint", acy)):
        for dtag, sh in (("corpus", False), ("shuffled", True)):
            D = patch_data(st, sh)
            supp = supports(D, ctxs)
            nz, tot = gamma_nonzero_count(ctxs, supp)
            print(f"    {st:>6}{tag:>10}{dtag:>10}{tot:>11}"
                  f"{f'{nz}/{tot}':>10}{cf(D, ctxs):>9.4f}")
print(f"\n  gamma != 0 requires a cyclic cover AND a support sparse enough that")
print(f"  the pinned section cannot be completed. CF is the graded companion:")
print(f"  it is a linear program on the patch and needs no support sparsity.")
