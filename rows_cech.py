#!/usr/bin/env python3
"""ROWS_CECH -- exact Bregman (KL) Cech filtration on the rule rows.

    python3 rows_cech.py --data /tmp/cH --load-P /tmp/cH/sm.npz
    python3 rows_cech.py --data /tmp/cH --load-P /tmp/cH/sm.npz --tucker 2,3,4

WHERE THIS CONSTRUCTION IS LEGITIMATE
-------------------------------------
betti.py and hyperpers.py have no ambient point set: their vertices are blocks
and their parameter is a threshold on mutual information, so Bregman balls are
undefined there. The RULE ROWS are different. Row s of level l is a probability
distribution over child pairs, so the points live in a simplex, the natural
dissimilarity is KL, and KL is the Bregman divergence of negative entropy.
Edelsbrunner and Wagner's results then apply verbatim: KL restricted to the
simplex is of Legendre type, the Legendre map sends primal balls to convex dual
balls, common intersections of primal balls are empty or contractible, and the
Nerve Theorem gives Cech the homotopy type of the union of balls.

WHAT IS COMPUTED
----------------
Radius Function Lemma (i): the Cech radius of a simplex P is the radius of the
SMALLEST ENCLOSING DUAL BALL of P, i.e.

    rho(P) = min_c max_{p in P} D_KL(p || c).

We compute it exactly (to tolerance) with the Badoiu-Clarkson style iteration
for the Bregman 1-centre. With V = 8 rows there are 8 + 28 + 56 = 92 subsets up
to size three, so the whole filtration is computed without approximation.

THREE MATH CHECKS, ALL RUN BEFORE ANY CONCLUSION
------------------------------------------------
  C1  monotonicity: rho(face) <= rho(coface), so the radius function really
      defines a filtration of simplicial complexes.
  C2  Rips <= Cech, with equality on edges by definition; the interesting
      quantity is the largest gap on triples. Edelsbrunner and Wagner's
      Result 2 says the gap can be arbitrarily large for Bregman divergences
      -- here we measure it.
  C3  the Cech 0-dimensional merge order against greedy KL agglomeration
      (the V4 merge order), which should agree if the barcode is reading the
      same cluster structure.

TUCKER
------
--tucker R compresses each rule tensor to a rank-R core (HOSVD on the three
modes, clipped non-negative, rows renormalised) and repeats the whole analysis
on the compressed rows, reporting the KL reconstruction error and whether the
barcode survives. This is the measurement the V^3 -> R^3 scaling argument needs
and does not currently have.
"""
import json, math, argparse, itertools
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/tmp/cH")
ap.add_argument("--load-P", default="")
ap.add_argument("--levels", default="", help="default: all")
ap.add_argument("--tucker", default="", help="comma-separated ranks to test")
ap.add_argument("--iters", type=int, default=4000, help="1-centre iterations")
ap.add_argument("--tol", type=float, default=1e-12)
a = ap.parse_args()

M = json.load(open(f"{a.data}/rhm_meta.json"))
L, V, SEQ, NLEAF = M["depth"], M["nsym"], M["seq_len"], M["nleaf"]
def card(l): return NLEAF if l == 1 else V
P = {}
for l in range(1, L+1):
    t = np.zeros((V, card(l), card(l)))
    for s, r in M["rules_all"][str(l)].items():
        for (x, y) in r: t[int(s), x, y] += 1.0/len(r)
    P[l] = t
src = "true grammar"
if a.load_P:
    z = np.load(a.load_P); P = {l: z[f"P{l}"] for l in range(1, L+1)}
    src = a.load_P
levels = [int(x) for x in a.levels.split(",")] if a.levels else list(range(1, L+1))
EPS = 1e-12

# ------------------------------------------------- KL and the Bregman 1-centre
def kl(p, q):
    p = np.maximum(p, 0); m = p > EPS
    return float((p[m]*np.log(p[m]/np.maximum(q[m], EPS))).sum())

def one_centre(pts, iters, tol):
    """smallest enclosing DUAL KL ball: min_c max_p D(p||c).
    Badoiu-Clarkson iteration, which for a Bregman divergence moves the centre
    toward the farthest point with step 1/(i+1); the right-type Bregman
    centroid of a set is its arithmetic mean, which is what makes the convex
    combination the correct update."""
    c = pts.mean(0); c = c / c.sum()
    best = max(kl(p, c) for p in pts)
    for i in range(1, iters+1):
        d = np.array([kl(p, c) for p in pts])
        j = int(d.argmax())
        if d[j] - d.min() < tol: break
        c = (1.0 - 1.0/(i+1))*c + (1.0/(i+1))*pts[j]
        c = np.maximum(c, 0); c = c / c.sum()
        r = max(kl(p, c) for p in pts)
        if r < best: best = r
    return best

def radii(rows, iters, tol):
    """Cech radius of every subset of size 1, 2, 3"""
    n = len(rows)
    rho = {}
    for k in (1, 2, 3):
        for S in itertools.combinations(range(n), k):
            rho[S] = 0.0 if k == 1 else one_centre(rows[list(S)], iters, tol)
    return rho

# ------------------------------------------------------------- homology
def rank_gf2(Mx):
    A = (Mx % 2).astype(np.uint8).copy(); r = 0
    for c in range(A.shape[1]):
        piv = next((rr for rr in range(r, A.shape[0]) if A[rr, c]), None)
        if piv is None: continue
        A[[r, piv]] = A[[piv, r]]
        sel = A[:, c] == 1; sel[r] = False
        A[sel] ^= A[r]; r += 1
        if r == A.shape[0]: break
    return r

def betti01(n, edges, tris):
    par = list(range(n))
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for (i, j) in edges:
        ri, rj = find(i), find(j)
        if ri != rj: par[ri] = rj
    b0 = len({find(v) for v in range(n)})
    cyc = len(edges) - n + b0
    if not tris or cyc == 0: return b0, cyc
    ei = {e: k for k, e in enumerate(edges)}
    d2 = np.zeros((len(edges), len(tris)), dtype=np.uint8)
    for t, (i, j, k) in enumerate(tris):
        for e in ((i, j), (j, k), (i, k)):
            if e in ei: d2[ei[e], t] = 1
    return b0, cyc - rank_gf2(d2)

def filtration(n, rho, rips=False):
    """(threshold, b0, b1) at every distinct radius"""
    ed = {S: v for S, v in rho.items() if len(S) == 2}
    if rips:
        tr = {S: max(ed[(S[0], S[1])], ed[(S[0], S[2])], ed[(S[1], S[2])])
              for S in rho if len(S) == 3}
    else:
        tr = {S: v for S, v in rho.items() if len(S) == 3}
    ts = np.unique(np.array(sorted(set(ed.values()) | set(tr.values()))))
    out = []
    for t in ts:
        E = sorted(S for S, v in ed.items() if v <= t)
        T = sorted(S for S, v in tr.items() if v <= t)
        out.append((float(t), *betti01(n, E, T), len(E), len(T)))
    return out

def merge_order_cech(n, rho):
    """0-dimensional merge order: the sequence of edges that join components"""
    ed = sorted(((v, S) for S, v in rho.items() if len(S) == 2))
    par = list(range(n)); order = []
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for v, (i, j) in ed:
        ri, rj = find(i), find(j)
        if ri != rj: par[ri] = rj; order.append(((i, j), v))
    return order

def merge_order_kl(rows):
    """greedy agglomeration by the V4 cost: merging two clusters into their
    mixture costs n_a KL(R_a||R_m) + n_b KL(R_b||R_m) with equal weights"""
    groups = [[i] for i in range(len(rows))]
    cent = rows.copy(); order = []
    while len(groups) > 1:
        best = None
        for i, j in itertools.combinations(range(len(groups)), 2):
            wa, wb = len(groups[i]), len(groups[j])
            m = (wa*cent[i] + wb*cent[j])/(wa+wb)
            c = wa*kl(cent[i], m) + wb*kl(cent[j], m)
            if best is None or c < best[0]: best = (c, i, j, m)
        c, i, j, m = best
        order.append((tuple(sorted(groups[i][:1] + groups[j][:1])), c))
        groups[i] = groups[i] + groups[j]; cent[i] = m
        del groups[j]; cent = np.delete(cent, j, 0)
    return order

# ------------------------------------------------------------- Tucker
def tucker(T, R):
    """rank-(R,R,R) HOSVD, clipped non-negative, rows renormalised so the
    result is still a valid rule tensor"""
    facs = []
    for mode in range(3):
        Un = np.moveaxis(T, mode, 0).reshape(T.shape[mode], -1)
        U, _, _ = np.linalg.svd(Un, full_matrices=False)
        facs.append(U[:, :min(R, U.shape[1])])
    G = T
    for mode, U in enumerate(facs):
        G = np.moveaxis(np.tensordot(U.T, np.moveaxis(G, mode, 0), axes=1), 0, mode)
    Tr = G
    for mode, U in enumerate(facs):
        Tr = np.moveaxis(np.tensordot(U, np.moveaxis(Tr, mode, 0), axes=1), 0, mode)
    Tr = np.maximum(Tr, 0) + 1e-12
    return Tr / Tr.sum((1, 2), keepdims=True)

# ------------------------------------------------------------------ run
print(f"  rows_cech: exact Bregman (KL) Cech filtration on rule rows")
print(f"  model = {src},  V = {V} rows per level\n")

def analyse(tag, rows):
    n = len(rows)
    rho = radii(rows, a.iters, a.tol)
    # C1 monotonicity
    bad = 0; worst = 0.0
    for S, v in rho.items():
        if len(S) < 2: continue
        for f in itertools.combinations(S, len(S)-1):
            if rho[f] > v + 1e-9:
                bad += 1; worst = max(worst, rho[f]-v)
    # C2 Rips vs Cech on triples
    ed = {S: v for S, v in rho.items() if len(S) == 2}
    gaps = []
    for S, v in rho.items():
        if len(S) != 3: continue
        rp = max(ed[(S[0], S[1])], ed[(S[0], S[2])], ed[(S[1], S[2])])
        gaps.append((v - rp, S, rp, v))
    gaps.sort(reverse=True)
    fc = filtration(n, rho); fr = filtration(n, rho, rips=True)
    b1c = max(x[2] for x in fc); b1r = max(x[2] for x in fr)
    mo_c = [e for e, _ in merge_order_cech(n, rho)]
    mo_k = [e for e, _ in merge_order_kl(rows)]
    agree = sum(1 for x, y in zip(mo_c, mo_k) if x == y)
    print(f"  {tag}")
    print(f"    C1 monotonicity rho(face) <= rho(coface): "
          f"{'PASS' if bad == 0 else f'FAIL ({bad} violations, worst {worst:.2e})'}")
    print(f"    C2 Rips vs Cech on triples: largest gap {gaps[0][0]:+.4f} "
          f"(Rips {gaps[0][2]:.4f} vs Cech {gaps[0][3]:.4f} on {gaps[0][1]}), "
          f"mean {np.mean([g[0] for g in gaps]):+.4f}")
    print(f"       max b1: Cech {b1c}, Rips {b1r}"
          f"   {'-- SAME' if b1c == b1r else '-- DIFFERENT, Rips is not a substitute'}")
    print(f"    C3 merge order, Cech vs greedy-KL: {agree}/{len(mo_c)} agree")
    print(f"    barcode (b0 over the filtration): "
          + " ".join(str(x[1]) for x in fc[:12]) + (" ..." if len(fc) > 12 else ""))
    return rho, fc

for l in levels:
    rows = P[l].reshape(V, -1)
    rho, fc = analyse(f"level {l}  ({rows.shape[1]} cells per row)", rows)
    if a.tucker:
        for R in (int(x) for x in a.tucker.split(",")):
            Tr = tucker(P[l], R)
            err = float(np.mean([kl(P[l][s].ravel(), Tr[s].ravel()) for s in range(V)]))
            print(f"    -- Tucker rank {R}: mean KL(row || compressed) = {err:.5f}, "
                  f"params {R**3 + R*(V + 2*card(l))} vs {V*card(l)**2}")
            analyse(f"   level {l}, Tucker rank {R}", Tr.reshape(V, -1))
    print()
