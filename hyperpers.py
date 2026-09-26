#!/usr/bin/env python3
"""HYPERPERS -- Dowker filtration on interaction information.

    python3 hyperpers.py                      # self-tests + synthetic study
    python3 hyperpers.py --data /tmp/cH --load-P /tmp/cH/sm.npz --level 1

WHY NOT A CLIQUE COMPLEX, AND WHY NOT A THRESHOLDED SIMPLICIAL COMPLEX
----------------------------------------------------------------------
Two constructions are ruled out before starting.

(1) The clique complex of a thresholded dependence graph is a Vietoris-Rips
    complex. Edelsbrunner and Wagner show that for Bregman divergences the Rips
    filtration can be arbitrarily far from the true persistence -- pairwise
    overlap does not imply triplewise overlap -- so it cannot be read as an
    approximation to anything. (betti.py is this construction, used as a
    calibrated statistic and nothing more.)

(2) "Include S when |I_{|S|}(S)| >= t" is NOT a simplicial complex. The parity
    coupling is the counterexample, measured on real data:

        |I_3({0,3,5})| = 1.69     while     |I_2({0,3})| = 0.07

    so at t = 0.5 the triple is in and a face is out. Subset closure fails
    generically, because synergy is by definition higher-order structure with
    little lower-order shadow.

WHAT IS BUILT INSTEAD
---------------------
A hypergraph H(t) = { S : |I_{|S|}(S)| >= t }, with no closure requirement, and
then its DOWKER COMPLEX: simplices are sets of vertices contained in a common
hyperedge. This is closed under subsets by construction, so no monotonicity of
I_k is needed, and it is monotone in H, so t |-> D(H(t)) is a genuine
filtration of simplicial complexes.

THE SIGNATURE THIS PRODUCES
---------------------------
In the Dowker complex a hyperedge contributes its FULL simplex. Hence:

    three blocks coupled only pairwise  ->  three edges, no triangle
                                        ->  a hollow cycle, beta_1 = 1
    the same three blocks with genuine
    three-way structure                 ->  the triple enters as a 2-simplex
                                        ->  the cycle is FILLED, beta_1 = 0

So a cycle that dies when a triple enters is evidence OF three-way structure,
and a cycle that survives is pairwise structure with no three-way explanation.
That is the opposite reading from a Rips complex, where triangles are forced by
their edges and can never distinguish the two cases.

The script validates the homology code on hand-built hypergraphs with known
answers, then on synthetic data with planted structure, before touching any
fitted model.
"""
import json, math, argparse, itertools
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="")
ap.add_argument("--load-P", default="")
ap.add_argument("--level", type=int, default=1)
ap.add_argument("--n", type=int, default=40000)
ap.add_argument("--nvar", type=int, default=6, help="variables in the synthetic study")
ap.add_argument("--nsym", type=int, default=4, help="alphabet in the synthetic study")
ap.add_argument("--boot", type=int, default=10)
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--chunk", type=int, default=4096)
a = ap.parse_args()

# ===================================================== homology on a hypergraph
def dowker(hyperedges):
    """Dowker complex of the vertex--hyperedge relation: a set of vertices is a
    simplex iff some hyperedge contains it all. Closed under subsets by
    construction, and monotone in the hyperedge set."""
    verts, edges, tris = set(), set(), set()
    for e in hyperedges:
        e = tuple(sorted(e))
        verts.update(e)
        edges.update(itertools.combinations(e, 2))
        tris.update(itertools.combinations(e, 3))
    return sorted(verts), sorted(edges), sorted(tris)

def rank_gf2(Mx):
    A = (Mx % 2).astype(np.uint8).copy(); r = 0
    rows, cols = A.shape
    for c in range(cols):
        piv = next((rr for rr in range(r, rows) if A[rr, c]), None)
        if piv is None: continue
        A[[r, piv]] = A[[piv, r]]
        sel = A[:, c] == 1; sel[r] = False
        A[sel] ^= A[r]; r += 1
        if r == rows: break
    return r

def betti01(verts, edges, tris):
    if not verts: return 0, 0
    par = {v: v for v in verts}
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for (i, j) in edges:
        ri, rj = find(i), find(j)
        if ri != rj: par[ri] = rj
    b0 = len({find(v) for v in verts})
    cyc = len(edges) - len(verts) + b0          # dim ker d_1
    if not tris or cyc == 0: return b0, cyc
    ei = {e: k for k, e in enumerate(edges)}
    d2 = np.zeros((len(edges), len(tris)), dtype=np.uint8)
    for t, (i, j, k) in enumerate(tris):
        for e in ((i, j), (j, k), (i, k)): d2[ei[e], t] = 1
    return b0, cyc - rank_gf2(d2)

def selftest():
    cases = [
        ("two disjoint edges",        [(0, 1), (2, 3)],                     2, 0),
        ("hollow triangle (3 pairs)", [(0, 1), (1, 2), (0, 2)],             1, 1),
        ("one triple (fills it)",     [(0, 1, 2)],                          1, 0),
        ("pairs + the triple",        [(0, 1), (1, 2), (0, 2), (0, 1, 2)],  1, 0),
        ("4-cycle",                   [(0, 1), (1, 2), (2, 3), (0, 3)],     1, 1),
        ("4-cycle + one diagonal",    [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2)], 1, 2),
        ("two triples sharing an edge", [(0, 1, 2), (1, 2, 3)],             1, 0),
    ]
    print("  SELF-TEST of the Dowker + homology code")
    ok = True
    for name, he, b0x, b1x in cases:
        b0, b1 = betti01(*dowker(he))
        good = (b0, b1) == (b0x, b1x)
        ok &= good
        print(f"    {name:<30} b0={b0} b1={b1}   expected {b0x},{b1x}   "
              f"{'PASS' if good else 'FAIL'}")
    print(f"    -> {'PASS' if ok else 'FAIL'}: note rows 2 and 3 -- the SAME three")
    print("       vertices give a hole when coupled only pairwise and no hole once")
    print("       the triple is present. That is the signature the filtration reads.")
    return ok

# ===================================================== interaction information
def Hc(c):
    p = c / c.sum(); m = p > 0
    return float(-(p[m]*np.log(p[m])).sum())

def weights(codes, V):
    """|I_2| on pairs and |I_3| on triples, Miller-Madow corrected"""
    n, nv = codes.shape
    h1 = {i: Hc(np.bincount(codes[:, i], minlength=V).astype(float))
          for i in range(nv)}
    h2 = {}
    for i, j in itertools.combinations(range(nv), 2):
        h2[(i, j)] = Hc(np.bincount(codes[:, i]*V + codes[:, j],
                                    minlength=V*V).astype(float))
    W = {}
    for i, j in itertools.combinations(range(nv), 2):
        W[(i, j)] = max(h1[i] + h1[j] - h2[(i, j)] - (V-1)**2/(2*n), 0.0)
    for i, j, k in itertools.combinations(range(nv), 3):
        h3 = Hc(np.bincount(codes[:, i]*V*V + codes[:, j]*V + codes[:, k],
                            minlength=V**3).astype(float))
        i3 = (h1[i] + h1[j] + h1[k] - h2[(i, j)] - h2[(i, k)] - h2[(j, k)] + h3)
        W[(i, j, k)] = abs(i3)
    return W

def thresholds(W):
    """every distinct weight, largest first, so that every distinct hypergraph
    in the filtration is visited. A linspace over sorted values undersamples
    the top, where the structure is: in the pairwise-triangle case it admitted
    one of the three planted pairs and then jumped to the noise floor, so the
    hollow triangle never formed."""
    # Use the weights THEMSELVES as thresholds, not rounded copies: rounding
    # can push a threshold just above the weight it came from, so the top
    # hyperedge fails its own ">= t" test and the first row comes out empty.
    v = np.unique(np.array([float(x) for x in W.values()]))
    v = v[v > 0][::-1]
    return v[:a.steps] if len(v) > a.steps else v

def profile(W, thrs):
    out = []
    for t in thrs:
        he = [s for s, v in W.items() if v >= t]
        out.append(betti01(*dowker(he)))
    return out

def report(name, W, Wn, thrs):
    pd_ = profile(W, thrs); pn = [profile(w, thrs) for w in Wn]
    # a threshold is STRUCTURAL if it is above every weight the permutation
    # null produces; below that line the hypergraph is finite-sample noise
    noise = max((max(w.values()) for w in Wn), default=0.0)
    print(f"\n  {name}   (noise floor {noise:.4f})")
    print(f"    {'threshold':>10}{'pairs':>7}{'triples':>9}{'b0':>4}{'b1':>4}"
          f"{'b1 null':>9}{'null max':>10}   region")
    for ti, t in enumerate(thrs):
        he = [s for s, v in W.items() if v >= t]
        np_, nt = sum(1 for s in he if len(s) == 2), sum(1 for s in he if len(s) == 3)
        bn = np.array([pn[r][ti][1] for r in range(len(Wn))])
        b0, b1 = pd_[ti]
        reg = "STRUCTURE" if t > noise else "noise"
        print(f"    {t:>10.4f}{np_:>7}{nt:>9}{b0:>4}{b1:>4}"
              f"{bn.mean():>9.2f}{int(bn.max()):>10}   {reg}")
    return pd_

# ===================================================== synthetic generators
def gen(kind, n, nv, V, rng):
    """simple synthetic data with known structure"""
    X = rng.randint(0, V, size=(n, nv))
    if kind == "independent":
        pass
    elif kind == "pairwise-triangle":
        # 0,1,2 pairwise coupled with NO three-way structure: each pair agrees
        # with probability p, independently imposed -- no variable is a
        # function of the other two.
        for (i, j) in ((0, 1), (1, 2), (0, 2)):
            m = rng.rand(n) < 0.45
            X[m, j] = X[m, i]
    elif kind == "parity":
        # 2 = 0 XOR 1 : the triple is determined, the pairs are near-uniform
        X[:, 2] = np.bitwise_xor(X[:, 0], X[:, 1]) % V
    elif kind == "parity+pair":
        X[:, 2] = np.bitwise_xor(X[:, 0], X[:, 1]) % V
        m = rng.rand(n) < 0.5
        X[m, 4] = X[m, 3]
    return X

def permuted(X, rng):
    """null: permute each variable independently, destroying all dependence
    while preserving every marginal"""
    Y = X.copy()
    for c in range(Y.shape[1]): Y[:, c] = Y[rng.permutation(len(Y)), c]
    return Y

# ===================================================== run
print(f"  hyperpers: Dowker filtration on interaction information\n")
ok = selftest()
if not ok: raise SystemExit("self-test failed; not proceeding to data")

rng = np.random.RandomState(a.seed)
if not a.data:
    n, nv, V = a.n, a.nvar, a.nsym
    print(f"\n  SYNTHETIC STUDY  n={n}, {nv} variables, alphabet {V}, "
          f"{a.boot} permutation replicates")
    for kind in ("independent", "pairwise-triangle", "parity", "parity+pair"):
        X = gen(kind, n, nv, V, rng)
        W = weights(X, V)
        Wn = [weights(permuted(X, rng), V) for _ in range(a.boot)]
        thrs = thresholds(W)
        report(f"{kind}   (variables 0,1,2 carry the planted structure)", W, Wn, thrs)
else:
    M = json.load(open(f"{a.data}/rhm_meta.json"))
    L, V, SEQ, NLEAF = M["depth"], M["nsym"], M["seq_len"], M["nleaf"]
    tr = np.array(json.load(open(f"{a.data}/train_ids.json")), dtype=np.int64)
    X0 = tr[:(len(tr)//SEQ)*SEQ].reshape(-1, SEQ)[:a.n]
    def card(l): return NLEAF if l == 1 else V
    P = {}
    for l in range(1, L+1):
        t = np.zeros((V, card(l), card(l)))
        for s, r in M["rules_all"][str(l)].items():
            for (x, y) in r: t[int(s), x, y] += 1.0/len(r)
        P[l] = t
    if a.load_P:
        z = np.load(a.load_P); P = {l: z[f"P{l}"] for l in range(1, L+1)}
    def up(Pl, lo, hi):
        B, nn, cx = lo.shape; cy = hi.shape[2]; p = Pl.shape[0]
        lo2, hi2 = lo.reshape(-1, cx), hi.reshape(-1, cy)
        Pt = Pl.transpose(1, 0, 2).reshape(cx, p*cy)
        out = np.empty((lo2.shape[0], p))
        for s in range(0, lo2.shape[0], a.chunk):
            t = (lo2[s:s+a.chunk] @ Pt).reshape(-1, p, cy)
            out[s:s+a.chunk] = (t * hi2[s:s+a.chunk, None, :]).sum(-1)
        return out.reshape(B, nn, p)
    o = np.zeros(X0.shape + (NLEAF,)); np.put_along_axis(o, X0[..., None], 1.0, -1)
    t = o
    for l in range(1, a.level+1): t = up(P[l], t[:, 0::2], t[:, 1::2])
    codes = t.argmax(-1)
    W = weights(codes, V)
    Wn = [weights(permuted(codes, rng), V) for _ in range(a.boot)]
    thrs = thresholds(W)
    report(f"level {a.level}: {codes.shape[1]} blocks from "
           f"{'fitted model' if a.load_P else 'true grammar'}", W, Wn, thrs)

print("\n  READING: b1 > 0 means a cycle of pairwise couplings that NO hyperedge")
print("  fills -- pairwise structure with no higher-order explanation. A cycle")
print("  that dies as the threshold falls and a triple enters is the opposite:")
print("  genuine three-way structure. The permutation null gives the level of")
print("  each that finite samples produce on their own.")
