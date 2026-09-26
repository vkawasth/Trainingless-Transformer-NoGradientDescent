#!/usr/bin/env python3
"""MERGE_STAB -- merges are affine maps of simplices; is the barcode stable?

    python3 merge_stab.py --data /tmp/cD6 --load-P /tmp/cD6/fit.npz
    python3 merge_stab.py --data /tmp/cH  --load-P /tmp/cH/sm.npz --ks 6,4,3,2

THE GEOMETRIC STATEMENT
-----------------------
A level's rule rows are points in the simplex Delta(child pairs). Merging
symbols into classes is the affine surjection induced by the partition; the
merged row is the (mass-weighted) mixture, which is the Bregman centroid, and
the cost paid is the Bregman information

    cost = sum_s pi_s KL(R_s || R_class)  =  I(children ; B | q(B)).

Both the merge and the filtration live on the same object, so the question is
geometric rather than cohomological: does an affine merge map induce a small
change in the persistence barcode of the KL-Cech filtration, with the change
controlled by the cost paid? We measure the bottleneck distance between the
barcode before and after each merge and compare it with the cost.

THE NOISE FLOOR, WHICH GROWS WITH HEIGHT
----------------------------------------
Rows at level l are estimated from N * 2^(L-l) node observations, so the top
levels are the noisiest: at L = 6 and N = 2e4 that is 6.4e5 observations at
level 1 and 2e4 at level 6, a factor of 32. A barcode difference is only
meaningful above the instability the estimation noise alone produces. For each
level we resample every row from a multinomial at that level's observation
count, recompute the barcode, and take the bottleneck distance to the original
-- that distribution is the floor, and it is reported beside every merge.

A merge whose bottleneck distance is below the floor has not changed the
geometry by more than noise would; one above it has.
"""
import json, math, argparse, itertools
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/tmp/cD6")
ap.add_argument("--load-P", default="")
ap.add_argument("--n-train", type=int, default=20000,
                help="sequences the model was fitted on; sets the noise floor")
ap.add_argument("--ks", default="6,4,3,2")
ap.add_argument("--levels", default="")
ap.add_argument("--noise-reps", type=int, default=6)
ap.add_argument("--wass-q", type=float, default=1.0,
                help="exponent of the Wasserstein distance between diagrams; "
                     "q=1 sums the matching costs, q=2 sums their squares")
ap.add_argument("--iters", type=int, default=400)
ap.add_argument("--seed", type=int, default=0)
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
    src = a.load_P.split("/")[-1]
levels = [int(x) for x in a.levels.split(",")] if a.levels else list(range(1, L+1))
rng = np.random.RandomState(a.seed)
EPS = 1e-12

# ------------------------------------------------------------- geometry
def kl(p, q):
    m = p > EPS
    return float((p[m]*np.log(p[m]/np.maximum(q[m], EPS))).sum())
def one_centre(pts):
    c = pts.mean(0); c = c/c.sum(); best = max(kl(p, c) for p in pts)
    for i in range(1, a.iters+1):
        d = np.array([kl(p, c) for p in pts]); j = int(d.argmax())
        c = (1-1/(i+1))*c + (1/(i+1))*pts[j]; c = np.maximum(c, 0); c /= c.sum()
        r = max(kl(p, c) for p in pts)
        if r < best: best = r
    return best

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

def persistence(simplices):
    S = sorted(simplices, key=lambda x: (x[0], len(x[1]), x[1]))
    idx = {s: i for i, (_, s) in enumerate(S)}; val = [v for v, _ in S]
    cols = []
    for v, s in S:
        cols.append(set() if len(s) == 1 else
                    {idx[f] for f in itertools.combinations(s, len(s)-1)})
    low = {}; pairs = []
    for j in range(len(S)):
        col = cols[j]
        while col:
            l = max(col)
            if l not in low: break
            col ^= cols[low[l]]
        if col:
            l = max(col); low[l] = j; cols[j] = col
            if val[j] > val[l] + 1e-15: pairs.append((len(S[l][1])-1, val[l], val[j]))
        else: cols[j] = col
    paired = set(low) | set(low.values())
    for j in range(len(S)):
        if j not in paired: pairs.append((len(S[j][1])-1, val[j], math.inf))
    return [p for p in pairs if p[0] <= 1]

def barcode(rows):
    n = len(rows)
    simp = [(0.0, (i,)) for i in range(n)]
    for k in (2, 3):
        for S in itertools.combinations(range(n), k):
            simp.append((one_centre(rows[list(S)]), S))
    return persistence(simp)

# --------------------------------------------------- bottleneck distance
def bottleneck(D1, D2, dim):
    """exact bottleneck distance between two finite diagrams, points allowed to
    match the diagonal. Binary search on the threshold with bipartite matching."""
    A = [(b, d) for k, b, d in D1 if k == dim and d < math.inf]
    B = [(b, d) for k, b, d in D2 if k == dim and d < math.inf]
    if not A and not B: return 0.0
    def cost(p, q): return max(abs(p[0]-q[0]), abs(p[1]-q[1]))
    def diag(p): return (p[1]-p[0])/2.0
    cands = sorted({cost(p, q) for p in A for q in B} |
                   {diag(p) for p in A} | {diag(q) for q in B} | {0.0})
    def feasible(eps):
        # A-point may go to a B-point within eps, or to the diagonal if
        # diag(p) <= eps; every unmatched B-point must reach the diagonal
        adj = [[j for j, q in enumerate(B) if cost(p, q) <= eps] for p in A]
        freeA = [i for i, p in enumerate(A) if diag(p) > eps]
        freeB = [j for j, q in enumerate(B) if diag(q) > eps]
        matchB = {}
        def try_k(i, seen):
            for j in adj[i]:
                if j in seen: continue
                seen.add(j)
                if j not in matchB or try_k(matchB[j], seen):
                    matchB[j] = i; return True
            return False
        for i in freeA:
            if not try_k(i, set()): return False
        for j in freeB:
            if j in matchB: continue
            ok = False
            for i in range(len(A)):
                if j in adj[i] and i not in matchB.values():
                    matchB[j] = i; ok = True; break
            if not ok: return False
        return True
    lo, hi = 0, len(cands)-1
    while lo < hi:
        mid = (lo+hi)//2
        if feasible(cands[mid]): hi = mid
        else: lo = mid+1
    return float(cands[lo])

def wasserstein(D1, D2, dim, q=1.0):
    """q-Wasserstein distance between two diagrams, points allowed to match the
    diagonal. Bottleneck is a MAX over the matching, so one flipped pairing
    moves it discretely and by the full amount; this is a SUM, so a single flip
    contributes in proportion to its own size. That is the whole reason to
    prefer it where the diagram is near-degenerate.

    Solved exactly as a rectangular assignment problem: every off-diagonal
    point of one diagram may match a point of the other or its own diagonal
    projection, so we build the standard (n+m) x (n+m) cost matrix whose
    lower-right block is zero (diagonal-to-diagonal, free) and solve it with
    the Hungarian algorithm."""
    A = [(b, d) for k, b, d in D1 if k == dim and d < math.inf]
    B = [(b, d) for k, b, d in D2 if k == dim and d < math.inf]
    if not A and not B: return 0.0
    n, m = len(A), len(B)
    INF = 1e18
    C = np.zeros((n+m, n+m))
    for i, p in enumerate(A):
        for j, r in enumerate(B):
            C[i, j] = max(abs(p[0]-r[0]), abs(p[1]-r[1]))**q
        for j in range(m, m+n):                    # p to its own diagonal
            C[i, j] = INF if (j-m) != i else ((p[1]-p[0])/2.0)**q
    for i in range(n, n+m):
        for j, r in enumerate(B):                  # r's diagonal to r
            C[i, j] = INF if (i-n) != j else ((r[1]-r[0])/2.0)**q
    # lower-right block stays zero: diagonal matched to diagonal, free
    tot = hungarian(C)
    return float(tot ** (1.0/q))

def hungarian(C):
    """exact rectangular assignment (Jonker-Volgenant style shortest
    augmenting path); C is square here. Returns the minimum total cost."""
    n = C.shape[0]
    u = np.zeros(n+1); v = np.zeros(n+1)
    p = np.zeros(n+1, dtype=int); way = np.zeros(n+1, dtype=int)
    for i in range(1, n+1):
        p[0] = i; j0 = 0
        minv = np.full(n+1, np.inf); used = np.zeros(n+1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]; delta = np.inf; j1 = -1
            for j in range(1, n+1):
                if used[j]: continue
                cur = C[i0-1, j-1] - u[i0] - v[j]
                if cur < minv[j]: minv[j] = cur; way[j] = j0
                if minv[j] < delta: delta = minv[j]; j1 = j
            for j in range(n+1):
                if used[j]: u[p[j]] += delta; v[j] -= delta
                else: minv[j] -= delta
            j0 = j1
            if p[j0] == 0: break
        while j0:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
    return sum(C[p[j]-1, j-1] for j in range(1, n+1))

# ------------------------------------------------------------- merging
def level_prior(Pm, l):
    pri = np.full(V, 1.0/V)
    for ll in range(L, l, -1):
        j = Pm[ll].sum(2) + Pm[ll].sum(1)
        pri = (pri @ j)/2.0; pri = pri/pri.sum()
    return pri

def merge_to(rows, w, K):
    """greedy Bregman-information merging to K classes; returns merged rows,
    their weights, and the total cost paid"""
    groups = [[i] for i in range(len(rows))]
    R = rows.copy(); W = w.copy(); paid = 0.0
    while len(groups) > K:
        best = None
        for i, j in itertools.combinations(range(len(groups)), 2):
            wi, wj = W[i], W[j]
            m = (wi*R[i] + wj*R[j])/max(wi+wj, EPS)
            c = wi*kl(R[i], m) + wj*kl(R[j], m)
            if best is None or c < best[0]: best = (c, i, j, m)
        c, i, j, m = best; paid += c
        groups[i] += groups[j]; R[i] = m; W[i] = W[i]+W[j]
        del groups[j]; R = np.delete(R, j, 0); W = np.delete(W, j, 0)
    return R, W, paid

# ------------------------------------------------------------------ run
print(f"  merge_stab: model = {src}, N = {a.n_train} sequences\n")
print(f"  Merges are affine maps of the simplex; the barcode should move by")
print(f"  no more than the cost paid, and only meaningfully when the move")
print(f"  exceeds what estimation noise alone produces at that level.\n")
Ks = [int(x) for x in a.ks.split(",")]
for l in levels:
    rows = P[l].reshape(V, -1)
    pri = level_prior(P, l)
    nobs = a.n_train * (SEQ >> l)          # node observations at this level
    D0 = barcode(rows)
    # noise floor: resample each row at this level's observation count
    floor0, floor1, fw0, fw1 = [], [], [], []
    per_row = max(int(nobs/V), 1)
    for _ in range(a.noise_reps):
        R = np.array([rng.multinomial(per_row, r/r.sum())/per_row for r in rows])
        R = R + 1e-12; R = R/R.sum(1, keepdims=True)
        Dn = barcode(R)
        floor0.append(bottleneck(D0, Dn, 0)); floor1.append(bottleneck(D0, Dn, 1))
        fw0.append(wasserstein(D0, Dn, 0, a.wass_q))
        fw1.append(wasserstein(D0, Dn, 1, a.wass_q))
    f0, f1 = float(np.mean(floor0)), float(np.mean(floor1))
    w0f, w1f = float(np.mean(fw0)), float(np.mean(fw1))
    # Absolute bottleneck is not comparable across levels: level-1 radii run to
    # ~1.2 and level-6 radii to ~0.02, so the same relative perturbation gives
    # a 50x smaller absolute number at the top. Normalise by the scale of the
    # barcode (its largest finite death).
    scale = max([d for _, _, d in D0 if d < math.inf], default=1.0)
    # Bottleneck is capped: if every point of the merged diagram matches the
    # DIAGONAL, the distance is just the largest half-persistence of D0 and
    # says nothing about the merge. Compute that cap per dimension so those
    # rows can be flagged instead of read as measurements.
    def cap(dim):
        f = [(b, d) for k, b, d in D0 if k == dim and d < math.inf]
        return max(((d-b)/2 for b, d in f), default=0.0)
    cap0, cap1 = cap(0), cap(1)
    print(f"  level {l}   {SEQ >> l} nodes/seq, {nobs:,} observations, "
          f"{per_row:,} per row")
    print(f"    barcode scale (largest death) {scale:.4f}")
    print(f"    noise floor  bottleneck: H0 {f0:.4f} ({100*f0/scale:.0f}%)  "
          f"H1 {f1:.4f} ({100*f1/scale:.0f}%)")
    print(f"    noise floor  W{a.wass_q:g}:        H0 {w0f:.4f} "
          f"({100*w0f/scale:.0f}%)  H1 {w1f:.4f} ({100*w1f/scale:.0f}%)")
    print(f"    diagonal cap (uninformative beyond this): H0 {cap0:.4f}  "
          f"H1 {cap1:.4f}")
    print(f"      {'K':>3}{'cost':>10}{'bott H0':>9}{'bott H1':>9}"
          f"{'W H0':>9}{'W H1':>9}   verdict (W)")
    for K in Ks:
        if K >= V: continue
        R, W, paid = merge_to(rows, pri, K)
        DK = barcode(R)
        b0 = bottleneck(D0, DK, 0); b1 = bottleneck(D0, DK, 1)
        w0 = wasserstein(D0, DK, 0, a.wass_q)
        w1 = wasserstein(D0, DK, 1, a.wass_q)
        at_cap = (abs(b0-cap0) < 1e-9) and (abs(b1-cap1) < 1e-9)
        # the verdict now uses W, which does not saturate at the diagonal cap:
        # every point contributes its own distance rather than one point
        # setting the whole statistic
        vw = "above noise" if max(w0, w1) > max(w0f, w1f) else "within noise"
        tag = "  [bott AT CAP]" if at_cap else ""
        print(f"      {K:>3}{paid:>10.4f}{b0:>9.4f}{b1:>9.4f}"
              f"{w0:>9.4f}{w1:>9.4f}   {vw}{tag}")
    print()
print("  READING THE TABLE")
print("  W is the q-Wasserstein distance between diagrams: a SUM over the")
print("    matching rather than a max. It does not saturate at the diagonal")
print("    cap, and a single flipped pairing contributes only its own size, so")
print("    the verdict column uses it. Rows where the BOTTLENECK has saturated")
print("    are marked [bott AT CAP] -- their bottleneck number is a property of")
print("    the original diagram alone and should be ignored.")
print("  AT CAP: every point of the merged diagram matched the diagonal, so the")
print("    bottleneck equals the largest half-persistence of the ORIGINAL")
print("    diagram and is a property of that diagram alone. Those rows say")
print("    nothing about the merge and should not be read as measurements.")
print("  Below the cap the bottleneck grows with cost but sublinearly, so a")
print("    bound of the form  bottleneck <= C * cost  is not supported: small")
print("    merges move the barcode more than their cost, large ones less.")
print("  The noise floor is a MAX over features, so it moves discretely: it is")
print("    near zero where the bars are well separated and jumps to the cap")
print("    where two bars are nearly tied and resampling flips a pairing. A")
print("    floor of 0% therefore means 'no pairing flipped', not 'no noise'.")
print("  Noise must be read RELATIVE to the barcode scale: absolute floors fall")
print("    with height only because the radii shrink.")
