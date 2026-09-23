#!/usr/bin/env python3
"""BETTI -- is the block dependency structure a tree, or does it have cycles?

    python3 betti.py --data /tmp/cH --load-P /tmp/cH/fit.npz --level 1
    python3 betti.py --data /tmp/cH --load-P /tmp/cH/fit.npz --level 1 --plant 1,14,0.3

THE PREDICTION BEING TESTED
---------------------------
Decode each block of 2^l leaves to its level-l symbol, then measure mutual
information between every PAIR of blocks. Under a tree, the dependency
structure is nested: blocks under a common level-(l+1) parent are the most
dependent, then blocks under a common level-(l+2) ancestor, and so on. Every
threshold therefore cuts the graph into a union of CLIQUES (each clique = the
blocks under one ancestor), and a clique complex of cliques has no holes:

    tree  =>  beta_1 = 0 at every threshold.

A dependency that the tree cannot carry -- two blocks coupled without a short
tree path -- adds an edge across cliques, creating a cycle that no triangle
fills:

    non-tree  =>  beta_1 >= 1 over a range of thresholds.

beta_1 is computed exactly on the clique complex (2-skeleton is enough):
    beta_1 = dim ker d_1 - rank d_2 = (E - V + C) - rank d_2   over GF(2).

CONTROLS
  null band : beta_1 of datasets SAMPLED FROM THE FITTED TREE, same size. A
              tree model can still show spurious cycles from finite-sample
              noise; the null band says how many.
  true data : the corpus as generated (a tree) -- should sit inside the band.
  planted   : --plant i,j,q copies leaf i into leaf j, coupling two blocks
              with no short tree path. Should leave the band.

DISJOINTNESS: blocks share no leaves, and each is decoded from its own inside
vector only, so no block sees its neighbour's evidence.
"""
import json, math, argparse, collections
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/tmp/cH")
ap.add_argument("--load-P", default="", help="npz of fitted rule tables; omit to use the true grammar")
ap.add_argument("--decoder", default="fitted", choices=["fitted", "true"])
ap.add_argument("--null", default="fitted", choices=["fitted", "true"],
                help="which model generates the null band. 'fitted' asks "
                     "whether the data look non-tree RELATIVE TO THE FIT, and "
                     "will flag wherever the fit under-produces dependence. "
                     "'true' asks whether the DATA are a tree at all, and is "
                     "the control that isolates non-tree structure.")
ap.add_argument("--level", type=int, default=1)
ap.add_argument("--n", type=int, default=40000, help="sequences per dataset")
ap.add_argument("--boot", type=int, default=20)
ap.add_argument("--plant", default="",
                help="i,j,q[;i,j,q...] copy leaf i into leaf j w.p. q. NOTE a "
                     "SINGLE extra coupling is a bridge between two cliques and "
                     "creates no cycle; beta_1 counts independent loops, so two "
                     "couplings between distinct clique pairs are needed.")
ap.add_argument("--plant-block", default="",
                help="bi,bj,q[;...] copy the LEAVES of block bi onto block bj "
                     "w.p. q. Unlike copying a single leaf this keeps every "
                     "level-l production grammatical, so the decoder still "
                     "works and the coupling shows up as block dependence.")
ap.add_argument("--show-mi", action="store_true", help="print the block MI matrix")
ap.add_argument("--steps", type=int, default=25, help="thresholds in the sweep")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--chunk", type=int, default=4096)
a = ap.parse_args()

M = json.load(open(f"{a.data}/rhm_meta.json"))
L, V, SEQ, NLEAF = M["depth"], M["nsym"], M["seq_len"], M["nleaf"]
tr = np.array(json.load(open(f"{a.data}/train_ids.json")), dtype=np.int64)
X0 = tr[:(len(tr)//SEQ)*SEQ].reshape(-1, SEQ)[:a.n]
def card(l): return NLEAF if l == 1 else V

PT = {}
for l in range(1, L+1):
    t = np.zeros((V, card(l), card(l)))
    for s, rules in M["rules_all"][str(l)].items():
        for (x, y) in rules: t[int(s), x, y] += 1.0 / len(rules)
    PT[l] = t
if a.load_P:
    z = np.load(a.load_P); PF = {l: z[f"P{l}"] for l in range(1, L+1)}
else:
    PF = PT
PDEC = PF if a.decoder == "fitted" else PT     # decoder
PGEN = PF if a.null == "fitted" else PT        # model the null is sampled from

if a.plant:
    g = np.random.RandomState(31337); X0 = X0.copy()
    for spec in a.plant.split(";"):
        i_, j_, q_ = spec.split(","); pi, pj, pq = int(i_), int(j_), float(q_)
        m = g.rand(len(X0)) < pq
        X0[m, pj] = X0[m, pi]
        print(f"  PLANTED: leaf {pi} -> leaf {pj} w.p. {pq}  "
              f"(couples blocks {pi >> a.level} and {pj >> a.level})")

if a.plant_block:
    g = np.random.RandomState(4242); X0 = X0.copy(); w = 2 ** a.level
    for spec in a.plant_block.split(";"):
        bi, bj, q = spec.split(","); bi, bj, q = int(bi), int(bj), float(q)
        m = g.rand(len(X0)) < q
        X0[np.ix_(m, range(bj*w, (bj+1)*w))] = X0[np.ix_(m, range(bi*w, (bi+1)*w))]
        print(f"  PLANTED BLOCK COPY: block {bi} -> block {bj} w.p. {q}")

# ------------------------------------------------------------------ inside
def up(P, lo, hi):
    B, n, cx = lo.shape; cy = hi.shape[2]; p = P.shape[0]
    lo2, hi2 = lo.reshape(-1, cx), hi.reshape(-1, cy)
    Pt = P.transpose(1, 0, 2).reshape(cx, p * cy)
    out = np.empty((lo2.shape[0], p))
    for s in range(0, lo2.shape[0], a.chunk):
        t = (lo2[s:s+a.chunk] @ Pt).reshape(-1, p, cy)
        out[s:s+a.chunk] = (t * hi2[s:s+a.chunk, None, :]).sum(-1)
    return out.reshape(B, n, p)
def onehot(X):
    o = np.zeros(X.shape + (NLEAF,)); np.put_along_axis(o, X[..., None], 1.0, -1)
    return o
def inside(lf, P):
    tabs = [lf]
    for l in range(1, L+1):
        tabs.append(up(P[l], tabs[-1][:, 0::2], tabs[-1][:, 1::2]))
    return tabs
def sample_model(Pm, n, g):
    cur = g.randint(0, V, size=(n, 1))
    for l in range(L, 0, -1):
        c = card(l); D = c * c
        cdf = np.cumsum(Pm[l].reshape(V, D), 1); cdf[:, -1] = 1.0
        u = g.rand(*cur.shape); pick = np.zeros(cur.shape, dtype=np.int64)
        for s in range(V):
            m = cur == s
            if m.any(): pick[m] = np.searchsorted(cdf[s], u[m], side="right")
        pick = np.minimum(pick, D - 1)
        nxt = np.empty((n, cur.shape[1]*2), dtype=np.int64)
        nxt[:, 0::2] = pick // c; nxt[:, 1::2] = pick % c
        cur = nxt
    return cur

# ------------------------------------------------------------- MI and betti
def mi_mm(A, B_, K):
    J = np.bincount(A * K + B_, minlength=K*K).reshape(K, K).astype(float)
    J = J / J.sum(); px, py = J.sum(1), J.sum(0); m = J > 0
    val = float((J[m] * np.log(J[m] / np.outer(px, py)[m])).sum())
    kx, ky = (px > 0).sum(), (py > 0).sum()
    return max(val - (kx - 1) * (ky - 1) / (2 * len(A)), 0.0)

def block_mi(X):
    d = inside(onehot(X), PDEC)[a.level].argmax(-1)
    nb = d.shape[1]
    Wm = np.zeros((nb, nb))
    for i in range(nb):
        for j in range(i+1, nb):
            Wm[i, j] = Wm[j, i] = mi_mm(d[:, i], d[:, j], V)
    return Wm

def rank_gf2(Mx):
    A = (Mx % 2).astype(np.uint8).copy(); r = 0
    rows, cols = A.shape
    for c in range(cols):
        piv = None
        for rr in range(r, rows):
            if A[rr, c]: piv = rr; break
        if piv is None: continue
        A[[r, piv]] = A[[piv, r]]
        sel = (A[:, c] == 1); sel[r] = False
        A[sel] ^= A[r]
        r += 1
        if r == rows: break
    return r

def betti1(Wm, thr):
    nb = Wm.shape[0]
    E = [(i, j) for i in range(nb) for j in range(i+1, nb) if Wm[i, j] >= thr]
    if not E: return 0
    eidx = {e: k for k, e in enumerate(E)}
    Eset = set(E)
    # components of the 1-skeleton
    par = list(range(nb))
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    verts = set()
    for (i, j) in E:
        verts.add(i); verts.add(j)
        ri, rj = find(i), find(j)
        if ri != rj: par[ri] = rj
    Cn = len({find(v) for v in verts})
    cyc = len(E) - len(verts) + Cn                     # dim ker d_1
    tri = [(i, j, k) for i in range(nb) for j in range(i+1, nb)
           for k in range(j+1, nb)
           if (i, j) in Eset and (j, k) in Eset and (i, k) in Eset]
    if not tri: return cyc
    d2 = np.zeros((len(E), len(tri)), dtype=np.uint8)
    for t, (i, j, k) in enumerate(tri):
        for e in ((i, j), (j, k), (i, k)): d2[eidx[e], t] = 1
    return cyc - rank_gf2(d2)

# ------------------------------------------------------------------- run
nb = SEQ >> a.level
print(f"  level {a.level}: {nb} blocks of {2**a.level} leaves, {V} symbols, "
      f"{len(X0)} sequences, decoder = {a.decoder}, null from = {a.null}")
Wdata = block_mi(X0)
if a.show_mi:
    print("\n    block MI (data):")
    for r in Wdata: print("      " + " ".join(f"{v:7.4f}" for v in r))
g = np.random.RandomState(a.seed)
Wnull = [block_mi(sample_model(PGEN, len(X0), g)) for _ in range(a.boot)]
# Threshold at the DATA's own edge weights, so every distinct subgraph in the
# filtration is visited. A quantile grid over pooled weights can skip the
# window where a cycle exists -- with nb=4 there are only 6 edges and the grid
# jumped straight past the 4-edge graph.
vals = np.unique(Wdata[np.triu_indices(nb, 1)])
vals = vals[vals > 0]
if len(vals) > a.steps:
    vals = vals[np.linspace(0, len(vals) - 1, a.steps).astype(int)]
thrs = vals
print(f"\n    {'threshold':>11}{'edges':>7}{'b1 data':>9}"
      f"{'b1 null mean':>14}{'null max':>10}   verdict")
flagged = 0
for t in thrs:
    bd = betti1(Wdata, t)
    bn = np.array([betti1(w, t) for w in Wnull])
    nE = int((Wdata[np.triu_indices(nb, 1)] >= t).sum())
    v = ""
    if bd > bn.max():
        v = "  <== CYCLE beyond the tree null"; flagged += 1
    print(f"    {t:>11.5f}{nE:>7}{bd:>9}{bn.mean():>14.2f}{int(bn.max()):>10}{v}")
print(f"\n    thresholds with beta_1 above the tree null: {flagged} of {len(thrs)}")
print("    0 of N  => block dependence is nested cliques: consistent with a tree")
print("    several => a cycle no triangle fills: the blocks are NOT organised by")
print("               this tree, and a complex (not a tree) is the right structure")
