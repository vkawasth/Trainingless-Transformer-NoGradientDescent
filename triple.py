#!/usr/bin/env python3
"""TRIPLE -- three-way dependence between blocks, which no pairwise test sees.

    python3 triple.py --data /tmp/cH --load-P /tmp/cH/sm.npz --level 1
    python3 triple.py --data /tmp/cH --load-P /tmp/cH/sm.npz --level 1 --parity 0,3,5

WHY
---
Every instrument built so far is pairwise: V6 compares leaf pairs, V7 compares
block pairs, and betti.py takes the clique complex of a thresholded graph --
which is a Vietoris-Rips complex, because pairwise numbers are the only thing
that can decide its simplices. Edelsbrunner and Wagner show that for Bregman
divergences the Rips filtration can be arbitrarily far from the true (Cech)
persistence: pairwise overlap does not imply triplewise overlap. The same gap
appears here in probabilistic form.

The sharp case is a PARITY coupling. Make three blocks satisfy
    s_c = s_a XOR s_b      (in some labelling of the symbols)
and every pairwise mutual information is exactly zero, while the triple is
completely determined. V6, V7 and beta_1 all report "tree"; the data are
grossly non-tree. --parity plants exactly this.

WHAT IS MEASURED
----------------
For each triple of blocks (a,b,c), the interaction information

    I(A;B;C) = I(A;B) - I(A;B | C)

which is symmetric in the three arguments, is zero when no three-way structure
is present, and is NEGATIVE for parity-like (synergistic) coupling. It is
compared against datasets sampled from the fitted model, exactly as V6/V7 do,
so the plug-in bias cancels and the null is calibrated.

DISJOINTNESS holds as before: blocks share no leaves and each is decoded from
its own inside vector only.

CAVEAT ON ESTIMATION
--------------------
I(A;B;C) is a difference of entropies on a V^3 table. At V=8 that is 512 cells
and n = 40000 is comfortable; at V=64 it is 262144 cells and the plug-in
estimate is dominated by bias. The bootstrap keeps the test VALID at any size
(the same estimator is applied to data and null), but the POWER degrades.
"""
import json, math, argparse, itertools
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/tmp/cH")
ap.add_argument("--load-P", default="")
ap.add_argument("--level", type=int, default=1)
ap.add_argument("--n", type=int, default=40000)
ap.add_argument("--boot", type=int, default=20)
ap.add_argument("--parity", default="", help="a,b,c: plant s_c = s_a XOR s_b")
ap.add_argument("--z-flag", type=float, default=4.0)
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
    for s, r in M["rules_all"][str(l)].items():
        for (x, y) in r: t[int(s), x, y] += 1.0/len(r)
    PT[l] = t
P = dict(PT)
if a.load_P:
    z = np.load(a.load_P); P = {l: z[f"P{l}"] for l in range(1, L+1)}

def up(Pl, lo, hi):
    B, n, cx = lo.shape; cy = hi.shape[2]; p = Pl.shape[0]
    lo2, hi2 = lo.reshape(-1, cx), hi.reshape(-1, cy)
    Pt = Pl.transpose(1, 0, 2).reshape(cx, p*cy)
    out = np.empty((lo2.shape[0], p))
    for s in range(0, lo2.shape[0], a.chunk):
        t = (lo2[s:s+a.chunk] @ Pt).reshape(-1, p, cy)
        out[s:s+a.chunk] = (t * hi2[s:s+a.chunk, None, :]).sum(-1)
    return out.reshape(B, n, p)
def onehot(X):
    o = np.zeros(X.shape + (NLEAF,)); np.put_along_axis(o, X[..., None], 1.0, -1)
    return o
def decode(X, Pm, level):
    t = onehot(X)
    for l in range(1, level+1): t = up(Pm[l], t[:, 0::2], t[:, 1::2])
    return t.argmax(-1)
def sample_model(Pm, n, g):
    cur = g.randint(0, V, size=(n, 1))
    for l in range(L, 0, -1):
        c = card(l); D = c*c
        cdf = np.cumsum(Pm[l].reshape(V, D), 1); cdf[:, -1] = 1.0
        u = g.rand(*cur.shape); pick = np.zeros(cur.shape, dtype=np.int64)
        for s in range(V):
            m = cur == s
            if m.any(): pick[m] = np.searchsorted(cdf[s], u[m], side="right")
        pick = np.minimum(pick, D-1)
        nxt = np.empty((n, cur.shape[1]*2), dtype=np.int64)
        nxt[:, 0::2] = pick // c; nxt[:, 1::2] = pick % c
        cur = nxt
    return cur

# ------------------------------------------------------------- planting
w = 2 ** a.level
if a.parity:
    ia, ib, ic = (int(v) for v in a.parity.split(","))
    d = decode(X0, P, a.level)
    # choose, for each target symbol value, a canonical block realisation, then
    # rewrite block c's leaves so that its decoded symbol is s_a XOR s_b. Every
    # rewritten block is a real block drawn from the data, so all productions
    # stay grammatical; only the three-way relation is imposed.
    reps = {}
    for s in range(V):
        idx = np.nonzero(d[:, ic] == s)[0]
        if len(idx): reps[s] = X0[idx[0], ic*w:(ic+1)*w].copy()
    tgt = np.bitwise_xor(d[:, ia], d[:, ib]) % V
    X0 = X0.copy(); miss = 0
    for s in range(V):
        m = tgt == s
        if s in reps: X0[np.ix_(m, range(ic*w, (ic+1)*w))] = reps[s]
        else: miss += int(m.sum())
    print(f"  PLANTED PARITY: block {ic} = block {ia} XOR block {ib}"
          + (f"   ({miss} sequences had no realisation and were left alone)"
             if miss else ""))

# ------------------------------------------------- interaction information
def H(counts):
    p = counts / counts.sum()
    m = p > 0
    return float(-(p[m]*np.log(p[m])).sum())
def triple_stats(codes, i, j, k):
    """(I(A;B), I(A;B|C), I(A;B;C)) by inclusion-exclusion on entropies"""
    A, B_, C = codes[:, i], codes[:, j], codes[:, k]
    hABC = H(np.bincount(A*V*V + B_*V + C, minlength=V**3).astype(float))
    hAB = H(np.bincount(A*V + B_, minlength=V*V).astype(float))
    hAC = H(np.bincount(A*V + C, minlength=V*V).astype(float))
    hBC = H(np.bincount(B_*V + C, minlength=V*V).astype(float))
    hA = H(np.bincount(A, minlength=V).astype(float))
    hB = H(np.bincount(B_, minlength=V).astype(float))
    hC = H(np.bincount(C, minlength=V).astype(float))
    iAB = hA + hB - hAB
    iABgC = hAC + hBC - hABC - hC
    return iAB, iABgC, iAB - iABgC          # last is I(A;B;C)

nb = SEQ >> a.level
codes = decode(X0, P, a.level)
g = np.random.RandomState(a.seed)
nulls = [decode(sample_model(P, len(X0), g), P, a.level) for _ in range(a.boot)]
trips = list(itertools.combinations(range(nb), 3))
print(f"  level {a.level}: {nb} blocks, V={V}, {len(X0)} sequences, "
      f"{len(trips)} triples, {a.boot} bootstrap replicates")
print(f"\n    {'triple':>10}{'I(A;B)':>9}{'I(A;B|C)':>10}{'I(A;B;C)':>10}"
      f"{'null mean':>11}{'z':>8}   verdict")
flagged = []
for (i, j, k) in trips:
    iab, iabc_, ii = triple_stats(codes, i, j, k)
    nul = np.array([triple_stats(w_, i, j, k)[2] for w_ in nulls])
    z = (ii - nul.mean()) / (nul.std(ddof=1) + 1e-12)
    v = ""
    if abs(z) > a.z_flag:
        v = "  <== THREE-WAY structure the model cannot produce"; flagged.append((i, j, k))
    # print only the informative rows plus any flag, or the table is 56 lines
    if abs(z) > 2.0 or (i, j, k) in trips[:4]:
        print(f"    {f'{i},{j},{k}':>10}{iab:>9.4f}{iabc_:>10.4f}{ii:>10.4f}"
              f"{nul.mean():>11.4f}{z:>+8.1f}{v}")
print(f"\n    triples flagged: {len(flagged)} of {len(trips)}"
      + (f"   {flagged}" if flagged else ""))
print("    I(A;B;C) < 0 is synergy: the triple carries structure that no pair")
print("    does. A parity coupling has I(A;B) = 0 for every pair and")
print("    I(A;B;C) = -H(symbol), so it is INVISIBLE to V6, V7 and beta_1 and")
print("    visible only here.")
