#!/usr/bin/env python3
"""SPLITMERGE -- split-merge EM for the hierarchical grammar.

    python3 splitmerge.py --data /tmp/cH --n-train 20000 --rounds 4

WHY
---
Plain EM from the clustered init reaches held-out -17.51 on Corpus B while the
true grammar scores -15.32, and the `--init-true` diagnostic shows the truth is
a fixed point with a very wide basin (an init carrying 10% of the truth
recovers fully). So the gap is basin selection, not the model class, not the
data, and not the optimiser's strength. Deterministic annealing moved it by
0.22 nats; blending the init with uniform (--init-mix 0.9) moved it to -15.87.
Split-merge is the standard remaining move (Petrov et al. 2006 for PCFGs):

  SPLIT   each level-l symbol into two, perturbed, doubling that level's
          alphabet. The child axes of level l+1 double with it.
  EM      let the split model re-fit.
  MERGE   greedily collapse back to V symbols by least cost, where the cost of
          merging two symbols is exactly the Bregman information
              sum_s n_s KL(R_s || R_merged) = N I(children ; B | q(B))
          verified as an identity elsewhere in this work. Crucially the merge
          is over ALL 2V symbols, so it may recombine across the original
          split -- that is the escape mechanism.

Everything is numpy and gradient-free. Held-out log-likelihood is reported at
every stage so a round that hurts is visible.
"""
import json, math, argparse, time, collections
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/tmp/cH")
ap.add_argument("--n-train", type=int, default=20000)
ap.add_argument("--n-val", type=int, default=2000)
ap.add_argument("--warmup", type=int, default=60, help="EM sweeps before round 1")
ap.add_argument("--rounds", type=int, default=4)
ap.add_argument("--split-sweeps", type=int, default=25)
ap.add_argument("--post-sweeps", type=int, default=25)
ap.add_argument("--delta", type=float, default=0.25, help="split perturbation")
ap.add_argument("--init-mix", type=float, default=0.9)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--chunk", type=int, default=4096)
ap.add_argument("--save", default="")
a = ap.parse_args()

M = json.load(open(f"{a.data}/rhm_meta.json"))
L, V, SEQ, NLEAF = M["depth"], M["nsym"], M["seq_len"], M["nleaf"]
KIND = M["pos_kind"]
tr = np.array(json.load(open(f"{a.data}/train_ids.json")), dtype=np.int64)
va = np.array(json.load(open(f"{a.data}/val_ids.json")), dtype=np.int64)
TR = tr[:(len(tr)//SEQ)*SEQ].reshape(-1, SEQ)[:a.n_train]
VA = va[:(len(va)//SEQ)*SEQ].reshape(-1, SEQ)[:a.n_val]
rng = np.random.RandomState(a.seed)

# nsym[l] is the alphabet at level l; nsym[0] = leaves. Split changes these,
# so every routine below reads shapes from the tables rather than a constant.
def up(P, lo, hi):
    B, n, cx = lo.shape; cy = hi.shape[2]; p = P.shape[0]
    lo2, hi2 = lo.reshape(-1, cx), hi.reshape(-1, cy)
    Pt = P.transpose(1, 0, 2).reshape(cx, p * cy)
    out = np.empty((lo2.shape[0], p))
    for s in range(0, lo2.shape[0], a.chunk):
        t = (lo2[s:s+a.chunk] @ Pt).reshape(-1, p, cy)
        out[s:s+a.chunk] = (t * hi2[s:s+a.chunk, None, :]).sum(-1)
    return out.reshape(B, n, p)

def down(P, po, lo, hi):
    B, n, p = po.shape; cx, cy = lo.shape[2], hi.shape[2]
    po2, lo2, hi2 = po.reshape(-1, p), lo.reshape(-1, cx), hi.reshape(-1, cy)
    Pf = P.reshape(p, cx * cy)
    lo_o = np.empty((po2.shape[0], cx)); hi_o = np.empty((po2.shape[0], cy))
    for s in range(0, po2.shape[0], a.chunk):
        t = (po2[s:s+a.chunk] @ Pf).reshape(-1, cx, cy)
        lo_o[s:s+a.chunk] = (t * hi2[s:s+a.chunk, None, :]).sum(-1)
        hi_o[s:s+a.chunk] = (t * lo2[s:s+a.chunk, :, None]).sum(-2)
    return lo_o.reshape(B, n, cx), hi_o.reshape(B, n, cy)

def cnts(P, po, lo, hi, w):
    B, n, p = po.shape; cx, cy = lo.shape[2], hi.shape[2]
    po2 = (po * w[:, None, None]).reshape(-1, p)
    lo2, hi2 = lo.reshape(-1, cx), hi.reshape(-1, cy)
    acc = np.zeros((p, cx * cy))
    for s in range(0, po2.shape[0], a.chunk):
        o = (lo2[s:s+a.chunk, :, None] * hi2[s:s+a.chunk, None, :]).reshape(-1, cx*cy)
        acc += po2[s:s+a.chunk].T @ o
    return acc.reshape(p, cx, cy) * P

def onehot(X):
    o = np.zeros(X.shape + (NLEAF,)); np.put_along_axis(o, X[..., None], 1.0, -1)
    return o
def inside(X, P):
    tabs = [onehot(X)]
    for l in range(1, L+1): tabs.append(up(P[l], tabs[-1][:, 0::2], tabs[-1][:, 1::2]))
    return tabs
def outside(tabs, P):
    B = tabs[0].shape[0]; vroot = P[L].shape[0]
    outs = [None]*(L+1); outs[L] = np.full((B, 1, vroot), 1.0/vroot)
    for l in range(L, 0, -1):
        lo, hi = tabs[l-1][:, 0::2], tabs[l-1][:, 1::2]
        lo_o, hi_o = down(P[l], outs[l], lo, hi)
        ch = np.zeros((B, lo.shape[1]*2, lo.shape[2]))
        ch[:, 0::2] = lo_o; ch[:, 1::2] = hi_o
        outs[l-1] = ch
    return outs
def Zof(t, P): return t[L][:, 0, :].sum(-1) / P[L].shape[0]
def loglik(X, P):
    return float(np.log(np.maximum(Zof(inside(X, P), P), 1e-300)).mean())
def estep(X, P):
    t = inside(X, P); o = outside(t, P); Z = np.maximum(Zof(t, P), 1e-300)
    C = {l: cnts(P[l], o[l], t[l-1][:, 0::2], t[l-1][:, 1::2], 1.0/Z)
         for l in range(1, L+1)}
    return C, float(np.log(Z).mean())
def mstep(C):
    return {l: (C[l] + 1e-9) / (C[l] + 1e-9).sum((1, 2), keepdims=True) for l in C}
def em(P, X, n):
    for _ in range(n):
        C, _ = estep(X, P); P = mstep(C)
    return P

# ---------------------------------------------------------------- init
def card0(l): return NLEAF if l == 1 else V
soft = onehot(TR); P = {}
for l in range(1, L+1):
    c = card0(l)
    hard = soft.argmax(-1); lo, hi = hard[:, 0::2], hard[:, 1::2]
    pid = lo * c + hi; nn_ = lo.shape[1]
    ctx = collections.defaultdict(collections.Counter)
    for row in pid.tolist():
        for i, p_ in enumerate(row):
            if i > 0: ctx[p_][row[i-1]] += 1
            if i+1 < len(row): ctx[p_][row[i+1]] += 1
    cnt_all = collections.Counter(pid.reshape(-1).tolist())
    pairs = sorted(cnt_all); idx = {p_: i for i, p_ in enumerate(pairs)}
    if nn_ >= 4 and len(ctx) >= 2:
        Cm = np.zeros((len(pairs), len(pairs)))
        for p_, d in ctx.items():
            for q, n in d.items():
                if q in idx: Cm[idx[p_], idx[q]] = n
        rs = Cm.sum(1, keepdims=True); rs[rs == 0] = 1
        U, S, _ = np.linalg.svd(Cm/rs, full_matrices=False)
        cum = np.cumsum(S)/max(S.sum(), 1e-300)
        k = max(2, min(int(np.searchsorted(cum, 0.90)+1), len(S)))
        emb = U[:, :k]*S[:k]
    else:
        emb = np.zeros((len(pairs), 2*c))
        for p_ in pairs: emb[idx[p_], p_//c] = 1.0; emb[idx[p_], c + p_ % c] = 1.0
    g = np.random.RandomState(a.seed)
    cen = emb[g.choice(len(pairs), min(V, len(pairs)), replace=False)].astype(float)
    for _ in range(25):
        lab = ((emb[:, None, :]-cen[None])**2).sum(-1).argmin(1)
        for j in range(cen.shape[0]):
            m = lab == j
            if m.any(): cen[j] = emb[m].mean(0)
    tab = np.zeros((V, c, c))
    for p_ in pairs: tab[lab[idx[p_]] % V, p_//c, p_ % c] += cnt_all[p_]
    tab += 1e-6; P[l] = tab / tab.sum((1, 2), keepdims=True)
    if a.init_mix > 0:
        P[l] = (1-a.init_mix)*P[l] + a.init_mix/(c*c)
    soft = up(P[l], soft[:, 0::2], soft[:, 1::2])
    soft = soft / np.maximum(soft.sum(-1, keepdims=True), 1e-300)

PT = {}
for l in range(1, L+1):
    t = np.zeros((V, card0(l), card0(l)))
    for s, r in M["rules_all"][str(l)].items():
        for (x, y) in r: t[int(s), x, y] += 1.0/len(r)
    PT[l] = t
print(f"  L={L} V={V} leaves={NLEAF}   init-mix={a.init_mix}   "
      f"true grammar held-out {loglik(VA, PT):.4f}")
t0 = time.time()
P = em(P, TR, a.warmup)
best = ({l: P[l].copy() for l in P}, loglik(VA, P))
print(f"  warmup {a.warmup} sweeps: held-out {best[1]:.4f}")

# ------------------------------------------------------- split and merge
def split_level(P, l, delta):
    """double level l's alphabet; level l+1's child axes double with it"""
    Pn = {k: v.copy() for k, v in P.items()}
    v0 = P[l].shape[0]
    rows = np.repeat(P[l], 2, axis=0)
    pert = 1.0 + delta * rng.uniform(-1, 1, size=rows.shape)
    rows = rows * pert
    Pn[l] = rows / rows.sum((1, 2), keepdims=True)
    if l < L:                                   # children of level l+1 doubled
        T = P[l+1]
        T2 = np.repeat(np.repeat(T, 2, axis=1), 2, axis=2) / 4.0
        Pn[l+1] = T2 / T2.sum((1, 2), keepdims=True)
    return Pn

def nH(c):
    n = c.sum(-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(c > 0, c*np.log(c/np.maximum(n, 1e-300)), 0.0)
    return -t.sum(-1)

def merge_level(P, C, l, target):
    """greedily merge level-l symbols down to `target`, least Bregman
    information first. The merge ranges over ALL current symbols, so it can
    recombine across the split -- this is the escape."""
    Cl = C[l] + 1e-9
    D = Cl.shape[1]*Cl.shape[2]
    cc = Cl.reshape(Cl.shape[0], D).copy()
    groups = [[s] for s in range(Cl.shape[0])]
    E = nH(cc); paid = 0.0
    while len(groups) > target:
        cost = nH(cc[:, None, :] + cc[None, :, :]) - E[:, None] - E[None, :]
        np.fill_diagonal(cost, np.inf)
        i, j = np.unravel_index(np.argmin(cost), cost.shape)
        i, j = min(i, j), max(i, j)
        paid += float(cost[i, j])
        groups[i] += groups[j]; cc[i] += cc[j]
        del groups[j]; cc = np.delete(cc, j, 0); E = nH(cc)
    lab = np.zeros(Cl.shape[0], dtype=int)
    for gi, g in enumerate(groups):
        for s in g: lab[s] = gi
    Pn = {k: v.copy() for k, v in P.items()}
    pooled = np.zeros((target, Cl.shape[1], Cl.shape[2]))
    for s in range(Cl.shape[0]): pooled[lab[s]] += Cl[s]
    Pn[l] = pooled / pooled.sum((1, 2), keepdims=True)
    if l < L:                                   # collapse child axes above
        T = P[l+1]
        A = np.zeros((T.shape[0], target, T.shape[2]))
        for s in range(T.shape[1]): A[:, lab[s], :] += T[:, s, :]
        B_ = np.zeros((T.shape[0], target, target))
        for s in range(T.shape[2]): B_[:, :, lab[s]] += A[:, :, s]
        Pn[l+1] = B_ / B_.sum((1, 2), keepdims=True)
    return Pn, paid

for r in range(a.rounds):
    for l in range(1, L+1):
        Ps = split_level(P, l, a.delta)
        Ps = em(Ps, TR, a.split_sweeps)
        C, _ = estep(TR, Ps)
        Pm, paid = merge_level(Ps, C, l, V)
        Pm = em(Pm, TR, a.post_sweeps)
        ll = loglik(VA, Pm)
        keep = ll > best[1]
        print(f"  round {r+1} level {l}: split->{2*V} EM merge->{V} "
              f"(paid {paid/len(TR):.3f} nats/seq)   held-out {ll:.4f}"
              f"   {'ACCEPT' if keep else 'reject'}")
        if keep:
            P = Pm; best = ({k: v.copy() for k, v in Pm.items()}, ll)
        else:
            P = {k: v.copy() for k, v in best[0].items()}
P = best[0]
print(f"\n  best held-out {best[1]:.4f}   (true grammar {loglik(VA, PT):.4f}, "
      f"plain EM from this init reaches about -15.87)   [{time.time()-t0:.0f}s]")
if a.save:
    np.savez(a.save, **{f"P{l}": P[l] for l in P}); print(f"  saved {a.save}")
