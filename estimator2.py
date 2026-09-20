#!/usr/bin/env python3
"""ESTIMATOR2 -- grammar estimation with a real clustering, no gradients.

    python3 estimator2.py [--n-train 20000] [--polish 10]

WHY THE PREVIOUS TWO FAILED
---------------------------
1. Joint EM from random init over a 64x64x64 table: recovered 0.3% of rule
   mass, held-out loglik -51.18, lost to the transformer.
2. Bottom-up with round-robin assignment: level 1 saw 658 distinct sibling
   pairs out of 2304 -- real structure -- but the hard round-robin grouping
   by frequency threw it away, and level 2 then saw 4096 of 4096 possible
   pairs, i.e. no structure at all. The hand-off destroyed the signal.

WHAT CHANGES HERE
-----------------
  a) CLUSTER BY CO-OCCURRENCE, not by frequency rank. Two sibling pairs belong
     to the same parent if they appear in the same CONTEXTS -- the same
     neighbouring pairs. That is a spectral clustering of the pair-context
     matrix, which is how a parent symbol is actually identified.
  b) SOFT hand-off. Pass a posterior over parent symbols up to the next level
     instead of a hard label, so an uncertain assignment does not become a
     fact the next level has to live with.
  c) Polish with EM over the whole tree from that initialisation.

Scored on the same held-out split, same local/higher split, wall-clock
reported. No gradient descent anywhere.
"""
import json, math, argparse, time, collections
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/tmp")
ap.add_argument("--n-train", type=int, default=20000)
ap.add_argument("--n-val", type=int, default=2000)
ap.add_argument("--polish", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

M = json.load(open(f"{a.data}/rhm_meta.json"))
L, V, SEQ, NLEAF = M["depth"], M["nsym"], M["seq_len"], M["nleaf"]
KIND = M["pos_kind"]
tr = torch.tensor(json.load(open(f"{a.data}/train_ids.json")), dtype=torch.long)
va = torch.tensor(json.load(open(f"{a.data}/val_ids.json")), dtype=torch.long)
TR = tr[:(len(tr)//SEQ)*SEQ].view(-1, SEQ)[:a.n_train]
VA = va[:(len(va)//SEQ)*SEQ].view(-1, SEQ)[:a.n_val]
np.random.seed(a.seed)
print(f"  L={L} v={V} leaves={NLEAF}  fit {len(TR)} seqs, score {len(VA)}")
t0 = time.time()

def card(l): return NLEAF if l == 1 else V

# ----------------------------------------------------------- initialise
# Soft state: for each node at the current level, a distribution over symbols.
# Level 0 is one-hot on the leaf tokens.
soft = torch.zeros(len(TR), SEQ, NLEAF, dtype=torch.float64)
soft.scatter_(2, TR.unsqueeze(-1), 1.0)
P = {}
for l in range(1, L+1):
    c = card(l)
    hard = soft.argmax(-1)                                  # for counting
    lo, hi = hard[:, 0::2], hard[:, 1::2]
    nn_ = lo.shape[1]
    pid = (lo * c + hi)                                     # observed pair id
    # ---- context matrix: which pairs occur next to which other pairs
    ctx = collections.defaultdict(collections.Counter)
    for b in range(pid.shape[0]):
        row = pid[b].tolist()
        for i, p in enumerate(row):
            if i > 0: ctx[p][row[i-1]] += 1
            if i + 1 < len(row): ctx[p][row[i+1]] += 1
    cnt_all = collections.Counter(pid.reshape(-1).tolist())
    pairs = sorted(cnt_all)
    idx = {p: i for i, p in enumerate(pairs)}
    # The top level has ONE node per sequence and the level below it has two,
    # so there are no (or almost no) neighbouring pairs to build a context
    # profile from. Context clustering is only well-posed where nn_ is large.
    use_ctx = nn_ >= 4 and len(ctx) >= 2
    if use_ctx:
        Cm = np.zeros((len(pairs), len(pairs)))
        for p_, d in ctx.items():
            if p_ not in idx: continue
            for q, n in d.items():
                if q in idx: Cm[idx[p_], idx[q]] = n
        rs = Cm.sum(1, keepdims=True); rs[rs == 0] = 1
        Cn = Cm / rs
        k = max(2, min(V, min(Cn.shape) - 1))
        U, S, _ = np.linalg.svd(Cn, full_matrices=False)
        emb = U[:, :k] * S[:k]
    else:
        # no usable context: fall back to the pair's own child identities,
        # which at least groups pairs sharing a left or right child
        emb = np.zeros((len(pairs), 2 * c))
        for p_ in pairs:
            emb[idx[p_], p_ // c] = 1.0
            emb[idx[p_], c + (p_ % c)] = 1.0
    ncl = max(1, min(V, len(pairs)))
    g = np.random.RandomState(a.seed)
    cen = emb[g.choice(len(pairs), ncl, replace=False)].astype(float)
    lab = np.zeros(len(pairs), dtype=int)
    for _ in range(25):
        d2 = ((emb[:, None, :] - cen[None]) ** 2).sum(-1)
        lab = d2.argmin(1)
        for j in range(ncl):
            m = lab == j
            if m.any(): cen[j] = emb[m].mean(0)
    cnt = cnt_all
    tab = torch.zeros(V, c, c, dtype=torch.float64)
    for p in pairs:
        tab[lab[idx[p]] % V, p // c, p % c] += cnt[p]
    tab = tab + 1e-6
    P[l] = tab / tab.sum((1, 2), keepdim=True)
    print(f"    level {l}: {nn_} nodes/seq, {len(pairs)} distinct pairs -> "
          f"{len(set(lab.tolist()))} clusters "
          f"[{'context' if use_ctx else 'child-identity fallback'}]")
    # ---- SOFT hand-off: posterior over parents given the observed children
    losoft, hisoft = soft[:, 0::2, :], soft[:, 1::2, :]
    soft = torch.einsum('pxy,bnx,bny->bnp', P[l], losoft, hisoft)
    soft = soft / soft.sum(-1, keepdim=True).clamp_min(1e-300)

# ----------------------------------------------------------- inside/outside
def inside(X, P):
    t = torch.zeros(X.shape[0], SEQ, NLEAF, dtype=torch.float64)
    t.scatter_(2, X.unsqueeze(-1), 1.0)
    tabs = [t]
    for l in range(1, L+1):
        lo, hi = tabs[-1][:, 0::2, :], tabs[-1][:, 1::2, :]
        tabs.append(torch.einsum('pxy,bnx,bny->bnp', P[l], lo, hi))
    return tabs

def outside(tabs, P):
    B = tabs[0].shape[0]
    outs = [None]*(L+1)
    outs[L] = torch.full((B, 1, V), 1.0/V, dtype=torch.float64)
    for l in range(L, 0, -1):
        lo, hi = tabs[l-1][:, 0::2, :], tabs[l-1][:, 1::2, :]
        po = outs[l]
        lo_o = torch.einsum('pxy,bnp,bny->bnx', P[l], po, hi)
        hi_o = torch.einsum('pxy,bnp,bnx->bny', P[l], po, lo)
        ch = torch.zeros(B, lo.shape[1]*2, lo.shape[2], dtype=torch.float64)
        ch[:, 0::2, :] = lo_o; ch[:, 1::2, :] = hi_o
        outs[l-1] = ch
    return outs

def loglik(X, P):
    return torch.log((inside(X, P)[L][:, 0, :].sum(-1)/V).clamp_min(1e-300)).mean().item()

def em(X, P):
    tabs = inside(X, P); outs = outside(tabs, P)
    Z = (tabs[L][:, 0, :].sum(-1)/V).clamp_min(1e-300)
    new = {}
    for l in range(1, L+1):
        lo, hi = tabs[l-1][:, 0::2, :], tabs[l-1][:, 1::2, :]
        cnt = torch.einsum('bnp,bnx,bny->pxy', outs[l]/Z[:, None, None], lo, hi)*P[l]
        cnt = cnt + 1e-9
        new[l] = cnt / cnt.sum((1, 2), keepdim=True)
    return new

print(f"    init held-out loglik/seq = {loglik(VA, P):.4f}   "
      f"(joint EM reached -51.18, round-robin -51.45)")
best, bl = {l: P[l].clone() for l in P}, loglik(VA, P)
for it in range(a.polish):
    P = em(TR, P); ll = loglik(VA, P)
    if ll > bl: bl, best = ll, {l: P[l].clone() for l in P}
    print(f"    polish {it+1}: {ll:.4f}")
P = best
t_fit = time.time() - t0

# ----------------------------------------------------------- predict
t1 = time.time()
onehot = torch.zeros(len(VA), SEQ, NLEAF, dtype=torch.float64)
onehot.scatter_(2, VA.unsqueeze(-1), 1.0)
tot = np.zeros(SEQ)
for t in range(SEQ):
    lf = torch.ones(len(VA), SEQ, NLEAF, dtype=torch.float64)
    if t > 0: lf[:, :t, :] = onehot[:, :t, :]
    tt = [lf]
    for l in range(1, L+1):
        lo, hi = tt[-1][:, 0::2, :], tt[-1][:, 1::2, :]
        tt.append(torch.einsum('pxy,bnx,bny->bnp', P[l], lo, hi))
    marg = outside(tt, P)[0][:, t, :]
    pr = marg / marg.sum(-1, keepdim=True).clamp_min(1e-300)
    tot[t] = -torch.log(pr.gather(1, VA[:, t:t+1]).clamp_min(1e-300)).mean().item()
t_pred = time.time() - t1

loc = np.mean([tot[t] for t in range(SEQ) if KIND[t] == "local"])
hig = np.mean([tot[t] for t in range(SEQ) if KIND[t] == "higher"])
print(f"\n  fit {t_fit:.1f}s, inference {t_pred:.1f}s, NO GRADIENTS")
print(f"\n  {'':<26}{'local':>9}{'higher':>9}")
print(f"  {'spectral + soft + EM':<26}{loc:>9.4f}{hig:>9.4f}")
print(f"  {'round-robin bottom-up':<26}{2.5983:>9.4f}{3.8330:>9.4f}")
print(f"  {'joint EM from random':<26}{2.5663:>9.4f}{3.8308:>9.4f}")
print(f"  {'transformer 48k steps':<26}{2.0223:>9.4f}{3.7716:>9.4f}")
print(f"  {'exact floor':<26}{1.6034:>9.4f}{3.5527:>9.4f}")
print(f"\n  vs transformer: local {loc-2.0223:+.4f}  higher {hig-3.7716:+.4f}"
      f"   (negative = constructed predictor wins)")
